"""Reload a trained checkpoint in a FRESH process and run real inference.

A checkpoint that cannot be reloaded is not a completed run. This deliberately
runs as its own process -- nothing is inherited from the trainer, so anything
that only worked because of in-memory state will fail here.

Checks:
  1. The config round-trips from disk.
  2. Weights load, and actually match the safetensors file on disk (we compare
     a checksum against an independent read, so a silently randomly-initialised
     policy cannot pass).
  3. The pre/post-processor pipelines reload from the checkpoint (these carry
     the dataset normalisation stats -- a policy without them is useless).
  4. select_action() produces a finite action of the right shape.
  5. Two identical inputs give identical outputs (deterministic in eval mode).

usage: verify_checkpoint.py <checkpoint_dir> [device]
      e.g. verify_checkpoint.py outputs/act_so101_main/checkpoints/last
           verify_checkpoint.py outputs/act_so101_main/checkpoints/000002000 cpu

`device` defaults to cuda. Passing `cpu` lets you exercise the reload path
while the GPU is busy training, without competing for VRAM.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

REPO_ID = "lerobot/svla_so101_pickplace"
ckpt_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/act_so101_main/checkpoints/last")
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"
model_dir = ckpt_dir / "pretrained_model"

FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f'  -- {detail}' if detail else ''}")
    if not ok:
        FAIL.append(label)


print("=" * 72)
print(f"RELOADING CHECKPOINT (fresh process, pid={__import__('os').getpid()})")
print(f"  {model_dir}")
print("=" * 72)
check("checkpoint dir exists", model_dir.is_dir(), str(model_dir))
if not model_dir.is_dir():
    sys.exit(1)
for f in ["config.json", "model.safetensors", "train_config.json"]:
    check(f"{f} present", (model_dir / f).is_file())

# --- 1. config round-trips -------------------------------------------------
cfg = PreTrainedConfig.from_pretrained(model_dir)
cfg.pretrained_path = model_dir
cfg.device = DEVICE
print(f"\n  policy type      : {cfg.type}")
print(f"  chunk_size       : {cfg.chunk_size}")
print(f"  n_action_steps   : {cfg.n_action_steps}")
print(f"  vision_backbone  : {cfg.vision_backbone}")
check("config round-tripped from disk", cfg.type == "act")

# --- 2. dataset (for metadata + a real batch) ------------------------------
meta = LeRobotDatasetMetadata(REPO_ID)
ds = LeRobotDataset(REPO_ID, delta_timestamps=resolve_delta_timestamps(cfg, meta))

# --- 3. weights load AND match the file on disk ----------------------------
policy = make_policy(cfg=cfg, ds_meta=ds.meta)
policy.eval()
n_params = sum(p.numel() for p in policy.parameters())
print(f"  loaded params    : {n_params:,}")

disk = load_file(model_dir / "model.safetensors")
live = dict(policy.named_parameters())
compared = mismatched = 0
for k, v in disk.items():
    if k in live:
        compared += 1
        if not torch.allclose(live[k].detach().cpu().float(), v.float(), atol=1e-6):
            mismatched += 1
check(
    "live weights match model.safetensors on disk",
    compared > 0 and mismatched == 0,
    f"compared {compared} tensors, {mismatched} mismatched",
)

# --- 4. processors reload (they carry the normalisation stats) -------------
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=model_dir,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)
check("pre/post processors reloaded from checkpoint", preprocessor is not None)

# --- 5. real inference on a real batch -------------------------------------
print()
print("=" * 72)
print("INFERENCE ON A REAL BATCH")
print("=" * 72)
BS = 4
samples = [ds[i] for i in range(BS)]
batch = {}
for k in samples[0]:
    v = samples[0][k]
    batch[k] = (
        torch.stack([s[k] for s in samples]) if isinstance(v, torch.Tensor) else [s[k] for s in samples]
    )

with torch.no_grad():
    proc = preprocessor(batch)
    action = policy.select_action(proc)
    action = postprocessor(action)

print(f"  action shape     : {tuple(action.shape)}")
print(f"  action dtype     : {action.dtype}")
print(f"  action device    : {action.device}")
print(f"  action range     : [{action.min().item():+.4f}, {action.max().item():+.4f}]")
expected_dim = meta.features["action"]["shape"][0]
check("action has expected width", action.shape[-1] == expected_dim, f"got {action.shape[-1]}, expected {expected_dim}")
check("action is finite (no NaN/Inf)", bool(torch.isfinite(action).all().item()))

# --- 6. determinism --------------------------------------------------------
policy.reset()
with torch.no_grad():
    a2 = postprocessor(policy.select_action(preprocessor(batch)))
check("identical input -> identical output", torch.allclose(action, a2, atol=1e-5))

# --- 7. compare against ground-truth action (sanity, NOT a quality gate) ---
gt = torch.stack([s["action"][0] for s in samples]).to(action.device)
mae = (action - gt).abs().mean().item()
print(f"\n  MAE vs dataset action (first chunk step): {mae:.4f}")
print("  NOTE: this is a smoke signal only. Policy quality is explicitly NOT")
print("        a success criterion for this toolchain validation.")

print()
print("=" * 72)
if FAIL:
    print(f"RESULT: FAILED ({len(FAIL)} check(s))")
    for f in FAIL:
        print(f"  - {f}")
    print("=" * 72)
    sys.exit(1)
print("RESULT: CHECKPOINT RELOADS AND INFERS CORRECTLY.")
print("=" * 72)
