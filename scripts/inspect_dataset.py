"""Load the chosen dataset, dump its schema, and push one real batch through a DataLoader.

Two things this proves that a training log would not:
  1. The decode path is genuinely TorchCodec. LeRobot picks its backend with
     importlib.util.find_spec("torchcodec"), which only checks the module is
     findable on disk -- a broken torchcodec still reports as "torchcodec".
     So we wrap the actual decode function and assert it gets called.
  2. A batch materialises with the shapes/dtypes ACT expects, before we ever
     hand control to the trainer.
"""

import sys

import torch
from torch.utils.data import DataLoader

import lerobot.datasets.video_utils as video_utils
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/svla_so101_pickplace"

# --- Instrument the decoders BEFORE the dataset is constructed -------------
# We count calls so we can tell, empirically, which backend did the work.
CALLS = {"torchcodec": 0, "pyav_or_torchvision": 0}

_real_tc = video_utils.decode_video_frames_torchcodec
_real_tv = video_utils.decode_video_frames_torchvision


def _tc(*a, **k):
    CALLS["torchcodec"] += 1
    return _real_tc(*a, **k)


def _tv(*a, **k):
    CALLS["pyav_or_torchvision"] += 1
    return _real_tv(*a, **k)


video_utils.decode_video_frames_torchcodec = _tc
video_utils.decode_video_frames_torchvision = _tv
# LeRobotDataset calls decode_video_frames, which dispatches to the above by
# name from within this same module, so patching the module attrs is enough.

print("=" * 72)
print(f"LOADING  {REPO_ID}")
print("=" * 72)
ds = LeRobotDataset(REPO_ID)
meta = ds.meta

print(f"  codebase version : {meta._version}")
print(f"  revision fetched : {ds.revision}")
print(f"  video backend    : {ds.video_backend}   <-- configured")
print(f"  fps              : {meta.fps}")
print(f"  episodes         : {meta.total_episodes}")
print(f"  frames           : {meta.total_frames}")
print(f"  tasks            : {meta.total_tasks}")
try:
    print(f"  task list        : {list(meta.tasks.index)[:5]}")
except Exception:
    pass

print()
print("=" * 72)
print("FEATURE SCHEMA")
print("=" * 72)
for k, v in meta.features.items():
    print(f"  {k:<38} dtype={v['dtype']:<10} shape={v['shape']}")

print()
print("=" * 72)
print("CAMERA KEYS")
print("=" * 72)
print(f"  camera_keys : {meta.camera_keys}")
print(f"  video_keys  : {meta.video_keys}")
print(f"  image_keys  : {getattr(meta, 'image_keys', [])}")

print()
print("=" * 72)
print("SINGLE SAMPLE (exercises the video decode path)")
print("=" * 72)
s = ds[0]
for k, v in sorted(s.items()):
    if isinstance(v, torch.Tensor):
        print(f"  {k:<38} {str(tuple(v.shape)):<22} {v.dtype}")
    else:
        print(f"  {k:<38} {type(v).__name__}: {v}")

print()
print("=" * 72)
print("ONE BATCH THROUGH A DATALOADER")
print("=" * 72)
BATCH = 8
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
batch = next(iter(loader))
for k, v in sorted(batch.items()):
    if isinstance(v, torch.Tensor):
        rng = ""
        if v.is_floating_point():
            rng = f"  min={v.min().item():+.3f} max={v.max().item():+.3f}"
        print(f"  {k:<38} {str(tuple(v.shape)):<22} {str(v.dtype):<16}{rng}")
    else:
        print(f"  {k:<38} {type(v).__name__}")

# Move it to the GPU the way the trainer will.
dev = torch.device("cuda")
moved = {k: (v.to(dev, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
img_key = meta.camera_keys[0]
print(f"\n  moved to GPU OK -- {img_key} now on {moved[img_key].device}")

print()
print("=" * 72)
print("TRAINING-SHAPED BATCH (with delta_timestamps, as the trainer builds it)")
print("=" * 72)
# The raw dataset above returns action as (6,). ACT predicts action *chunks*
# (chunk_size=100), so the trainer builds the dataset with delta_timestamps via
# make_dataset(). Without them the policy fails on a dimension mismatch, so this
# is the shape that actually matters.
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402

act_cfg = make_policy_config("act", device="cuda", push_to_hub=False)
delta = resolve_delta_timestamps(act_cfg, meta)
print(f"  delta_timestamps keys : {list(delta) if delta else None}")
print(f"  chunk_size            : {act_cfg.chunk_size}")

ds_chunked = LeRobotDataset(REPO_ID, delta_timestamps=delta)
loader_c = DataLoader(ds_chunked, batch_size=BATCH, shuffle=True, num_workers=0)
bc = next(iter(loader_c))
for k in ("action", "observation.state", meta.camera_keys[0]):
    print(f"  {k:<30} {tuple(bc[k].shape)}")
assert bc["action"].shape[1] == act_cfg.chunk_size, "action chunk missing -- trainer would fail"
print(f"  ==> action is chunked ({BATCH}, {act_cfg.chunk_size}, 6). Trainer-ready.")

print()
print("=" * 72)
print("DECODE BACKEND -- EMPIRICAL, NOT CONFIGURED")
print("=" * 72)
print(f"  decode_video_frames_torchcodec   calls: {CALLS['torchcodec']}")
print(f"  decode_video_frames_torchvision  calls: {CALLS['pyav_or_torchvision']}  (pyav path)")

ok = CALLS["torchcodec"] > 0 and CALLS["pyav_or_torchvision"] == 0
if ok:
    print("\n  ==> TorchCodec did the decoding. On the mainline path.")
else:
    print("\n  ==> !!! WARNING !!! Decoding did NOT go through TorchCodec.")
    print("      This means a silent fallback. Do not trust downstream timings.")
    sys.exit(1)

print("=" * 72)
print("DATASET VALIDATED -- safe to invoke the trainer.")
print("=" * 72)
