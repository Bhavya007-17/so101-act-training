"""Find the largest ACT batch size that fits in 8 GB, empirically.

Builds the real policy, the real preprocessor pipeline, and the real
optimizer the trainer builds, then runs a genuine forward/backward/step at
each candidate batch size and reads peak VRAM back. Guessing from parameter
counts is useless here -- with two 480x640 camera streams the ResNet-18
activations dominate, not the weights.

Mirrors lerobot_train.py's loop exactly:
    batch = next(dl_iter) -> preprocessor(batch) -> policy.forward(batch)
"""

import gc
import sys

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

REPO_ID = "lerobot/svla_so101_pickplace"
CANDIDATES = [2, 4, 6, 8, 12, 16]
NUM_WORKERS = 2

TOTAL_VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)}  ({TOTAL_VRAM_GIB:.2f} GiB)")
print(f"Probing ACT batch sizes: {CANDIDATES}\n")

# ACT predicts action *chunks* (chunk_size=100), so the dataset must be built
# with delta_timestamps -- otherwise `action` comes back (6,) instead of
# (100, 6) and the policy fails on a dimension mismatch. This is exactly what
# make_dataset() does inside the trainer.
_probe_cfg = make_policy_config("act", device="cuda", push_to_hub=False)
_meta = LeRobotDatasetMetadata(REPO_ID)
_delta = resolve_delta_timestamps(_probe_cfg, _meta)
print(f"delta_timestamps keys: {list(_delta) if _delta else None}")
ds = LeRobotDataset(REPO_ID, delta_timestamps=_delta)
print(f"sample action shape: {tuple(ds[0]['action'].shape)}\n")
results = []

for bs in CANDIDATES:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy = opt = pre = loader = None
    try:
        cfg = make_policy_config("act", device="cuda", push_to_hub=False)
        policy = make_policy(cfg=cfg, ds_meta=ds.meta)
        pre, _ = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds.meta.stats)
        policy.train()
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)

        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        it = iter(loader)

        # Two steps: the first allocates, the second reaches steady-state peak.
        for _ in range(2):
            batch = pre(next(it))
            loss, _ = policy.forward(batch)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        pct = 100 * reserved / TOTAL_VRAM_GIB
        print(
            f"  bs={bs:<3} OK    peak_alloc={peak:5.2f} GiB   "
            f"peak_reserved={reserved:5.2f} GiB  ({pct:3.0f}% of card)   loss={loss.item():.3f}"
        )
        results.append((bs, peak, reserved, pct))

    except torch.cuda.OutOfMemoryError:
        print(f"  bs={bs:<3} OOM   <-- ceiling found")
        results.append((bs, None, None, None))
        del policy, opt, pre, loader
        gc.collect()
        torch.cuda.empty_cache()
        break
    except Exception as e:  # noqa: BLE001
        print(f"  bs={bs:<3} ERROR {type(e).__name__}: {e}")
        break
    finally:
        del policy, opt, pre, loader
        gc.collect()
        torch.cuda.empty_cache()

ok = [r for r in results if r[1] is not None]
if not ok:
    print("\nNothing fit. Something is wrong.")
    sys.exit(1)

print()
print("=" * 72)
# Pick the largest batch whose *reserved* footprint stays under ~80% of the
# card. Reserved (not allocated) is what actually has to fit, and headroom
# matters because fragmentation grows over a long run.
safe = [r for r in ok if r[3] <= 80.0]
choice = safe[-1] if safe else ok[0]
print(
    f"RECOMMENDED batch_size = {choice[0]}  "
    f"(peak reserved {choice[2]:.2f} GiB = {choice[3]:.0f}% of {TOTAL_VRAM_GIB:.2f} GiB)"
)
print("Largest batch staying under 80% reserved, leaving headroom for")
print("allocator fragmentation over a long run.")
print("=" * 72)
