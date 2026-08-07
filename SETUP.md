# SETUP.md — LeRobot 0.5.0 on WSL2 + Blackwell (RTX 5070 Laptop)

The install sequence **as actually executed**, including every error hit and what
fixed it. Follow this top to bottom on a fresh machine.

Written for: WSL2 / Ubuntu 24.04 / NVIDIA Blackwell (sm_120) / 8 GB VRAM.
If your GPU is *not* Blackwell, see [Step 3](#step-3--pytorch-first-deliberately)
— the CUDA index choice is the one thing you must re-derive for your hardware.

---

## 0. Verified environment

Everything below was confirmed before anything was installed. **Verify, don't assume** —
several later decisions depend on these exact values.

| Property | Value | How to check |
|---|---|---|
| WSL | 2.7.10.0, kernel `6.18.33.2-microsoft-standard-WSL2` | `wsl.exe --version` |
| Distro | Ubuntu 24.04.4 LTS (noble) | `lsb_release -a` |
| Project FS | `/home/bhavy/P1` → `/dev/sdd`, **ext4** (not a `/mnt/c` 9p mount) | `findmnt -T .` |
| Disk free | 912 GB | `df -hT .` |
| GPU | **NVIDIA GeForce RTX 5070 Laptop GPU** | `nvidia-smi` |
| **Compute capability** | **12.0 → `sm_120` (Blackwell)** | `nvidia-smi --query-gpu=compute_cap --format=csv` |
| VRAM | 8151 MiB (7.96 GiB) | `nvidia-smi` |
| Driver (Windows host) | 592.82, CUDA 13.1 | `nvidia-smi` |
| CPU / RAM | 16 cores / 12 GB + 32 GB swap | `nproc`, `free -h` |
| ffmpeg | 6.1.1-3ubuntu5 (system) | `ffmpeg -version` |
| Python | 3.12.3 | `python3 --version` |

### Driver sanity (WSL-specific, do not skip)

The NVIDIA driver must live **only** on the Windows host. A driver installed
inside WSL breaks the passthrough. Confirm it is clean:

```bash
dpkg -l | grep -iE 'nvidia|cuda'          # expect: no output
lsmod | grep -i nvidia                     # expect: no output
which nvidia-smi                           # expect: /usr/lib/wsl/lib/nvidia-smi
ls /dev/dxg                                # expect: exists (GPU passthrough device)
```

On this machine all four were clean: no `nvidia-*` packages, no kernel modules,
`nvidia-smi` resolving to the WSL passthrough path, `/dev/dxg` present. **Nothing
was installed inside WSL to make the GPU work, and nothing should be.**

---

## Step 1 — Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.bashrc to persist
```

Installed `uv 0.12.2` to `~/.local/bin`.

---

## Step 2 — Scaffold the project

```bash
cd /home/bhavy/P1
uv init --name lerobot-toolchain-validation --python 3.12 --no-workspace
rm -rf src                                # vestigial; this project is not a package
```

`uv init` also runs `git init` and writes a starter `.gitignore` (replaced below).

---

## Step 3 — PyTorch first, deliberately

> **This is the step that matters most on this hardware.** Do it *before*
> installing LeRobot so no transitive resolve gets to pick the torch wheel.

### Why `cu128` specifically

The RTX 5070 Laptop is **Blackwell, compute capability 12.0 (`sm_120`)**.
PyTorch wheels built against **CUDA 12.6 and earlier contain no `sm_120` kernels**.
They do not fall back gracefully — you get
`no kernel image is available for execution on the device`, or worse, a harness
that swallows it and silently runs on CPU. **`cu128` is the floor.** The host
driver reports CUDA 13.1, so `cu128` / `cu129` / `cu130` are all driver-compatible;
`cu128` is the most well-trodden Blackwell path.

### Why torch 2.10 / torchvision 0.25 / torchcodec 0.10

LeRobot 0.5.0 declares `torch<2.11`, `torchvision<0.26`, `torchcodec<0.11`.
Those three caps line up **exactly** at torch 2.10 / tv 0.25 / tc 0.10, which is
strong evidence that triple is the top of its intended range. It also satisfies
TorchCodec's published compatibility table (`torchcodec 0.10 ↔ torch 2.10`).

> **TorchCodec declares no `torch` dependency in its metadata.** The pairing is
> documented in its README but *not enforced by the resolver*, so a resolver will
> happily install an ABI-incompatible combination that only explodes at import.
> Pin all three explicitly.

### `pyproject.toml`

The `explicit = true` on the index is load-bearing: it means **nothing** resolves
from the PyTorch index unless named in `[tool.uv.sources]`, which is what stops a
transitive resolve from quietly substituting a CPU-only PyPI wheel.

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch==2.10.0",
    "torchvision==0.25.0",
    "torchcodec==0.10.0",
    "lerobot==0.5.0",
]

[tool.uv]
package = false

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
# torchcodec deliberately NOT here -- see Error 1.
```

```bash
uv sync
```

### Verify immediately — and check more than `is_available()`

```bash
uv run python scripts/verify_cuda.py
```

`torch.cuda.is_available()` returning `True` **does not** prove Blackwell works —
it is true even for a wheel whose compiled arch list stops at `sm_90`.
[`scripts/verify_cuda.py`](scripts/verify_cuda.py) additionally asserts:

- `torch.version.cuda >= 12.8`
- **`sm_120 ∈ torch.cuda.get_arch_list()`**  ← the check that catches a Blackwell mismatch
- the *device's own* capability is in that arch list
- real cuBLAS matmul (verified against CPU), cuDNN conv2d fwd **and** bwd, bf16 autocast

Observed:

```
torch 2.10.0+cu128, CUDA 12.8, cuDNN 91002
arch list: ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
cuda:0 NVIDIA GeForce RTX 5070 Laptop GPU, sm_120, 7.96 GiB, 36 SMs
RESULT: CUDA stack VERIFIED -- real GPU, real kernels, sm_120 present.
```

**Do not proceed past a failure here.**

---

## Step 4 — LeRobot 0.5.0

```bash
uv add "lerobot==0.5.0"
```

Resolved in ~17 s. `numpy` was downgraded 2.5.1 → 2.2.6 (LeRobot pins `numpy<2.3`).
**torch/torchvision/torchcodec were untouched** — the `explicit` index pin held.
Re-verify anyway:

```bash
uv pip list | grep -E '^(torch|torchvision|torchcodec) '
# torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / torchcodec 0.10.0
uv run python scripts/verify_cuda.py      # still passes
```

### Extras required for ACT: **none**

Read from the installed source rather than trusted to memory —
`lerobot/policies/act/modeling_act.py` imports only `einops`, `numpy`, `torch`,
`torchvision`, all of which are core dependencies. LeRobot 0.5.0's extras
(`aloha`, `pusht`, `smolvla`, …) are for simulators, other policies and hardware
backends. **Plain `lerobot==0.5.0` is sufficient to train ACT on a real dataset.**

### The two anticipated WSL failure modes — what actually happened

**`evdev` — did not occur.** `pynput` (a *core* LeRobot dep) does pull
`evdev>=1.3` on Linux, so the risk is real, but `evdev 1.9.3` ships prebuilt
manylinux wheels, so nothing was compiled. Ubuntu 24.04 also already ships
`linux-libc-dev` (providing `/usr/include/linux/input.h`), so the source-build
path would have worked as a fallback. **No workaround needed.** If you hit it on
another distro: `sudo apt install linux-libc-dev build-essential python3-dev`.

**TorchCodec — did occur, but not for the documented reason.** See Error 1.

---

## Errors hit, and the fixes

### Error 1 — TorchCodec fails to import: `libnppicc.so.12: cannot open shared object file`

**Symptom.** After installing `torchcodec==0.10.0+cu128` from the PyTorch CUDA
index, any import fails:

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment. ...
[start of libtorchcodec loading traceback]
FFmpeg version 8: OSError: libnppicc.so.12: cannot open shared object file
FFmpeg version 7: OSError: libnppicc.so.12: cannot open shared object file
... (identical for 6, 5, 4)
```

**Do not be misled by the error text.** It blames FFmpeg first, and the loader
tries FFmpeg 4–8 in turn so you get five near-identical tracebacks. FFmpeg was
fine the whole time. The real cause is the last line of each: **`libnppicc.so.12`**
— NVIDIA Performance Primitives (image colour conversion). The `+cu128` TorchCodec
build dynamically links NPP for GPU colour conversion, but **neither torch 2.10 nor
the TorchCodec wheel installs the NPP runtime**, and the wheel does not declare it.

**Root cause was self-inflicted:** I had sourced `torchcodec` from the `cu128`
index for ABI consistency with torch. LeRobot declares plain `torchcodec` with no
index, so the **PyPI (CPU) build is the mainline path.**

**Fix** — remove `torchcodec` from `[tool.uv.sources]` so it comes from PyPI:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
# torchcodec intentionally absent -> resolves from PyPI (CPU build, no NPP dep)
```

```bash
uv sync     # torchcodec 0.10.0+cu128 -> 0.10.0
```

CPU decode is also the *correct* choice on an 8 GB card — video decoding should not
compete with training for VRAM, and 16 CPU cores absorb it easily (measured
`data_s ≈ 0.005–0.075 s` vs `updt_s ≈ 0.5–0.8 s`, i.e. never the bottleneck).

**Alternative if you specifically want NVDEC GPU decode:** keep the `+cu128` build
and add an explicit `nvidia-npp-cu12` dependency. Not done here; not needed.

**Verify the fix:**

```bash
uv run python -c "
from torchcodec.decoders import VideoDecoder
from torchcodec._core.ops import get_ffmpeg_library_versions
print(get_ffmpeg_library_versions())"
# {'libavcodec': [60,31,102], ..., 'ffmpeg_version': '6.1.1-3ubuntu5'}
```

### Error 2 — `lerobot-train --help` crashes

**Symptom.**

```
ValueError: unsupported format character ' ' (0x20) at index 142
  ... argparse.py in _expand_help: return self._get_help_string(action) % params
```

**Cause.** `draccus` turns **inline dataclass field comments** into argparse help
text. `lerobot/configs/train.py:54` contains the comment
`# This disables cudnn.benchmark and may reduce training speed by ~10-20%.`
Argparse then `%`-interpolates that string and chokes on the literal `%`.
This is an upstream bug — `--help` is simply unusable in 0.5.0.

**Workaround.** Read the CLI surface from the config dataclasses, which is where
draccus generates it from anyway (`--<field>`, nested as `--<parent>.<child>`):

| File | Provides |
|---|---|
| `lerobot/configs/train.py` → `TrainPipelineConfig` | `--batch_size`, `--steps`, `--num_workers`, `--save_freq`, `--log_freq`, `--eval_freq`, `--output_dir`, `--job_name`, `--seed`, `--resume`, `--save_checkpoint` |
| `lerobot/configs/default.py` → `DatasetConfig`, `WandBConfig` | `--dataset.repo_id`, `--dataset.root`, `--dataset.episodes`, `--dataset.revision`, `--dataset.video_backend`, `--wandb.enable` |
| `lerobot/configs/policies.py` → `PreTrainedConfig` | `--policy.type`, `--policy.device`, `--policy.use_amp`, `--policy.push_to_hub`, `--policy.repo_id` |
| `lerobot/policies/act/configuration_act.py` → `ACTConfig` | `--policy.chunk_size`, `--policy.n_action_steps`, `--policy.vision_backbone`, `--policy.dim_model`, `--policy.kl_weight`, … |

> **Gotcha:** `policy.push_to_hub` defaults to **`True`**, and `validate()` raises
> if it is true without a `policy.repo_id`. You must pass
> `--policy.push_to_hub=false` for any local-only run.

### Error 3 — `RuntimeError: Tensors must have same number of dimensions: got 3 and 2`

Hit while writing the VRAM probe, **not** a LeRobot bug — my own harness error,
recorded because it is an easy trap.

**Cause.** ACT predicts action *chunks* (`chunk_size=100`), so the dataset must be
built with `delta_timestamps`; otherwise `action` comes back `(6,)` instead of
`(100, 6)`. `lerobot-train` does this inside `make_dataset()`; a hand-built
`LeRobotDataset(repo_id)` does not.

**Fix.**

```python
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

meta  = LeRobotDatasetMetadata(REPO_ID)
delta = resolve_delta_timestamps(policy_cfg, meta)
ds    = LeRobotDataset(REPO_ID, delta_timestamps=delta)   # action -> (100, 6)
```

### Error 4 — spurious TorchCodec decode failure in the batch-size probe

**Symptom.** `RuntimeError: Could not push packet to decoder: Invalid data found
when processing input`, raised from a DataLoader worker at `bs=12`.

**This was a false alarm, and worth writing down so nobody re-debugs it.**
It is *not* a corrupt dataset and *not* a real batch-size ceiling:

- `bs=12` and `bs=16` both run fine in a **fresh process**.
- `ffmpeg -v error -i <file> -f null -` decodes the videos clean.

The probe was constructing a new policy + DataLoader (with worker processes) for
each candidate batch size **within a single process** and never tearing the worker
pools down, so file handles and RAM accumulated (this box has only 12 GB RAM).
**Lesson: probe each batch size in its own process.**

---

## Step 5 — Lock, commit, and prove the lock reproduces

```bash
uv lock
git add -A && git commit -m "..."      # uv.lock IS committed
```

Then rebuild from the lockfile **alone** and re-run the CUDA check. `--frozen`
forbids re-resolution, so this installs exactly what the lock records:

```bash
UV_PROJECT_ENVIRONMENT=.venv-lockcheck uv sync --frozen
.venv-lockcheck/bin/python scripts/verify_cuda.py
```

Result: identical `torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / torchcodec 0.10.0`,
`sm_120` verified, TorchCodec live, LeRobot backend `torchcodec`.

> Caveat, stated honestly: that rebuild used uv's local wheel cache, so it proves
> the pins are exact and hash-identical — it is not a cold-network download test.

---

## Step 6 — Dataset

```bash
uv run python scripts/inspect_dataset.py
```

`lerobot/svla_so101_pickplace` — 50 episodes, 11,939 frames @ 30 fps, two 480×640
cameras (`observation.images.up`, `observation.images.side`), 6-DoF state/action.
Rationale in [DECISIONS.md](DECISIONS.md).

> **LeRobot 0.5.0 requires dataset format `v3.0`** (`CODEBASE_VERSION = "v3.0"` in
> `lerobot/datasets/lerobot_dataset.py`). `LeRobotDataset` defaults `revision` to
> `"v3.0"`, so it fetches that git tag/branch of the dataset repo. Many older
> well-known LeRobot datasets are v2.0/v2.1 and **will not load**. Check for a
> `v3.0` ref before choosing a dataset:
> ```python
> from huggingface_hub import HfApi
> refs = HfApi().list_repo_refs("<repo_id>", repo_type="dataset")
> print([b.name for b in refs.branches], [t.name for t in refs.tags])
> ```

---

## Step 7 — Training

```bash
# smoke: prove the loop closes and a checkpoint lands
uv run lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/smoke --job_name=act_so101_smoke \
  --batch_size=2 --steps=100 --save_freq=50 --log_freq=10 \
  --num_workers=2 --wandb.enable=false
```

```bash
# real run: bs=12 + bf16 AMP  (see DECISIONS.md for both choices)
ACCELERATE_MIXED_PRECISION=bf16 uv run lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_so101_main --job_name=act_so101_main \
  --batch_size=12 --steps=10000 --save_freq=2000 --log_freq=200 \
  --num_workers=4 --seed=1000 --wandb.enable=false
```

### ⚠️ `--policy.use_amp=true` does nothing during training in 0.5.0

Verified by reading the installed source:

- `use_amp` is **never referenced** in `lerobot/scripts/lerobot_train.py`.
- `Accelerator(...)` (`lerobot_train.py:183`) is constructed **without**
  `mixed_precision=`, so the `accelerator.autocast()` in `update_policy()` is a
  no-op and training runs in fp32.
- `use_amp` is only honoured in `lerobot_eval.py` / `lerobot_record.py` (inference).

**To actually get AMP during training, set Accelerate's own env var:**

```bash
ACCELERATE_MIXED_PRECISION=bf16 uv run lerobot-train ...
```

Measured at `bs=12`, 100 steps: **104 s → 91 s (−13%)** and
**6628 MiB → 5964 MiB peak (−664 MiB)**. bf16 needs no gradient scaler and is
well supported on Blackwell.

---

## Step 8 — Verify the checkpoint reloads

```bash
uv run python scripts/verify_checkpoint.py outputs/act_so101_main/checkpoints/last
```

Runs in a **fresh process** and asserts the config round-trips, the live weights
byte-match `model.safetensors` on disk (so a silently re-initialised policy cannot
pass), the pre/post-processors reload (they carry the normalisation stats),
and `select_action()` yields a finite, correctly-shaped, deterministic action.

---

## Reproduce from scratch

```bash
git clone <this repo> && cd <repo>
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python scripts/verify_cuda.py         # MUST pass before anything else
uv run python scripts/inspect_dataset.py
```

## Version summary

| Package | Version | Source |
|---|---|---|
| uv | 0.12.2 | astral.sh installer |
| Python | 3.12 | uv-managed |
| torch | **2.10.0+cu128** | `download.pytorch.org/whl/cu128` |
| torchvision | **0.25.0+cu128** | `download.pytorch.org/whl/cu128` |
| torchcodec | **0.10.0** (CPU) | PyPI — *deliberately not cu128* |
| lerobot | **0.5.0** | PyPI, no extras |
| numpy | 2.2.6 | PyPI (LeRobot pins `<2.3`) |
| evdev | 1.9.3 | PyPI wheel (no source build) |
| ffmpeg | 6.1.1-3ubuntu5 | Ubuntu system package |
