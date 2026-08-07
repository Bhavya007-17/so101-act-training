# DECISIONS.md

Every non-obvious call made during this toolchain validation, and why.
Companion to [SETUP.md](SETUP.md), which records *what was executed*; this
records *why it was chosen*.

**Framing:** the goal is a validated toolchain, not a good policy. Success is
"a training run finished cleanly and the environment reproduces from a lockfile."
Loss curve quality is explicitly **not** a success criterion. Every decision below
is optimised for *attribution* — so that a future failure can be blamed on the
data rather than the environment.

---

## 1. Dataset: `lerobot/svla_so101_pickplace`

**Chosen.** 50 episodes · 11,939 frames @ 30 fps · 82 MiB · two 480×640 RGB
cameras (`observation.images.up`, `observation.images.side`) · 6-DoF state and
action · single task, *"pink lego brick into the transparent box"*.

The binding constraint turned out not to be popularity but **format version**:
LeRobot 0.5.0 hard-requires dataset `CODEBASE_VERSION = "v3.0"`, and
`LeRobotDataset` defaults `revision="v3.0"`, so it fetches that specific git ref.
Most of the famous LeRobot datasets (`lerobot/pusht`, the ALOHA sim sets) are
v2.0/v2.1 and simply will not load — which silently eliminates most of what a
search or a stale tutorial would suggest. `svla_so101_pickplace` carries both a
`v2.1` and a **`v3.0` tag**, so it loads on the default revision with no pinning.
Against the stated priorities it wins on all three: it is **(a) known-good with
ACT** — it is the official `lerobot`-org dataset used in the upstream SO-101
tutorial, and ACT is the reference policy for it; **(b) small** — 82 MiB total
downloads in ~10 s and the 11,939 frames give ~995 steps per epoch at batch 12,
so a 10k-step run covers ~10 epochs in about an hour; and **(c) SO-101 native** —
the 6-DoF observation/action space (5 joints + gripper) and the two-camera
up/side rig are exactly the topology of the hardware being fabricated in
September, so the feature schema validated here is the schema that real data will
have to match. That last point is the whole reason to prefer it: when real SO-101
data eventually fails to train, the *shape* of the pipeline will already be known-good.

**Rejected alternatives.** The `lerobot/aloha_*` sim datasets are the most
battle-tested with ACT but are 14-DoF bimanual, v2.1-era, and would validate a
feature schema that will never match this hardware. `lerobot/pusht` is tiny and
fast but is 2-DoF state-only with no camera rig worth speaking of, so it would
not exercise the video decode path — the single most fragile part of this install.
The many community `so101_*` datasets (`bjb7/so101_pen_mug`, etc.) are the right
robot but have unknown provenance, unverified v3.0 refs, and no published
ACT results, which would reintroduce exactly the "is it my data or my setup?"
ambiguity this exercise exists to remove.

---

## 2. Batch size: **12**

Measured, not guessed. Two 480×640 camera streams mean ResNet-18 **activations**
dominate VRAM, not the 52M parameters — so a parameter-count estimate is useless
here. Each candidate was run in its **own process** (see SETUP.md Error 4) doing
real forward/backward/optimizer steps:

| batch | peak reserved | % of 7.96 GiB | verdict |
|---|---|---|---|
| 2 | 1.61 GiB | 20% | wasteful |
| 4 | 2.29 GiB | 29% | |
| 6 | 3.06 GiB | 38% | |
| 8 | 4.11 GiB | 52% | safe fallback |
| **12** | **5.62 GiB** | **71%** | **chosen** |
| 16 | 7.30 GiB | 92% | too tight |

**Why 12 and not 16.** 16 fits in a clean-room benchmark, but 92% occupancy
leaves no room for allocator fragmentation across a 10,000-step run, and an OOM
at step 9,000 costs an hour. 71% is the largest step on the ladder that keeps
real headroom. With bf16 AMP the actual observed peak came down further to
**6096 MiB (75% of the card, nvidia-smi process total)**.

Note that `nvidia-smi` (6096 MiB) reads higher than torch's
`max_memory_reserved()` (5.62 GiB ≈ 5755 MiB) because the CUDA context, cuDNN
workspaces and kernel code live outside the caching allocator. **The nvidia-smi
number is the one that has to fit in 8 GB**, so that is what is reported.

---

## 3. Mixed precision: **bf16, via `ACCELERATE_MIXED_PRECISION`, not `--policy.use_amp`**

**`--policy.use_amp=true` is inert for training in LeRobot 0.5.0.** Verified by
reading the installed source: `use_amp` is never referenced in
`lerobot/scripts/lerobot_train.py`, and `Accelerator(...)` (line 183) is built
**without** `mixed_precision=`, so the `accelerator.autocast()` inside
`update_policy()` is a no-op. The flag only takes effect in `lerobot_eval.py` and
`lerobot_record.py` (inference paths).

This is exactly the class of bug this exercise is meant to catch: a documented
flag that appears to work, changes nothing, and would have left me reporting
"AMP enabled" over an fp32 run.

Enabling it through Accelerate's own channel does work, measured at bs=12/100 steps:

| | wall clock | peak VRAM |
|---|---|---|
| fp32 (baseline) | 104 s | 6628 MiB |
| **bf16 AMP** | **91 s (−13%)** | **5964 MiB (−664 MiB)** |

**bf16 over fp16** because it needs no gradient scaler (fewer moving parts, no
scale-overflow failure mode) and has native Blackwell support. The gain is
modest because ACT at this resolution is substantially bandwidth- and
dataloader-bound rather than pure tensor-core-bound, but it is free.

---

## 4. Decode backend: **TorchCodec 0.10.0 (CPU build, PyPI), on FFmpeg 6.1.1**

**Active backend: `torchcodec`. This is the mainline path — no silent pyav fallback.**

Verified **empirically, not from config**. This matters because LeRobot picks its
backend with `importlib.util.find_spec("torchcodec")`
(`lerobot/datasets/video_utils.py:118`), which only checks the module is *findable
on disk* — a **broken** torchcodec still reports as `"torchcodec"` and then dies
inside a dataloader worker. So `scripts/inspect_dataset.py` wraps the actual
decode functions and counts calls:

```
decode_video_frames_torchcodec   calls: 18
decode_video_frames_torchvision  calls: 0     (the pyav path)
==> TorchCodec did the decoding. On the mainline path.
```

**CPU build rather than `+cu128`, deliberately.** The `+cu128` TorchCodec wheel
fails to import outright (`libnppicc.so.12`, SETUP.md Error 1). Even once fixable
via an explicit `nvidia-npp-cu12` pin, CPU decode is the better choice here: (a)
it is what plain `pip install lerobot` resolves to, i.e. the mainline path; (b) on
an 8 GB card, video decode must not compete with training for VRAM; (c) 16 CPU
cores absorb it comfortably — measured `data_s ≈ 0.014–0.075 s` against
`updt_s ≈ 0.38–0.46 s`, so decode is **never** the bottleneck.

Dataset videos are **AV1** (`yuv420p`, 640×480, 30 fps), decoded by system FFmpeg
6.1.1 (`libavcodec 60.31.102`), inside TorchCodec's supported FFmpeg range of [4, 8].

---

## 5. Torch stack: `torch 2.10.0+cu128` / `torchvision 0.25.0+cu128` / `torchcodec 0.10.0`

**`cu128` is forced by the hardware.** RTX 5070 Laptop is Blackwell, `sm_120`.
PyTorch builds against CUDA ≤12.6 contain **no `sm_120` kernels**. The host driver
(592.82, CUDA 13.1) would also permit cu129/cu130; cu128 is the most well-trodden
Blackwell path and is what the ecosystem builds against.

**The version triple is inferred from LeRobot's own caps.** LeRobot 0.5.0 declares
`torch<2.11`, `torchvision<0.26`, `torchcodec<0.11`. Those three ceilings coincide
exactly at torch 2.10 / tv 0.25 / tc 0.10, which is strong evidence that is the
top of its tested range — so the newest compatible triple was taken rather than
the oldest.

**All three are pinned explicitly** because **TorchCodec declares no `torch`
dependency in its metadata**. The pairing (`torchcodec 0.10 ↔ torch 2.10`) is
documented only in its README, so a resolver is free to produce an
ABI-incompatible combination that fails at import. `[[tool.uv.index]]` with
`explicit = true` additionally guarantees nothing else can drift onto the CUDA
index or off it.

---

## 6. Deviations from the official docs

| Deviation | Rationale |
|---|---|
| `torchcodec` from PyPI, not the CUDA index | The `+cu128` build has an undeclared NPP runtime dependency and cannot import. PyPI CPU build is what `pip install lerobot` gives you anyway. |
| CLI flags read from config dataclasses, not `--help` | `lerobot-train --help` crashes in 0.5.0 (draccus/argparse `%` bug). The dataclasses are the source draccus generates from, so they are strictly more authoritative. |
| AMP via `ACCELERATE_MIXED_PRECISION=bf16`, not `--policy.use_amp` | The documented flag is a no-op in the 0.5.0 training script. |
| No LeRobot extras installed | ACT's module imports only core deps. Extras are for simulators/hardware/other policies. Verified by reading the installed source. |
| `--policy.push_to_hub=false` passed explicitly | Defaults to `True`; `validate()` raises without a `repo_id`. Required for any local-only run. |
| `num_workers=4` (not higher, despite 16 cores) | RAM, not CPU, is the constraint: 12 GB total with ~3.2 GB free under load. Decode is already not the bottleneck, so more workers buy nothing and risk swapping. |

---

## 7. Run length: 10,000 steps

Long enough to be a meaningful toolchain test — ~10 epochs over 11,939 frames,
five checkpoint writes, sustained thermal load, and enough elapsed time for a
dataloader or memory leak to surface — while still finishing in about an hour.
Going longer would only buy policy quality, which is explicitly not the goal.

`--seed=1000` (the LeRobot default) is passed explicitly so the run is restatable.
