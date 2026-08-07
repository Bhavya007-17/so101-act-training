# RESULTS.md — measured outcome

Closing numbers from the validation run. See [SETUP.md](SETUP.md) for how the
environment was built and [DECISIONS.md](DECISIONS.md) for why each knob was set.

---

## The run

| | |
|---|---|
| Command | `ACCELERATE_MIXED_PRECISION=bf16 lerobot-train --policy.type=act …` |
| Dataset | `lerobot/svla_so101_pickplace` (SO-101, 50 eps, 11,939 frames, 2×480×640) |
| Policy | ACT, **51,597,190** params, ResNet-18 (ImageNet-pretrained) backbone |
| Steps | **10,000 / 10,000** — completed, `exit code 0` |
| Batch size | 12 (effective 12 × 1) |
| Precision | bf16 AMP via Accelerate |
| Epochs covered | 10.05 (120K samples) |

## Headline numbers

| Metric | Value |
|---|---|
| **Wall clock** | **4,029 s — 67.2 min** |
| **Peak VRAM** | **6,664 MiB — 81.8% of 8,151 MiB** (nvidia-smi, 5,654 samples @ 0.5 s) |
| Mean VRAM | 6,014 MiB |
| **Final loss** | **0.161** (grad norm 12.96) |
| Throughput | 2.48 step/s sustained |
| Step time | `updt_s ≈ 0.35–0.38 s`, `data_s ≈ 0.014–0.034 s` |
| Checkpoints | 5 (every 2,000 steps) + `last` |
| Checkpoint size | 198 MB weights / 394 MB with optimizer state |

## Loss trajectory

| step | loss | grad norm |
|---|---|---|
| 200 | 5.918 | 125.29 |
| 1,000 | 1.631 | 54.14 |
| 2,000 | 0.931 | 41.29 |
| 4,000 | 0.358 | 23.55 |
| 6,000 | 0.237 | 17.71 |
| 8,000 | 0.186 | 14.55 |
| **10,000** | **0.161** | **12.96** |

Monotonic decrease, no divergence, no NaN. **This is not a success criterion** —
it is reported only as evidence the optimisation loop is wired correctly.

## Checkpoint reload (fresh process, CUDA)

```
[PASS] config.json / model.safetensors / train_config.json present
[PASS] config round-tripped from disk
[PASS] live weights match model.safetensors on disk -- compared 153 tensors, 0 mismatched
[PASS] pre/post processors reloaded from checkpoint
[PASS] action has expected width -- got 6, expected 6
[PASS] action is finite (no NaN/Inf)
[PASS] identical input -> identical output
RESULT: CHECKPOINT RELOADS AND INFERS CORRECTLY.
```

Separately confirmed the inference genuinely ran on the GPU (rather than trusting
a device string): policy parameters `{cuda:0}`, preprocessed images `cuda:0`, raw
policy output `cuda:0`. The final post-processed action is CPU-side **by design**
(robot-facing), not a fallback. Inference peak VRAM: 465 MiB.

---

## Known-good vs still unvalidated

### ✅ Known-good — validated by direct measurement

- **CUDA on Blackwell.** torch 2.10.0+cu128 with `sm_120` in the compiled arch
  list, matching the device's own capability. Verified with real cuBLAS matmul
  (checked against CPU), cuDNN conv2d forward **and** backward, and bf16 autocast.
- **Reproducibility from the lockfile.** A fresh venv built with `uv sync --frozen`
  yields a byte-identical stack and re-passes the CUDA check.
- **TorchCodec is the live decoder.** Proven by instrumented call counts
  (34 torchcodec / 0 pyav), not by reading a config field — which matters because
  LeRobot's selector only calls `find_spec`, and a *broken* torchcodec satisfies that.
- **AV1 video decode** through system FFmpeg 6.1.1 at training speed; decode never
  became the bottleneck.
- **ACT training end-to-end** on an SO-101-topology dataset: 10k steps, no crash,
  no OOM, no divergence.
- **Checkpointing and reload.** Five checkpoints written; the final one reloads in
  a fresh process with weights byte-matching disk, and infers deterministically.
- **VRAM envelope on 8 GB.** Measured ladder from batch 2 → 16; batch 12 sits at
  81.8% peak.
- **bf16 AMP via Accelerate** — measured −13% wall clock, −664 MiB.

### ❌ Still unvalidated — do not assume these work

- **Your own data.** Nothing here touches a dataset you recorded. That is the
  whole point: this environment is now a fixed reference, so the next failure is
  attributable to data.
- **`lerobot-record` / teleoperation / motor buses.** No SO-101 hardware was
  attached. `feetech`, `dynamixel`, camera and calibration paths are entirely
  unexercised — `pyserial`/`pynput`/`evdev` are installed but never talked to a device.
- **Policy quality.** Loss went down; the policy was never evaluated in an
  environment or on a robot. A final loss of 0.161 says nothing about task success.
- **`lerobot-eval`.** Never run — it needs a gym environment, and no sim extra
  (`aloha`, `pusht`, `libero`, …) is installed.
- **Multi-GPU / distributed.** Accelerate is present and drives the loop, but only
  ever single-process, single-GPU.
- **Resuming from a checkpoint** (`--resume=true`). The `training_state/` dirs were
  written but never loaded back into a running trainer.
- **Other policies.** Only ACT. Diffusion, VQ-BeT, SmolVLA, π0 are untested and
  several need extras that were deliberately not installed.
- **Cold-network lockfile install.** The `--frozen` rebuild reused uv's
  hash-verified local wheel cache; it proves the pins are exact, not that a
  clean-machine download resolves identically.
- **Datasets in v2.x format.** LeRobot 0.5.0 requires `v3.0`; only a v3.0 dataset
  was exercised.
- **Larger batch / higher resolution.** Batch 16 fit in a clean-room probe (92%)
  but was never run for a sustained period.
