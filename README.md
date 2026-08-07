# LeRobot 0.5.0 toolchain validation — WSL2 + Blackwell

A reproducible LeRobot **0.5.0** environment, proven end-to-end by training an
**ACT** policy on a public Hugging Face dataset until a checkpoint landed and
reloaded.

This is **toolchain validation, not science.** Success is "a training run
finished cleanly and the environment reproduces from a lockfile" — not "the
policy is good." Policy quality is explicitly not a success criterion.

The point is *attribution*: with this environment known-good, every later failure
is attributable to the data rather than the setup.

| | |
|---|---|
| Hardware | RTX 5070 Laptop GPU — **Blackwell, `sm_120`**, 8 GB VRAM |
| Host | WSL2 / Ubuntu 24.04.4 / driver 592.82 (Windows host only) |
| Stack | `torch 2.10.0+cu128` · `torchvision 0.25.0+cu128` · `torchcodec 0.10.0` · `lerobot 0.5.0` |
| Dataset | [`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) — SO-101, 50 eps, 11,939 frames |
| Policy | ACT, 51,597,190 params, ResNet-18 backbone |

## Outcome

**10,000 / 10,000 steps, exit code 0.** 67.2 min wall clock · peak VRAM 6,664 MiB
(81.8% of 8 GB) · final loss 0.161 · checkpoint reloads in a fresh process and
infers on GPU. Full numbers and the known-good/unvalidated split in
**[RESULTS.md](RESULTS.md)**.

## Documents

- **[SETUP.md](SETUP.md)** — the install sequence *as actually executed*, every
  version pin, every error hit and what fixed it. The load-bearing artifact.
- **[DECISIONS.md](DECISIONS.md)** — dataset choice, batch size, decode backend,
  and every deviation from the official docs with its rationale.
- **[RESULTS.md](RESULTS.md)** — measured outcome, and an explicit statement of
  what is now known-good versus still unvalidated.

## Reproduce

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python scripts/verify_cuda.py      # MUST pass before anything else
```

## Scripts

| Script | Purpose |
|---|---|
| [`scripts/verify_cuda.py`](scripts/verify_cuda.py) | Proves the CUDA stack is real. Checks the **compiled arch list contains `sm_120`**, not just `torch.cuda.is_available()`, then runs real cuBLAS/cuDNN fwd+bwd and bf16 autocast. |
| [`scripts/inspect_dataset.py`](scripts/inspect_dataset.py) | Dumps the dataset schema and pushes one batch through a DataLoader. **Instruments the decoders to prove TorchCodec — not pyav — did the work.** |
| [`scripts/probe_batch_size.py`](scripts/probe_batch_size.py) | Finds the VRAM ceiling empirically with real fwd/bwd/step at each batch size. |
| [`scripts/run_with_vram.sh`](scripts/run_with_vram.sh) | Runs a command while sampling `nvidia-smi`, reports true peak GPU memory. |
| [`scripts/verify_checkpoint.py`](scripts/verify_checkpoint.py) | Reloads a checkpoint in a **fresh process**, byte-checks weights against `model.safetensors`, and runs real inference. |

## Two traps this repo exists to document

1. **Blackwell needs `cu128`.** PyTorch wheels built against CUDA ≤12.6 contain no
   `sm_120` kernels. `torch.cuda.is_available()` returning `True` does not prove
   otherwise — check `torch.cuda.get_arch_list()`.
2. **`--policy.use_amp` does nothing during training in 0.5.0.** It is never
   referenced in `lerobot_train.py`. Use `ACCELERATE_MIXED_PRECISION=bf16`.

Training outputs, checkpoints and logs are gitignored and retained locally.
