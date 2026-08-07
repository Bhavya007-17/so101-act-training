"""Prove the CUDA stack is real, not a CPU fallback wearing a CUDA costume.

This is deliberately paranoid because the whole point of this repo is
attribution: a green check here must MEAN something. On Blackwell (sm_120)
the classic silent failure is a torch build whose compiled arch list stops
at sm_90 -- `torch.cuda.is_available()` still returns True, and you only
find out when a kernel launch dies or quietly runs somewhere unexpected.

Exit code 0 = stack verified. Non-zero = do not proceed.
"""

import sys

import torch

FAIL = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f'  -- {detail}' if detail else ''}")
    if not ok:
        FAIL.append(label)


print("=" * 72)
print("TORCH BUILD")
print("=" * 72)
print(f"  torch.__version__       : {torch.__version__}")
print(f"  torch.version.cuda      : {torch.version.cuda}")
print(f"  torch.version.git_version: {torch.version.git_version}")
print(f"  cuDNN                   : {torch.backends.cudnn.version()}")

# 1. The build must be a CUDA build at all. A CPU-only wheel reports None here.
check("torch is a CUDA build", torch.version.cuda is not None, f"cuda={torch.version.cuda}")

# 2. CUDA 12.8+ is the floor for sm_120 (Blackwell).
if torch.version.cuda:
    major, minor = (int(x) for x in torch.version.cuda.split(".")[:2])
    check(
        "CUDA runtime >= 12.8 (sm_120 floor)",
        (major, minor) >= (12, 8),
        f"built against CUDA {torch.version.cuda}",
    )

print()
print("=" * 72)
print("COMPILED ARCH LIST  <-- the check that catches a Blackwell mismatch")
print("=" * 72)
arch_list = torch.cuda.get_arch_list()
print(f"  {arch_list}")
check("sm_120 kernels compiled into this wheel", "sm_120" in arch_list)

print()
print("=" * 72)
print("DEVICE VISIBILITY")
print("=" * 72)
avail = torch.cuda.is_available()
check("torch.cuda.is_available()", avail)
if not avail:
    print("\n>>> ABORT: no CUDA device. Nothing below can be trusted.")
    sys.exit(1)

n = torch.cuda.device_count()
print(f"  device_count            : {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    cc = f"sm_{props.major}{props.minor}"
    print(f"  cuda:{i} name           : {props.name}")
    print(f"  cuda:{i} capability     : {cc} (compute {props.major}.{props.minor})")
    print(f"  cuda:{i} total VRAM     : {props.total_memory / 1024**3:.2f} GiB")
    print(f"  cuda:{i} multiprocessors: {props.multi_processor_count}")
    # The device's own capability must be in the compiled arch list.
    check(
        f"cuda:{i} capability {cc} is in compiled arch list",
        cc in arch_list,
        f"device={cc}, wheel has {arch_list}",
    )

print()
print("=" * 72)
print("REAL GPU WORK (not just allocation -- actual kernel launches)")
print("=" * 72)
dev = torch.device("cuda:0")

# matmul: exercises cuBLAS
a = torch.randn(2048, 2048, device=dev)
b = torch.randn(2048, 2048, device=dev)
c = a @ b
torch.cuda.synchronize()
expected = a.cpu() @ b.cpu()
err = (c.cpu() - expected).abs().max().item()
check("fp32 matmul on GPU matches CPU", err < 1e-2, f"max abs err={err:.3e}")
check("result tensor is on CUDA", c.is_cuda, f"device={c.device}")

# conv2d via cuDNN: this is what ACT's ResNet-18 backbone actually hits
conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3).to(dev)
x = torch.randn(4, 3, 224, 224, device=dev)
y = conv(x)
torch.cuda.synchronize()
check("cuDNN conv2d forward", tuple(y.shape) == (4, 64, 112, 112), f"out={tuple(y.shape)}")

# backward pass: exercises autograd + cuDNN backward kernels
y.sum().backward()
torch.cuda.synchronize()
check("conv2d backward produced grads", conv.weight.grad is not None)
check(
    "grads are finite",
    bool(torch.isfinite(conv.weight.grad).all().item()),
)

# AMP / bf16: what training will actually use
with torch.autocast("cuda", dtype=torch.bfloat16):
    z = conv(x)
torch.cuda.synchronize()
check("bf16 autocast forward", z.dtype == torch.bfloat16, f"dtype={z.dtype}")

alloc = torch.cuda.max_memory_allocated() / 1024**2
print(f"\n  peak allocated during checks: {alloc:.1f} MiB")

print()
print("=" * 72)
if FAIL:
    print(f"RESULT: FAILED ({len(FAIL)} check(s))")
    for f in FAIL:
        print(f"  - {f}")
    print("=" * 72)
    sys.exit(1)

print("RESULT: CUDA stack VERIFIED -- real GPU, real kernels, sm_120 present.")
print("=" * 72)
