"""Simple GPU test script using PyTorch."""
from __future__ import annotations

import torch


def find_available_gpu(max_size: int = 1000) -> int:
    """Find a GPU with enough free memory for a small test tensor."""
    for i in range(torch.cuda.device_count()):
        try:
            torch.randn(max_size, max_size, device=f"cuda:{i}")
            torch.cuda.empty_cache()
            return i
        except torch.AcceleratorError:
            continue
    return 0


def main() -> None:
    print("=" * 60)
    print("GPU Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available. No GPU detected.")
        return

    print(f"[OK] CUDA is available.")
    print(f"     Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"     Device {i}: {props.name}")
        print(f"       Compute capability: {props.major}.{props.minor}")
        print(f"       Total memory: {props.total_memory / 1024 / 1e6:.2f} GB")

    # Find a GPU with enough free memory
    gpu_idx = find_available_gpu()
    device = torch.device(f"cuda:{gpu_idx}")
    print(f"     Using GPU: {gpu_idx}")

    # Allocate tensor on GPU and perform a simple operation
    size = 500
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.matmul(a, b)
    _ = c.sum()
    torch.cuda.synchronize(device)

    print(f"[OK] Matrix multiplication ({size}x{size}) on GPU {gpu_idx} succeeded.")
    print("=" * 60)


if __name__ == "__main__":
    main()
