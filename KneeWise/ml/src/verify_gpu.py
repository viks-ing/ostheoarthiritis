#!/usr/bin/env python3
"""
verify_gpu.py - KneeWise ML Pipeline Hardware & PyTorch Verification Script

Checks PyTorch installation, CUDA availability, GPU device name, VRAM limits,
and verifies tensor execution on GPU. Alerts user with explicit warnings if
CUDA is unavailable.
"""

import sys
import os

def verify_gpu():
    print("=" * 65)
    print("      KneeWise ML Pipeline — Hardware & PyTorch Verification     ")
    print("=" * 65)

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch ('torch') is not installed in the active Python environment.")
        print("Please install PyTorch with CUDA enabled using:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"[*] Python version:  {sys.version.split()[0]}")
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] Built with CUDA: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("\n" + "!" * 65)
        print("[WARNING] CUDA IS NOT AVAILABLE. PyTorch is running in CPU-ONLY mode.")
        print("!" * 65)
        print("Medical image model training on CPU will be extremely slow.")
        print("To enable GPU acceleration on your NVIDIA RTX 3050:")
        print("  1. Verify NVIDIA drivers are installed (run: nvidia-smi).")
        print("  2. Reinstall PyTorch with CUDA 12.1 or CUDA 11.8 support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("!" * 65 + "\n")
        return False

    device_count = torch.cuda.device_count()
    device_id = 0
    device_name = torch.cuda.get_device_name(device_id)
    props = torch.cuda.get_device_properties(device_id)
    total_memory_gb = props.total_memory / (1024 ** 3)
    major, minor = props.major, props.minor

    print(f"\n[+] CUDA is AVAILABLE!")
    print(f"[*] GPU Device Count:     {device_count}")
    print(f"[*] Current Device ID:    {device_id}")
    print(f"[*] GPU Model Name:       {device_name}")
    print(f"[*] Compute Capability:   {major}.{minor} (Ampere architecture)")
    print(f"[*] Dedicated VRAM:       {total_memory_gb:.2f} GB ({props.total_memory / (1024**2):.0f} MB)")
    
    # Test tensor allocation and computation
    try:
        x = torch.randn(1000, 1000, device=f"cuda:{device_id}")
        y = torch.matmul(x, x)
        torch.cuda.synchronize(device_id)
        allocated_mb = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        del x, y
        torch.cuda.empty_cache()
        print(f"[+] CUDA Compute Test:    SUCCESS (1000x1000 matrix multiplication)")
        print(f"[*] Memory Allocated:     {allocated_mb:.2f} MB")
        print(f"[*] Memory Reserved:      {reserved_mb:.2f} MB")
    except Exception as e:
        print(f"\n[!] CUDA Tensor Execution Failed: {e}")
        return False

    # Hardware-specific optimization advice for RTX 3050 (4 GB VRAM)
    print("\n" + "-" * 65)
    print("Hardware Profile & Constraints for RTX 3050 (4 GB VRAM):")
    print("  • Recommended Batch Size: 8 (or 4 if encountering CUDA OOM)")
    print("  • Mixed Precision (FP16): Supported (Ampere native Tensor Cores)")
    print("  • Windows DataLoader:     num_workers=0 or 2 (avoids shared memory bottleneck)")
    print("  • Pin Memory:             Recommended (speeds up CPU->GPU tensor transfers)")
    print("-" * 65)
    return True

if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
