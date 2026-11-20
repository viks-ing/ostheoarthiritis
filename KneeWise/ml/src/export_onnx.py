#!/usr/bin/env python3
"""
export_onnx.py - KneeWise Model ONNX Exporter & Validation

Exports trained PyTorch ResNet18 model to standard ONNX format for
high-performance serving and future Django/FastAPI backend integration.
Validates ONNX model graph and numerical equivalence against PyTorch.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH,
    ONNX_EXPORT_PATH,
    DEFAULT_ARCHITECTURE,
    NUM_CLASSES,
)
from model import build_model, load_checkpoint


def export_model_to_onnx(
    checkpoint_path: Path = BEST_MODEL_PATH,
    output_path: Path = ONNX_EXPORT_PATH,
    batch_size: int = 1,
    opset_version: int = 14,
):
    """Exports model to ONNX format and verifies numerical tolerance."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — ONNX Model Export Module          ")
    print("=" * 65)

    model = build_model(architecture=DEFAULT_ARCHITECTURE, num_classes=NUM_CLASSES, pretrained=False)
    
    if checkpoint_path.exists():
        print(f"[*] Loading trained weights from: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device=torch.device("cpu"))
    else:
        print(f"[!] Warning: Trained checkpoint not found at: {checkpoint_path}")
        print("    Exporting baseline architecture for graph and pipeline verification.")

    model.eval()

    # Create representative dummy input tensor (batch, channels, height, width)
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

    # PyTorch reference output
    with torch.no_grad():
        pt_logits = model(dummy_input).numpy()

    # Check onnx prerequisite
    try:
        import onnx
    except ImportError:
        print("\n" + "!" * 65)
        print("[PREREQUISITE NOTICE] The 'onnx' package is required for exporting.")
        print("Please install onnx and onnxruntime using:")
        print("  pip install onnx onnxruntime")
        print("!" * 65 + "\n")
        return

    print(f"[*] Exporting graph to: {output_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["logits"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        print(f"[+] Export successful! File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    except Exception as e:
        print(f"[!] Export failed: {e}")
        return

    # Verification Step
    print("\n[*] Validating ONNX model graph & numerical tolerance...")
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX model syntax & graph structure: VALID")
    except ImportError:
        print("[*] 'onnx' package not found; install with 'pip install onnx' for formal graph linting.")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        ort_logits = ort_outs[0]

        # Verify output dimensions
        assert ort_logits.shape == (batch_size, NUM_CLASSES), f"Shape mismatch: {ort_logits.shape} vs {(batch_size, NUM_CLASSES)}"
        print(f"[+] Output shape verified: {ort_logits.shape}")

        # Verify numerical equivalence
        max_diff = np.max(np.abs(pt_logits - ort_logits))
        print(f"[*] Max absolute difference (PyTorch vs ONNX): {max_diff:.6e}")
        if max_diff < 1e-4:
            print("[+] Numerical equivalence test: PASSED (tolerance < 1e-4)")
        else:
            print(f"[!] Warning: Discrepancy exceeds 1e-4 ({max_diff})")

    except ImportError:
        print("[*] 'onnxruntime' package not found; install with 'pip install onnxruntime' to verify ONNX runtime inference.")

    print("\n" + "=" * 65)
    print(f"[+] ONNX Export Complete: {output_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Export KneeWise PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(ONNX_EXPORT_PATH))
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_model_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
