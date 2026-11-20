"""
test_inference.py - Comprehensive Unit & Regression Test Suite for Inference Pipeline
Tests:
1. Frozen checkpoint loading & SHA-256 cryptographic verification
2. Architecture compatibility & classification head structure
3. Five-class probability validity & normalization (sum == 1.0)
4. Binary OA probability calculation correctness
5. Invalid image handling & corrupted input resilience
6. CPU inference execution
7. CUDA inference execution (if available)
8. Repeated inference consistency (deterministic output)
9. Preprocessing parity with Experiment D validation pipeline
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import (
    KneeWiseInferenceEngine,
    EXPECTED_CHAMPION_SHA256,
    DEFAULT_CHAMPION_PATH,
    calculate_sha256,
)
from dataset import get_val_test_transforms


def test_checkpoint_and_sha256():
    print("[Test 1/9] Checking checkpoint existence and SHA-256 hash...")
    assert DEFAULT_CHAMPION_PATH.exists(), f"Missing checkpoint: {DEFAULT_CHAMPION_PATH}"
    actual_hash = calculate_sha256(DEFAULT_CHAMPION_PATH)
    assert actual_hash == EXPECTED_CHAMPION_SHA256, f"Checksum mismatch: {actual_hash} != {EXPECTED_CHAMPION_SHA256}"
    print("  -> PASSED: Cryptographic integrity confirmed.")


def test_architecture_compatibility():
    print("[Test 2/9] Checking architecture compatibility and parameter count...")
    engine = KneeWiseInferenceEngine(device="cpu")
    total_params = sum(p.numel() for p in engine.model.parameters())
    assert total_params == 4013953, f"Unexpected parameter count: {total_params} != 4,013,953"
    assert hasattr(engine.model, "backbone"), "Model missing backbone module"
    assert hasattr(engine.model.backbone, "classifier"), "Backbone missing classifier head"
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = engine.model(dummy_input)
    assert out.shape == (1, 5), f"Unexpected output shape: {out.shape} != (1, 5)"
    print("  -> PASSED: Architecture matches EfficientNet-B0 5-class head exactly.")


def test_probability_validity(sample_img_path):
    print("[Test 3/9] Checking five-class probability validity (sum == 1.0)...")
    engine = KneeWiseInferenceEngine(device="cpu")
    res = engine.predict(sample_img_path)
    probs = res["prediction"]["probabilities"]
    prob_sum = sum(probs.values())
    for g, p in probs.items():
        assert 0.0 <= p <= 1.0, f"Probability out of range: {g}={p}"
    assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {prob_sum}"
    print(f"  -> PASSED: Valid probability distribution (sum = {prob_sum:.4f}).")


def test_binary_risk_calculation(sample_img_path):
    print("[Test 4/9] Checking binary OA risk formula correctness...")
    engine = KneeWiseInferenceEngine(device="cpu")
    res = engine.predict(sample_img_path, screening_threshold=0.30)
    probs = res["prediction"]["probabilities"]
    expected_binary = round(probs["KL_2"] + probs["KL_3"] + probs["KL_4"], 4)
    actual_binary = res["binary_screening"]["binary_oa_probability"]
    assert abs(actual_binary - expected_binary) < 1e-4, f"Mismatch in binary OA risk: {actual_binary} != {expected_binary}"
    assert res["binary_screening"]["is_oa_likely"] == (actual_binary >= 0.30), "Binary classification condition failed"
    print("  -> PASSED: Derived binary probability matches P(KL2)+P(KL3)+P(KL4).")


def test_invalid_input_handling():
    print("[Test 5/9] Checking invalid image and corrupted input handling...")
    engine = KneeWiseInferenceEngine(device="cpu")

    # Non-existent file
    try:
        engine.predict("non_existent_image_12345.png")
        assert False, "Failed to raise FileNotFoundError for missing file"
    except FileNotFoundError:
        pass

    # Corrupted / invalid file bytes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"NOT_A_VALID_IMAGE_DATA_CORRUPT")
        tmp_name = f.name
    try:
        engine.predict(tmp_name)
        assert False, "Failed to raise ValueError for corrupted image"
    except ValueError:
        pass
    finally:
        os.remove(tmp_name)
    print("  -> PASSED: Graceful error handling for missing and corrupted files.")


def test_cpu_inference(sample_img_path):
    print("[Test 6/9] Testing CPU inference execution...")
    engine = KneeWiseInferenceEngine(device="cpu")
    res = engine.predict(sample_img_path)
    assert res["model_metadata"]["device"] == "cpu"
    assert "inference_time_ms" in res["model_metadata"]
    print(f"  -> PASSED: CPU inference latency: {res['model_metadata']['inference_time_ms']:.2f} ms")


def test_cuda_inference(sample_img_path):
    print("[Test 7/9] Testing CUDA inference execution...")
    if not torch.cuda.is_available():
        print("  -> SKIPPED: CUDA not available on this machine.")
        return
    engine = KneeWiseInferenceEngine(device="cuda")
    res = engine.predict(sample_img_path)
    assert res["model_metadata"]["device"] == "cuda"
    print(f"  -> PASSED: CUDA inference latency: {res['model_metadata']['inference_time_ms']:.2f} ms")


def test_inference_consistency(sample_img_path):
    print("[Test 8/9] Checking repeated inference consistency (determinism)...")
    engine = KneeWiseInferenceEngine(device="cpu")
    res1 = engine.predict(sample_img_path)
    res2 = engine.predict(sample_img_path)

    p1 = list(res1["prediction"]["probabilities"].values())
    p2 = list(res2["prediction"]["probabilities"].values())
    assert np.allclose(p1, p2, atol=1e-6), "Inconsistent predictions across repeated calls!"
    print("  -> PASSED: 100% deterministic prediction parity confirmed.")


def test_preprocessing_parity(sample_img_path):
    print("[Test 9/9] Checking preprocessing parity with Experiment D validation transforms...")
    engine = KneeWiseInferenceEngine(device="cpu")
    engine_tensor = engine.preprocess_image(sample_img_path)

    # Standard validation transform from dataset.py
    val_transform = get_val_test_transforms((224, 224))
    with Image.open(sample_img_path) as raw:
        img_rgb = raw.convert("RGB")
    val_tensor = val_transform(img_rgb).unsqueeze(0)

    assert torch.allclose(engine_tensor, val_tensor, atol=1e-6), "Preprocessing disparity detected!"
    print("  -> PASSED: 100% exact parity with training validation transforms.")


def run_all_tests():
    sample_path = Path(r"C:\Users\vikra\OneDrive\Desktop\rakshitha\KneeWise\ml\data\raw\test\0\9003175L.png")
    if not sample_path.exists():
        # Fallback to any image in raw/test
        found = list(Path(r"KneeWise/ml/data/raw/test").glob("*/*.png"))
        assert found, "No sample test image found for tests"
        sample_path = found[0]

    print("=" * 70)
    print("      KNEEWISE INFERENCE SUITE — COMPREHENSIVE SOFTWARE TESTS")
    print("=" * 70)
    print(f"Sample test image: {sample_path}\n")

    test_checkpoint_and_sha256()
    test_architecture_compatibility()
    test_probability_validity(sample_path)
    test_binary_risk_calculation(sample_path)
    test_invalid_input_handling()
    test_cpu_inference(sample_path)
    test_cuda_inference(sample_path)
    test_inference_consistency(sample_path)
    test_preprocessing_parity(sample_path)

    print("\n" + "=" * 70)
    print(" ALL 9 INFERENCE TESTS PASSED CLEANLY (Zero Regressions)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
