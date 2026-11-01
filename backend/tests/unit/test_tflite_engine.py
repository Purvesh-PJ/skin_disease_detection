"""
Unit Tests: Real TFLite Engine & Memory Diagnostics
---------------------------------------------------
Tests TensorFlow Lite Interpreter execution, probability outputs,
memory profiling, and error handling.
"""

import os
import numpy as np
import pytest

from app.core.config import config
from app.services.prediction_service import (
    run_single_model_inference,
    get_memory_usage_mb,
)


class TestTFLiteEngineAndDiagnostics:
    """Tests TFLite interpreter allocation, inference, and memory monitors."""

    def test_get_memory_usage_mb_returns_positive_float(self):
        """Verify that get_memory_usage_mb returns a valid positive float representing RAM."""
        ram_mb = get_memory_usage_mb()

        assert isinstance(ram_mb, (int, float))
        assert ram_mb > 0.0

    def test_real_tflite_inference_if_model_exists(self):
        """
        Verify that real TFLite flatbuffers on disk execute properly,
        allocating tensors and returning a 7-class normalized probability vector.
        """
        # Look for local densenet121.tflite (smallest model ~8MB)
        candidate_paths = [
            os.path.join(config.MODEL_DIR, "densenet121.tflite"),
            os.path.join(config.MODEL_DIR, "efficientnetb3.tflite"),
            os.path.join(config.MODEL_DIR, "resnet101.tflite"),
        ]

        model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if not model_path:
            pytest.skip("No local .tflite model files present on disk to execute real inference.")

        # Create dummy 4D input tensor (1, 224, 224, 3)
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

        # Execute real inference
        probabilities = run_single_model_inference("densenet", model_path, dummy_tensor)

        # Output validations
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (7,)
        assert all(0.0 <= p <= 1.0 for p in probabilities)
        # Probabilities should sum to approximately 1.0
        assert np.isclose(np.sum(probabilities), 1.0, atol=1e-2)

    def test_run_single_model_inference_invalid_path_raises_error(self):
        """Verify that an invalid model file path raises an exception and cleans up."""
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

        with pytest.raises(Exception):
            run_single_model_inference("fake_model", "invalid_path.tflite", dummy_tensor)

