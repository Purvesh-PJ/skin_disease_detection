"""
Unit Tests for ML Engine Automation Tools
------------------------------------------
Tests dataset verification, TFLite conversion functionality,
and pipeline parameter integrity.
"""

import os
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from pathlib import Path

from ml_engine.constants import TARGET_IMAGE_SIZE, NUM_CLASSES, CLASSES
from ml_engine.scripts.download_dataset import verify_dataset
from ml_engine.pipeline.data_loader import create_data_generators, compute_balanced_class_weights
from ml_engine.pipeline.orchestrator import SUPPORTED_MODELS
from ml_engine.pipeline.export_tflite import convert_h5_to_tflite, convert_all_models

class TestMLEngineAutomation:

    def test_verify_dataset_validates_real_base_dir(self):
        """Ensures HAM10000 base_dir contains all 7 classes across train/val/test."""
        dataset_dir = Path(__file__).resolve().parent.parent / "data" / "skin_disease_dataset" / "base_dir"
        summary, ready = verify_dataset(dataset_dir)

        assert ready is True
        assert "train_dir" in summary
        assert "val_dir" in summary
        assert "test_dir" in summary
        for cls in CLASSES:
            assert summary["train_dir"][cls] > 0
            assert summary["val_dir"][cls] > 0
            assert summary["test_dir"][cls] > 0

    def test_verify_dataset_fails_on_non_existent_dir(self):
        """Ensures verifier returns False on invalid path."""
        summary, ready = verify_dataset("fake_non_existent_dir_123")
        assert ready is False
        assert summary == {}

    def test_convert_h5_to_tflite_with_toy_model(self):
        """
        Creates a lightweight Keras model, saves it as .h5,
        converts it using convert_h5_to_tflite with quantization,
        and verifies the generated .tflite flatbuffer.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_file = tmp_path / "toy_model.h5"
            tflite_file = tmp_path / "toy_model.tflite"

            # 1. Create a minimal Keras model
            inputs = tf.keras.Input(shape=(*TARGET_IMAGE_SIZE, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.save(str(h5_file))

            assert os.path.exists(h5_file)

            # 2. Convert to quantized TFLite
            generated_path = convert_h5_to_tflite(h5_file, tflite_file, quantize=True)

            assert os.path.exists(generated_path)
            assert os.path.getsize(generated_path) > 0

            # 3. Verify TFLite Interpreter execution on converted flatbuffer
            interpreter = tf.lite.Interpreter(model_path=str(generated_path))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            assert list(input_details[0]["shape"]) == [1, 224, 224, 3]
            assert list(output_details[0]["shape"]) == [1, 7]

            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            interpreter.set_tensor(input_details[0]["index"], dummy_input)
            interpreter.invoke()
            output_tensor = interpreter.get_tensor(output_details[0]["index"])

            assert output_tensor.shape == (1, 7)
            assert np.isclose(np.sum(output_tensor), 1.0, atol=1e-3)

    def test_convert_h5_non_existent_file_raises_error(self):
        """Ensures FileNotFoundError is raised when converting a non-existent .h5 file."""
        with pytest.raises(FileNotFoundError):
            convert_h5_to_tflite("non_existent_file.h5")

    def test_convert_all_models_empty_directory_returns_empty(self):
        """Ensures convert_all_models gracefully handles directories with no .h5 files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = convert_all_models(tmp_dir)
            assert result == []

    def test_orchestrator_supported_models_list(self):
        """Verifies orchestrator includes all expected model architectures."""
        expected = ["resnet101", "densenet121", "efficientnetb3", "ensemble"]
        assert sorted(SUPPORTED_MODELS) == sorted(expected)

    def test_create_data_generators_and_class_weights(self):
        """Ensures data generators and balanced class weights are successfully generated."""
        dataset_dir = Path(__file__).resolve().parent.parent / "data" / "skin_disease_dataset" / "base_dir"
        train_gen, val_gen, test_gen = create_data_generators(dataset_dir, batch_size=32)

        assert train_gen is not None
        assert val_gen is not None
        assert test_gen is not None
        assert train_gen.num_classes == 7

        weights = compute_balanced_class_weights(train_gen)
        assert len(weights) == 7
        for cls_idx in range(7):
            assert cls_idx in weights
            assert weights[cls_idx] > 0

