"""
Unit Tests for ML Engine Automation Tools
------------------------------------------
Tests dataset verification, TFLite conversion functionality,
and pipeline parameter integrity.
"""

import os
import zipfile
import tempfile
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from ml_engine.constants import TARGET_IMAGE_SIZE, NUM_CLASSES, CLASSES
from ml_engine.scripts.download_dataset import verify_dataset
from ml_engine.pipeline.prepare_dataset import prepare_dataset
from ml_engine.pipeline.data_generators import create_data_generators
from ml_engine.pipeline.class_weights import compute_balanced_class_weights
from ml_engine.pipeline.model_evaluation import evaluate_model_predictions
from ml_engine.pipeline.training_pipeline import SUPPORTED_MODELS
from ml_engine.export.tflite import convert_h5_to_tflite, convert_all_models

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

    def test_prepare_dataset_end_to_end_with_nested_zip(self):
        """
        Validates prepare_dataset handles Kaggle raw archives, nested zip extraction,
        stratified 80/10/10 splitting, and class directory structuring end-to-end.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            raw_dir = tmp / "raw"
            out_dir = tmp / "output" / "base_dir"
            raw_dir.mkdir(parents=True)

            # 1. Create mock metadata (70 rows, 10 per class)
            data = []
            for idx in range(70):
                cls = CLASSES[idx % len(CLASSES)]
                data.append({
                    "lesion_id": f"LES_{idx}",
                    "image_id": f"IMG_{idx:03d}",
                    "dx": cls,
                    "dx_type": "histo",
                    "age": 45,
                    "sex": "male",
                    "localization": "back"
                })
            pd.DataFrame(data).to_csv(raw_dir / "HAM10000_metadata.csv", index=False)

            # 2. Package images into nested zip archive (Kaggle format)
            zip_path = raw_dir / "HAM10000_images_part_1.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for idx in range(70):
                    zf.writestr(f"IMG_{idx:03d}.jpg", b"fake_jpeg_content")

            # 3. Execute prepare_dataset
            prepare_dataset(str(raw_dir), str(out_dir))

            # 4. Verify output structure
            summary, valid = verify_dataset(str(out_dir))
            assert valid is True
            assert summary["train_dir"]["_total"] == 56  # 80% of 70
            assert summary["val_dir"]["_total"] == 7    # 10% of 70
            assert summary["test_dir"]["_total"] == 7   # 10% of 70
            for cls in CLASSES:
                assert summary["train_dir"][cls] == 8
                assert summary["val_dir"][cls] == 1
                assert summary["test_dir"][cls] == 1

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
            model.save(h5_file)

            # 2. Convert to TFLite
            converted_path = convert_h5_to_tflite(h5_file, tflite_file, quantize=True)
            assert os.path.exists(converted_path)
            assert os.path.getsize(converted_path) > 0

            # 3. Validate generated TFLite Interpreter
            interpreter = tf.lite.Interpreter(model_path=str(converted_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            assert list(input_details[0]["shape"]) == [1, *TARGET_IMAGE_SIZE, 3]
            assert list(output_details[0]["shape"]) == [1, NUM_CLASSES]

    def test_convert_all_models_skips_missing_safely(self):
        """Ensures convert_all_models gracefully handles missing model directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Empty directory with no .h5 files
            results = convert_all_models(models_dir=tmp_dir)
            assert results == []

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

    def test_evaluator_metrics_calculation(self):
        """Ensures evaluator produces classification report and confusion matrix."""
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        report, cm = evaluate_model_predictions(y_true, y_pred, class_names=CLASSES)

        assert report is not None
        assert "precision" in report
        assert "recall" in report
        assert cm.shape == (7, 7)
        assert np.trace(cm) == 7  # All correctly predicted along diagonal
