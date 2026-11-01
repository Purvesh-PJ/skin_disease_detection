"""
Unit Tests: Ensemble Voting & Prediction Service
------------------------------------------------
Tests multi-model probability fusion, confidence score calculation,
graceful partial-model fallback, and custom error handling.
"""

import numpy as np
import pytest

from app.core.constants import (
    CLASS_INDICES,
    IDX2CLASS,
    SKIN_DISEASE_CLASSES,
    USER_FRIENDLY_MAPPING,
)
from app.core.exceptions import ModelNotFoundError
from app.services.prediction_service import (
    predict_skin_disease,
    process_prediction,
)


class TestEnsembleConsensus:
    """Test suite for softmax consensus fusion and prediction business logic."""

    @pytest.fixture
    def mock_preprocess(self, mocker):
        """Mocks image preprocessing to return a dummy 4D tensor."""
        dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        return mocker.patch(
            "app.services.prediction_service.load_and_preprocess_image",
            return_value=dummy_tensor,
        )

    @pytest.fixture
    def mock_warmup(self, mocker):
        """Mocks ensure_models_available to avoid filesystem/cloud checks during tests."""
        return mocker.patch("app.services.prediction_service.ensure_models_available")

    def test_ensemble_consensus_predicts_correct_disease(
        self, mocker, mock_preprocess, mock_warmup
    ):
        """
        Verify that multi-model probabilities are averaged via softmax fusion,
        correctly identifying the consensus winner and calculating peak confidence.
        """
        # Mel index is 4 in SKIN_DISEASE_CLASSES ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        mel_idx = CLASS_INDICES["mel"]

        # 3 simulated output probability distributions (all pointing to Melanoma)
        resnet_probs = np.array([0.02, 0.01, 0.01, 0.01, 0.92, 0.02, 0.01], dtype=np.float32)
        densenet_probs = np.array([0.01, 0.01, 0.02, 0.01, 0.90, 0.03, 0.02], dtype=np.float32)
        efficientnet_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.00], dtype=np.float32)

        # Mock model file existence on disk
        mocker.patch("os.path.exists", return_value=True)

        # Mock inference runner to return distinct distributions for each architecture
        def mock_infer(model_name, path, tensor):
            if "resnet" in model_name:
                return resnet_probs
            elif "densenet" in model_name:
                return densenet_probs
            return efficientnet_probs

        mocker.patch(
            "app.services.prediction_service.run_single_model_inference",
            side_effect=mock_infer,
        )

        result = predict_skin_disease("dummy_path.jpg")

        # Verify consensus prediction
        assert result["predicted_disease"] == "mel"
        assert result["disease_details"]["name"] == "Melanoma"

        # Peak model confidence across the ensemble was 95% (efficientnet: 0.95)
        assert result["confidence"] == "95"

        # All 3 architectures contributed
        assert len(result["ensemble_models_used"]) == 3
        assert "resnet" in result["ensemble_models_used"]
        assert "densenet" in result["ensemble_models_used"]
        assert "efficientnet" in result["ensemble_models_used"]

    def test_partial_model_fallback_resilience(
        self, mocker, mock_preprocess, mock_warmup
    ):
        """
        Verify that if 1 model is missing from disk, the service gracefully
        computes consensus using the remaining 2 available models.
        """
        nv_idx = CLASS_INDICES["nv"]

        # DenseNet & EfficientNet point to 'nv' (Melanocytic Nevi)
        densenet_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.03], dtype=np.float32)
        efficientnet_probs = np.array([0.02, 0.01, 0.01, 0.01, 0.02, 0.88, 0.05], dtype=np.float32)

        # Simulate ResNet missing, other 2 present
        def mock_exists(path):
            if "resnet" in path:
                return False
            return True

        mocker.patch("os.path.exists", side_effect=mock_exists)

        def mock_infer(model_name, path, tensor):
            if "densenet" in model_name:
                return densenet_probs
            return efficientnet_probs

        mocker.patch(
            "app.services.prediction_service.run_single_model_inference",
            side_effect=mock_infer,
        )

        result = predict_skin_disease("dummy_path.jpg")

        assert result["predicted_disease"] == "nv"
        assert result["disease_details"]["name"] == "Melanocytic Nevus"
        assert len(result["ensemble_models_used"]) == 2
        assert "resnet" not in result["ensemble_models_used"]

    def test_all_models_missing_raises_model_not_found(
        self, mocker, mock_preprocess, mock_warmup
    ):
        """
        Verify that if no models exist on disk, ModelNotFoundError is cleanly raised.
        """
        mocker.patch("os.path.exists", return_value=False)

        with pytest.raises(ModelNotFoundError) as exc_info:
            predict_skin_disease("dummy_path.jpg")

        assert "No AI models could be executed" in str(exc_info.value)

    def test_process_prediction_enriches_metadata(
        self, mocker, mock_preprocess, mock_warmup
    ):
        """
        Verify that the high-level process_prediction service method enriches
        the prediction result with filename and success message.
        """
        dummy_result = {
            "predicted_disease": "bcc",
            "confidence": "94",
            "disease_details": USER_FRIENDLY_MAPPING["bcc"],
            "ensemble_models_used": ["resnet"],
        }
        mocker.patch(
            "app.services.prediction_service.predict_skin_disease",
            return_value=dummy_result,
        )

        final_response = process_prediction("some_temp_file.jpg", "patient_scan_01.jpg")

        assert final_response["filename"] == "patient_scan_01.jpg"
        assert final_response["message"] == "Image processed successfully"
        assert final_response["predicted_disease"] == "bcc"


class TestDiseaseMappings:
    """Verifies data integrity for disease labels, indices, and medical metadata."""

    def test_all_7_classes_have_user_friendly_mappings(self):
        """Ensure all 7 disease categories have descriptions and titles."""
        for code in SKIN_DISEASE_CLASSES:
            assert code in USER_FRIENDLY_MAPPING, f"Missing friendly mapping for '{code}'"
            entry = USER_FRIENDLY_MAPPING[code]
            assert "name" in entry and len(entry["name"]) > 0
            assert "description" in entry and len(entry["description"]) > 10

    def test_class_indices_bijection(self):
        """Ensure CLASS_INDICES and IDX2CLASS are exact 1-to-1 inverse mappings."""
        assert len(CLASS_INDICES) == len(IDX2CLASS) == 7
        for cls_name, idx in CLASS_INDICES.items():
            assert IDX2CLASS[idx] == cls_name
