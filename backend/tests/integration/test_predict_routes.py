"""
Integration Tests: Skin Lesion Prediction API
----------------------------------------------
Tests the /predict multipart POST endpoint, input validation,
file persistence/cleanup, and status code behavior.
"""

import io
import os
import pytest

from app.core.config import config
from app.core.exceptions import ModelNotFoundError, InvalidImageError


class TestPredictEndpoint:
    """Integration test suite for POST /predict."""

    def test_predict_success_with_valid_image(self, client, dummy_image_bytes, mocker):
        """
        Verify that uploading a valid JPEG image returns 200 OK
        with predicted disease, confidence, details, and metadata.
        """
        mock_result = {
            "predicted_disease": "mel",
            "confidence": "96",
            "disease_details": {
                "name": "Melanoma",
                "description": "A serious form of skin cancer.",
            },
            "ensemble_models_used": ["resnet", "densenet", "efficientnet"],
            "filename": "clinical_scan.jpg",
            "message": "Image processed successfully",
        }
        mocker.patch(
            "app.routes.predict.process_prediction",
            return_value=mock_result,
        )

        data = {
            "image": (dummy_image_bytes, "clinical_scan.jpg"),
        }

        response = client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["predicted_disease"] == "mel"
        assert json_data["confidence"] == "96"
        assert json_data["disease_details"]["name"] == "Melanoma"
        assert json_data["filename"] == "clinical_scan.jpg"
        assert json_data["message"] == "Image processed successfully"

    def test_predict_missing_image_payload_returns_400(self, client):
        """Verify that requests missing the 'image' field return 400 Bad Request."""
        response = client.post(
            "/predict",
            data={},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["error"] == "No image file provided"

    def test_predict_empty_filename_returns_400(self, client):
        """Verify that sending a file with an empty filename returns 400."""
        empty_file = (io.BytesIO(b"empty"), "")
        response = client.post(
            "/predict",
            data={"image": empty_file},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["error"] == "No selected file"

    def test_predict_domain_exception_returns_custom_status(
        self, client, dummy_image_bytes, mocker
    ):
        """
        Verify that domain exceptions (e.g. ModelNotFoundError - 503)
        return proper status codes instead of generic unhandled crashes.
        """
        mocker.patch(
            "app.routes.predict.process_prediction",
            side_effect=ModelNotFoundError("AI models unavailable"),
        )

        data = {"image": (dummy_image_bytes, "scan.jpg")}
        response = client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 530
        json_data = response.get_json()
        assert "AI models unavailable" in json_data["error"]

    def test_predict_invalid_image_domain_exception_returns_400(
        self, client, dummy_image_bytes, mocker
    ):
        """Verify that InvalidImageError raises a clean 400 Bad Request."""
        mocker.patch(
            "app.routes.predict.process_prediction",
            side_effect=InvalidImageError("Corrupt image data"),
        )

        data = {"image": (dummy_image_bytes, "corrupt.jpg")}
        response = client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert "Corrupt image data" in json_data["error"]

    def test_predict_temp_file_is_cleaned_up(
        self, client, dummy_image_bytes, mocker
    ):
        """Verify that uploaded images are removed from disk after request handling."""
        filename = "temp_cleanup_test.jpg"
        target_path = os.path.join(config.UPLOAD_FOLDER, filename)

        mocker.patch(
            "app.routes.predict.process_prediction",
            return_value={"message": "ok", "predicted_disease": "nv", "confidence": "90"},
        )

        data = {"image": (dummy_image_bytes, filename)}
        client.post("/predict", data=data, content_type="multipart/form-data")

        # The file MUST be cleaned up by the finally block
        assert not os.path.exists(target_path)
