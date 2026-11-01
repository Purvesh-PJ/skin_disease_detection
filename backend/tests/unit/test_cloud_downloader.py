"""
Unit Tests: Cloud Model Downloader Utility
------------------------------------------
Tests Google Drive URL transformations, cloud streaming, and zip extraction.
"""

import io
import os
import zipfile
import pytest

from app.inference.cloud_downloader import (
    transform_google_drive_url,
    download_file_from_cloud,
    download_and_extract_zip,
)


class TestCloudDownloader:
    """Test suite for cloud weights downloader and URL transformers."""

    def test_transform_google_drive_file_view_url(self):
        """Transforms standard /file/d/ Google Drive sharing link to raw direct download link."""
        raw_url = "https://drive.google.com/file/d/1AbC_xyz12345/view?usp=sharing"
        transformed = transform_google_drive_url(raw_url)

        assert transformed == "https://drive.usercontent.google.com/download?id=1AbC_xyz12345&confirm=t"

    def test_transform_google_drive_id_parameter_url(self):
        """Transforms Google Drive URL with id= query parameter to direct download link."""
        raw_url = "https://drive.google.com/uc?id=9Zyx_abc98765&export=download"
        transformed = transform_google_drive_url(raw_url)

        assert transformed == "https://drive.usercontent.google.com/download?id=9Zyx_abc98765&confirm=t"

    def test_transform_non_google_drive_url_unchanged(self):
        """Standard HTTP/HTTPS URLs (e.g. Hugging Face) should be returned unchanged."""
        hf_url = "https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/resnet101.tflite"
        transformed = transform_google_drive_url(hf_url)

        assert transformed == hf_url

    def test_download_file_from_cloud_success(self, tmp_path, mocker):
        """Verify successful stream download writes bytes to destination path."""
        dest_file = tmp_path / "downloaded_model.tflite"

        # Mock urllib.request.urlopen to return simulated model bytes
        mock_response = io.BytesIO(b"MOCK_TFLITE_MODEL_BYTES_DATA_STREAM")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *args: None

        mocker.patch("urllib.request.urlopen", return_value=mock_response)

        success = download_file_from_cloud("https://fake-cloud.org/model.tflite", str(dest_file))

        assert success is True
        assert os.path.exists(str(dest_file))
        with open(str(dest_file), "rb") as f:
            assert f.read() == b"MOCK_TFLITE_MODEL_BYTES_DATA_STREAM"

    def test_download_file_from_cloud_network_failure(self, tmp_path, mocker):
        """Verify that network exceptions are handled gracefully and return False."""
        dest_file = tmp_path / "failed_model.tflite"

        mocker.patch("urllib.request.urlopen", side_effect=Exception("Connection timed out"))

        success = download_file_from_cloud("https://fake-cloud.org/timeout.tflite", str(dest_file))

        assert success is False
        assert not os.path.exists(str(dest_file))

    def test_download_and_extract_zip(self, tmp_path, mocker):
        """Verify that zip download extracts contents and deletes the temporary zip archive."""
        # Create an in-memory zip archive with a test file inside
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("model_inside_zip.tflite", b"MODEL_CONTENT_INSIDE_ZIP")

        def mock_download_file(url, dest_path):
            with open(dest_path, "wb") as f:
                f.write(zip_buffer.getvalue())
            return True

        mocker.patch(
            "app.inference.cloud_downloader.download_file_from_cloud",
            side_effect=mock_download_file,
        )

        extract_dir = tmp_path / "extracted_models"
        extract_dir.mkdir()

        success = download_and_extract_zip("https://fake-cloud.org/models.zip", str(extract_dir))

        assert success is True
        extracted_file = extract_dir / "model_inside_zip.tflite"
        assert os.path.exists(str(extracted_file))
        assert not os.path.exists(str(extract_dir / "models_download.zip"))
