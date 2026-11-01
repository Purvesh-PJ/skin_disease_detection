"""
Unit Tests: Image Preprocessing Pipeline
----------------------------------------
Tests image loading, format normalization, resizing, and error handling
in app.services.prediction_service.
"""

import cv2
import numpy as np
import pytest

from app.core.constants import TARGET_IMAGE_SIZE
from app.core.exceptions import InvalidImageError
from app.services.prediction_service import load_and_preprocess_image


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a temporary valid RGB JPEG image on disk for testing."""
    # Create a 300x400 RGB test image with non-zero pixel data
    img_data = np.full((300, 400, 3), 128, dtype=np.uint8)
    image_file = tmp_path / "test_lesion.jpg"
    cv2.imwrite(str(image_file), img_data)
    return str(image_file)


class TestImagePreprocessing:
    """Test suite for image loading, validation, and tensor transformation."""

    def test_load_valid_image_shape_and_type(self, sample_image_path):
        """
        Verify that a valid image is loaded, converted to RGB, resized to (224, 224),
        normalized for CNN backbones, and returned as a 4D batch tensor.
        """
        tensor = load_and_preprocess_image(sample_image_path)

        # Must return a numpy ndarray
        assert isinstance(tensor, np.ndarray)

        # Batch dimension + 224x224 RGB -> shape must be exactly (1, 224, 224, 3)
        assert tensor.shape == (1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3)

        # Tensor must have data
        assert tensor.size > 0
        assert tensor.ndim == 4

    def test_load_non_existent_image_raises_error(self):
        """
        Verify that attempting to load a non-existent file path cleanly
        raises InvalidImageError instead of crashing the process.
        """
        non_existent_path = "non_existent_path_404_test.jpg"

        with pytest.raises(InvalidImageError) as exc_info:
            load_and_preprocess_image(non_existent_path)

        assert "Image not found or unreadable" in str(exc_info.value)

    def test_load_corrupt_file_raises_error(self, tmp_path):
        """
        Verify that a non-image text/corrupted file raises InvalidImageError.
        """
        corrupt_file = tmp_path / "corrupt_image.jpg"
        corrupt_file.write_text("This is plain text and definitely not a valid image format.")

        with pytest.raises(InvalidImageError) as exc_info:
            load_and_preprocess_image(str(corrupt_file))

        assert "Image not found or unreadable" in str(exc_info.value)

    def test_custom_target_size(self, sample_image_path):
        """
        Verify that custom target dimensions (e.g., 128x128) are properly applied.
        """
        custom_size = (128, 128)
        tensor = load_and_preprocess_image(sample_image_path, target_size=custom_size)

        assert tensor.shape == (1, 128, 128, 3)

