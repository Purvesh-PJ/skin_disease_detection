"""
Prediction Service & Endpoint Unit Tests
---------------------------------------
Automated unit tests verifying prediction service handling and error responses.
"""

import unittest
from unittest.mock import patch
from app.core.exceptions import ModelNotFoundError, InvalidImageError
from app.services.prediction_service import load_and_preprocess_image, predict_skin_disease

class TestPredictionService(unittest.TestCase):

    def test_load_and_preprocess_invalid_image(self):
        """Verifies that an invalid image path raises InvalidImageError."""
        with self.assertRaises(InvalidImageError):
            load_and_preprocess_image("non_existent_image_path.jpg")

    @patch('app.services.prediction_service.get_models')
    def test_predict_disease_missing_models(self, mock_get_models):
        """Verifies that ModelNotFoundError is raised when no models are available."""
        mock_get_models.return_value = {}
        with self.assertRaises(ModelNotFoundError):
            predict_skin_disease("some_dummy_path.jpg")

if __name__ == '__main__':
    unittest.main()
