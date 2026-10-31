"""
Prediction Service & Endpoint Unit Tests
---------------------------------------
Automated unit tests verifying prediction service handling and error responses.
"""

import unittest
from unittest.mock import patch
import numpy as np
from app.core.exceptions import ModelNotFoundError, InvalidImageError
from app.services.prediction_service import load_and_preprocess_image, predict_skin_disease

class TestPredictionService(unittest.TestCase):

    def test_load_and_preprocess_invalid_image(self):
        """Verifies that an invalid image path raises InvalidImageError."""
        with self.assertRaises(InvalidImageError):
            load_and_preprocess_image("non_existent_image_path.jpg")

    @patch('app.services.prediction_service.ensure_models_available')
    @patch('app.services.prediction_service.load_and_preprocess_image')
    @patch('os.path.exists', return_value=False)
    def test_predict_disease_missing_models(self, mock_exists, mock_preprocess, mock_ensure):
        """Verifies that ModelNotFoundError is raised when no models are available."""
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        with self.assertRaises(ModelNotFoundError):
            predict_skin_disease("some_dummy_path.jpg")

if __name__ == '__main__':
    unittest.main()
