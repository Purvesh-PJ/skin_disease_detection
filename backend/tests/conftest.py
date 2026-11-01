"""
Pytest Global Test Fixtures
---------------------------
Shared fixtures for Flask test client, mock data, and dummy image generation.
"""

import io
import cv2
import numpy as np
import pytest

from main import create_app


@pytest.fixture(scope="session")
def app():
    """Session-scoped Flask application initialized in test mode."""
    application = create_app(testing=True)
    yield application


@pytest.fixture
def client(app):
    """Flask HTTP test client fixture for executing API requests."""
    return app.test_client()


@pytest.fixture
def dummy_image_bytes():
    """Generates valid encoded JPEG image bytes in-memory."""
    # Create 224x224 RGB image
    img = np.full((224, 224, 3), 180, dtype=np.uint8)
    # Encode as JPEG
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return io.BytesIO(buffer.tobytes())

