"""
Unit Tests: Application Configuration
-------------------------------------
Tests environment variable loading, default values, and fallback settings.
"""

from app.core.config import AppConfig


class TestAppConfig:
    """Tests configuration attributes and fallback behavior."""

    def test_default_config_values(self):
        """Verify default configuration values."""
        cfg = AppConfig()

        assert cfg.SECRET_KEY is not None
        assert cfg.MODEL_DIR is not None
        assert cfg.UPLOAD_FOLDER is not None
        assert isinstance(cfg.MODEL_CLOUD_URLS, dict)
        assert len(cfg.MODEL_CLOUD_URLS) == 3

    def test_upload_folder_directory_exists(self):
        """Verify that UPLOAD_FOLDER is initialized on disk."""
        import os
        cfg = AppConfig()
        assert os.path.exists(cfg.UPLOAD_FOLDER)
