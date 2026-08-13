"""
Cloud Model Downloader Utility
------------------------------
Isolated helper for fetching pre-trained model weights from cloud storage.
"""

import os
import urllib.request
import zipfile
import shutil
import logging

logger = logging.getLogger(__name__)

def transform_google_drive_url(url: str) -> str:
    """Transforms Google Drive view/share links into direct raw content download URLs."""
    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        file_id = None
        if "/file/d/" in url:
            parts = url.split("/file/d/")
            if len(parts) > 1:
                file_id = parts[1].split("/")[0].split("?")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]

        if file_id:
            return f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
    return url

def download_file_from_cloud(url: str, dest_path: str) -> bool:
    """Downloads a file from cloud storage URL to local disk."""
    try:
        download_url = transform_google_drive_url(url)
        logger.info(f"Downloading file from cloud: {download_url} -> {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        req = urllib.request.Request(
            download_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Successfully downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
        return False

def download_and_extract_zip(zip_url: str, target_dir: str) -> bool:
    """Downloads and extracts a zip file into target directory."""
    zip_path = os.path.join(target_dir, "models_download.zip")
    if download_file_from_cloud(zip_url, zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Extracted zip into {target_dir}")
            os.remove(zip_path)
            return True
        except Exception as e:
            logger.error(f"Failed to extract zip {zip_path}: {e}")
    return False
