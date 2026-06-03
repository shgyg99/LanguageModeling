# utils/google_drive_downloader.py
import os
import torch
import gdown
from pathlib import Path


class GoogleDriveDownloader:
    MODEL_FILE_ID = "1vi9qU9fI0f98ucZagtvKLv11phfoC5Ie"

    @classmethod
    def get_model_path(cls, local_path):

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path):
            print(f"✅ Model found locally: {local_path}")
            return local_path

        print(f"⚠️ Local model not found. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={cls.MODEL_FILE_ID}"

        try:
            gdown.download(url, local_path, quiet=False, resume=True)
            print(f"✅ Model downloaded successfully to: {local_path}")
            return local_path
        except Exception as e:
            print(f"❌ ERROR: Could not download model from Google Drive. Reason: {e}")
            raise e
