#!/usr/bin/env python3
"""
Auto-download large model files from Google Drive
"""

import os
from pathlib import Path
import requests

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        print("\nDownload complete!")

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_drowsiness_model():
    """Download the drowsiness detection model"""
    
    model_path = Path('models/improved_cnn_best.keras')
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Drowsiness model already exists: {model_path}")
        return True
    
    print("📥 Downloading drowsiness detection model...")
    print("⏳ This may take 1-2 minutes...")
    
    # Add your drowsiness model Google Drive file ID here
    # Get it from: https://drive.google.com/file/d/FILE_ID/view
    file_id = 'PUT_YOUR_DROWSINESS_MODEL_FILE_ID_HERE'  # ← حطي الـ ID هنا
    
    try:
        # Create models directory if not exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        download_file_from_google_drive(file_id, str(model_path))
        
        # Verify download
        if model_path.exists() and model_path.stat().st_size > 100000:  # > 100KB
            print(f"✅ Drowsiness model downloaded successfully: {model_path}")
            return True
        else:
            print("❌ Download failed or file is too small")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading drowsiness model: {e}")
        return False

def download_distraction_model():
    """Download the distraction detection model"""
    
    model_path = Path('models/driver_distraction_model.keras')
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ Distraction model already exists: {model_path}")
        return True
    
    print("📥 Downloading distraction detection model from Google Drive...")
    print("⏳ This may take 2-5 minutes (file size: ~400MB)...")
    
    # Your Google Drive file ID
    file_id = '1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z'
    
    try:
        # Create models directory if not exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        download_file_from_google_drive(file_id, str(model_path))
        
        # Verify download
        if model_path.exists() and model_path.stat().st_size > 1000000:  # > 1MB
            print(f"✅ Distraction model downloaded successfully: {model_path}")
            return True
        else:
            print("❌ Download failed or file is too small")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading distraction model: {e}")
        return False

def download_all_models():
    """Download all required models"""
    print("🔄 Checking and downloading required models...")
    
    drowsy_ok = download_drowsiness_model()
    distract_ok = download_distraction_model()
    
    if drowsy_ok and distract_ok:
        print("✅ All models ready!")
        return True
    else:
        print("⚠️ Some models failed to download")
        return False

if __name__ == "__main__":
    download_all_models()
