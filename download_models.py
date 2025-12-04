#!/usr/bin/env python3
"""
Auto-download large distraction model from Google Drive
Small models (drowsiness) are already on GitHub
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
        
        print(f"📦 File size: {total_size / (1024*1024):.1f} MB")
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r⏳ Downloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", 
                              end='', flush=True)
        print("\n✅ Download complete!")

    URL = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_distraction_model():
    """Download the large distraction detection model from Google Drive"""
    
    model_path = Path('models/driver_distraction_model.keras')
    
    # Check if model already exists
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Distraction model already exists: {model_path} ({file_size:.1f} MB)")
        return True
    
    print("="*60)
    print("📥 DOWNLOADING DISTRACTION DETECTION MODEL")
    print("="*60)
    print("🔗 Source: Google Drive")
    print("📦 Expected size: ~400 MB")
    print("⏱️ Estimated time: 2-5 minutes")
    print("="*60)
    
    # Google Drive file ID from your link
    file_id = '1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z'
    
    try:
        # Create models directory if not exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        download_file_from_google_drive(file_id, str(model_path))
        
        # Verify download
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            if file_size > 10:  # At least 10MB
                print(f"✅ Model downloaded successfully: {model_path}")
                print(f"📦 Final size: {file_size:.1f} MB")
                return True
            else:
                print(f"❌ Downloaded file is too small ({file_size:.1f} MB)")
                print("⚠️ The file might be corrupted. Please try again.")
                return False
        else:
            print("❌ Download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("💡 Tip: Check your internet connection and try again")
        return False

if __name__ == "__main__":
    success = download_distraction_model()
    if success:
        print("\n🎉 All models ready! You can now run the app.")
    else:
        print("\n⚠️ Download failed. Please check the error above and try again.")
