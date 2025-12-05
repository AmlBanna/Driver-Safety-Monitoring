#!/usr/bin/env python3
"""
Download large distraction model from GitHub LFS or Google Drive
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm

def download_from_github_lfs():
    """Download model from GitHub LFS"""
    model_path = Path('models/driver_distraction_model.keras')
    
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {file_size:.1f} MB")
        return True
    
    print("="*60)
    print("📥 DOWNLOADING FROM GITHUB LFS")
    print("="*60)
    
    # GitHub raw URL for LFS file
    url = "https://github.com/AmlBanna/Driver-Safety-Monitoring/raw/main/models/driver_distraction_model.keras"
    
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🔗 URL: {url}")
        print("⏳ Downloading... (this may take 3-7 minutes)")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100
                        print(f"\r⏳ Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", 
                              end='', flush=True)
                print()
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Verify
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            if file_size > 50:  # At least 50MB
                print(f"✅ Downloaded successfully: {file_size:.1f} MB")
                return True
            else:
                print(f"❌ File too small: {file_size:.1f} MB")
                os.remove(model_path)
                return False
        
        return False
        
    except Exception as e:
        print(f"\n❌ GitHub download failed: {e}")
        return False

def download_from_google_drive():
    """Fallback: Download from Google Drive"""
    model_path = Path('models/driver_distraction_model.keras')
    
    if model_path.exists():
        return True
    
    print("\n" + "="*60)
    print("📥 FALLBACK: DOWNLOADING FROM GOOGLE DRIVE")
    print("="*60)
    
    try:
        import gdown
        
        file_id = '1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        print("⏳ Downloading from Drive...")
        gdown.download(url, str(model_path), quiet=False)
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            if file_size > 50:
                print(f"✅ Downloaded from Drive: {file_size:.1f} MB")
                return True
        
        return False
        
    except ImportError:
        print("❌ gdown not installed")
        return False
    except Exception as e:
        print(f"❌ Drive download failed: {e}")
        return False

def download_distraction_model():
    """Try GitHub first, then Drive as fallback"""
    
    # Try GitHub LFS
    if download_from_github_lfs():
        return True
    
    # Fallback to Drive
    print("\n🔄 Trying Google Drive as fallback...")
    return download_from_google_drive()

if __name__ == "__main__":
    if download_distraction_model():
        print("\n🎉 Model ready!")
    else:
        print("\n⚠️ Download failed from all sources")
