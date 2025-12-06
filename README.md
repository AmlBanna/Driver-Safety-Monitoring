# 🚗 Driver Safety Monitoring System

An AI-powered real-time driver safety monitoring system that detects **drowsiness** and **distracted driving** using computer vision and deep learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)

---

## 🎯 Features

### 🔴 **Drowsiness Detection** (Front Camera)
- Real-time eye closure monitoring
- Adaptive frame skipping for optimal performance (70-100 FPS)
- Multi-level severity alerts
- DNN-based face detection for accuracy

### 🔵 **Distraction Detection** (Side Camera)
- Detects multiple distraction types:
  - 📱 Phone usage
  - 🍺 Drinking
  - 💄 Hair/makeup
  - 📻 Radio adjustment
  - 🔄 Turning
  - ⚠️ Other activities
- Smoothed predictions for stability
- Confidence-based classification

### 📊 **Comprehensive Analytics**
- Real-time severity tracking
- Statistical summaries
- Interactive charts and graphs
- Alert history logging

### 🎨 **Professional Web Interface**
- Clean, modern UI built with Streamlit
- Multiple input modes:
  - 📹 Live camera feed
  - 📁 Video file upload
  - 🖼️ Batch image processing
- Dual-camera support (front + side)
- Real-time FPS counter

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/driver-safety-system.git
cd driver-safety-system
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models

#### A) Distraction Detection Model (Large File)
Due to GitHub file size limits, download manually:

1. **Download** from Google Drive:  
   [driver_distraction_model.keras](https://drive.google.com/file/d/1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z/view?usp=drive_link)

2. **Place** in `models/` directory:
   ```
   models/driver_distraction_model.keras
   ```

#### B) Face Detection Models (Optional but Recommended)
For improved accuracy, download the DNN face detector:

```bash
cd models
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
cd ..
```

#### C) Verify Model Structure
Your `models/` folder should look like:
```
models/
├── improved_cnn_best.keras              # Drowsiness detection
├── driver_distraction_model.keras       # Distraction detection
├── class_indices.json                   # Distraction classes
├── deploy.prototxt                      # DNN face detector (optional)
└── res10_300x300_ssd_iter_140000.caffemodel  # DNN weights (optional)
```

---

## 🚀 Usage

### Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

#### 1. **Live Camera Mode** 📹
- Select "Live Camera" from sidebar
- Choose camera type:
  - **Front (Drowsiness)**: Monitors eye closure
  - **Side (Distraction)**: Detects distracted behaviors
  - **Both Cameras**: Dual monitoring (split screen)
- Click "Start Monitoring"
- Real-time alerts and statistics appear automatically

#### 2. **Video Analysis** 📁
- Select "Upload Video"
- Choose analysis type
- Upload MP4, AVI, MOV, or MKV file
- Click "Analyze Video"
- View results and final statistics

#### 3. **Image Analysis** 🖼️
- Select "Upload Images"
- Upload multiple images (JPG, PNG)
- Batch analysis with individual results
- Overall statistics summary

---

## 📊 Alert System

### Severity Levels

| Level | Score | Color | Description |
|-------|-------|-------|-------------|
| **Safe** | 0 | 🟢 Green | Normal driving |
| **Low** | 1-3 | 🔵 Blue | Minor distraction |
| **Medium** | 4-6 | 🟡 Yellow | Moderate concern |
| **High** | 7-8 | 🟠 Orange | Significant risk |
| **Critical** | 9-10 | 🔴 Red | Immediate action required |

### Critical Alerts Triggered By:
- 🚨 **Drowsiness**: Eyes closed for 5+ frames
- 🚨 **Drinking**: Detected while driving
- ⚠️ **Phone Usage**: Handheld phone detected
- ⚠️ **Turning**: Looking away from road

---

## 🏗️ Project Structure

```
driver-safety-system/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # AI models (download separately)
│   ├── improved_cnn_best.keras
│   ├── driver_distraction_model.keras
│   ├── class_indices.json
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── utils/                          # Core detection modules
│   ├── drowsiness_detector.py      # Eye closure detection
│   ├── distraction_detector.py     # Activity classification
│   └── alert_system.py             # Alert management
│
└── assets/                         # UI resources (optional)
    └── logo.png
```

---

## ⚙️ Configuration

### Performance Tuning

Edit configuration in detector files:

**`drowsiness_detector.py`:**
```python
INPUT_SIZE = (48, 48)           # Model input size
CLOSED_THRESHOLD = 5            # Frames before drowsy alert
MIN_FACE_SIZE = 80              # Minimum face detection size
BASE_SCALE = 0.7                # Frame downscaling (0.7 = 70%)
```

**`distraction_detector.py`:**
```python
skip = 1                        # Process every Nth frame (1 = every 2nd)
```

### Confidence Thresholds

Adjust in `distraction_detector.py` → `get_final_label()`:
```python
if cls == 'c6' and conf > 0.30:  # Drinking threshold
    return 'drinking'
```

---

## 🧠 Model Details

### Drowsiness Detection
- **Architecture**: Custom CNN
- **Input**: 48x48 grayscale eye images
- **Output**: Binary (open/closed)
- **Performance**: 70-100 FPS

### Distraction Detection
- **Architecture**: Pre-trained CNN (224x224)
- **Classes**: 10 driver activities
- **Smoothing**: 8-frame history buffer
- **Performance**: 30-60 FPS

---

## 🐛 Troubleshooting

### Camera Not Opening
```bash
# Linux: Grant camera permissions
sudo chmod 666 /dev/video0

# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Model Loading Errors
```bash
# Verify model files exist
ls -lh models/

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Low FPS Performance
- Reduce `BASE_SCALE` in drowsiness detector
- Increase `skip` value in distraction detector
- Use TFLite models if available
- Close other applications

### "Model file not found"
Ensure you downloaded the distraction model from Google Drive and placed it in `models/`.

---

## 📈 Performance Metrics

| Configuration | FPS | Accuracy | Latency |
|--------------|-----|----------|---------|
| Drowsiness Only | 70-100 | 95%+ | <15ms |
| Distraction Only | 30-60 | 92%+ | <35ms |
| Both Cameras | 40-60 | 93%+ | <50ms |

*Tested on: Intel i5-8250U, 8GB RAM, 720p webcam*

---


## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** for computer vision tools
- **TensorFlow** for deep learning framework
- **Streamlit** for rapid web app development
- **Haar Cascades** for face/eye detection

---

---



---

<div align="center">
  <p>Made with ❤️ for safer roads</p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>
