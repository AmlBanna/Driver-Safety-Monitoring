# 🚀 Quick Start Guide

Get the Driver Safety System running in **5 minutes**!

---

## ✅ Step 1: Prerequisites

Make sure you have:
- Python 3.8 or higher
- Webcam (for live monitoring)
- 4GB RAM minimum
- Internet connection (for initial setup)

---

## ✅ Step 2: Installation

### A) Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/driver-safety-system.git
cd driver-safety-system
```

### B) Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Installation takes ~2-5 minutes depending on your connection.

---

## ✅ Step 3: Download the Distraction Model

**IMPORTANT**: This model is too large for GitHub.

1. **Click this link**: [Download Model (Google Drive)](https://drive.google.com/file/d/1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z/view?usp=drive_link)

2. **Save as**: `models/driver_distraction_model.keras`

3. **Verify**:
   ```bash
   ls -lh models/driver_distraction_model.keras
   ```
   Should show ~200-400 MB file.

---

## ✅ Step 4: Run the App

```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The app will **automatically open** in your browser!

---

## 🎮 Step 5: Test the System

### Test 1: **Live Camera** (Easiest)
1. In sidebar, select: **📹 Live Camera**
2. Choose: **Front (Drowsiness)**
3. Click: **▶️ Start Monitoring**
4. Close your eyes for 3 seconds → Should trigger alert!

### Test 2: **Upload Video**
1. Select: **📁 Upload Video**
2. Upload any driving video (MP4, AVI, MOV)
3. Choose: **Both Cameras**
4. Click: **▶️ Analyze Video**

### Test 3: **Image Analysis**
1. Select: **🖼️ Upload Images**
2. Upload 2-3 photos of drivers
3. Click: **🔍 Analyze Images**

---

## 🔧 Troubleshooting

### ❌ "Cannot import name 'DrowsinessDetector'"
**Solution**: Make sure you're in the project root directory:
```bash
pwd  # Should show: .../driver-safety-system
```

### ❌ "Model file not found"
**Solution**: Download the model from Google Drive (Step 3 above)

### ❌ "Camera cannot be opened"
**Solution**: 
- Close other apps using the camera (Zoom, Skype, etc.)
- On Linux: `sudo chmod 666 /dev/video0`
- Try different camera index in code: `cv2.VideoCapture(1)`

### ❌ Low FPS / Laggy
**Solutions**:
- Close browser tabs
- Reduce video resolution
- Edit `drowsiness_detector.py`: change `BASE_SCALE = 0.5`

---

## 📊 Expected Results

### ✅ Good Performance:
- **Drowsiness**: 70-100 FPS
- **Distraction**: 30-60 FPS
- **Combined**: 40-60 FPS
- **Latency**: <50ms

### ⚠️ If Lower:
- Check CPU usage
- Reduce frame processing rate
- Use smaller input size

---

## 🎯 Next Steps

Once running successfully:

1. **Customize Alerts**  
   Edit `utils/alert_system.py` thresholds

2. **Add More Classes**  
   Retrain distraction model with your data

3. **Deploy to Cloud**  
   Use Streamlit Cloud or AWS

4. **Add Audio Alerts**  
   Integrate `pygame` or `playsound`

5. **Export Reports**  
   Add PDF report generation

---

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Model architecture: See `utils/` folder
- Configuration options: Check detector files

---

## 💡 Tips

- **Best lighting**: Natural daylight or bright indoor
- **Camera position**: Eye-level, 1-2 feet from face
- **First run**: Models load slower (~10 seconds)
- **False positives**: Adjust confidence thresholds in code

---

## 🆘 Still Having Issues?

1. Check [GitHub Issues](https://github.com/YOUR_USERNAME/driver-safety-system/issues)
2. Open a new issue with:
   - Error message
   - Python version (`python --version`)
   - OS (Windows/Mac/Linux)
   - Screenshot if possible

---

<div align="center">
  <p>🎉 <b>You're all set!</b> Enjoy safer driving! 🚗</p>
</div>
