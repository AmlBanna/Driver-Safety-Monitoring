import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import tempfile
from PIL import Image
import os

# Check and download large model
distract_model = Path('models/driver_distraction_model.keras')
if not distract_model.exists():
    st.info("🔄 First-time setup: Downloading AI model...")
    with st.spinner("⏳ Downloading from GitHub (3-7 minutes)..."):
        try:
            from download_models import download_distraction_model
            if download_distraction_model():
                st.success("✅ Model ready!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Download failed. Please refresh page.")
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# Import detectors
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from utils.alert_system import AlertSystem

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Driver Safety Monitoring",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .alert-critical {
        background-color: #dc3545;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# INITIALIZE
# ================================
if 'drowsy_detector' not in st.session_state:
    st.session_state.drowsy_detector = None
if 'distraction_detector' not in st.session_state:
    st.session_state.distraction_detector = None
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Load models
@st.cache_resource
def load_detectors():
    return DrowsinessDetector(), DistractionDetector()

if st.session_state.drowsy_detector is None:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.drowsy_detector, st.session_state.distraction_detector = load_detectors()

drowsy_det = st.session_state.drowsy_detector
distract_det = st.session_state.distraction_detector
alert_sys = st.session_state.alert_system

# ================================
# HEADER
# ================================
st.markdown('<div class="main-header">🚗 Driver Safety Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666;">Real-time Drowsiness & Distraction Detection with AI</p>', 
            unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio(
        "Analysis Mode",
        ["📹 Live Camera", "📁 Upload Videos (Dual)", "🖼️ Upload Images"],
        key="mode"
    )
    
    st.divider()
    
    # Audio alert toggle
    audio_alert = st.checkbox("🔊 Audio Alerts", value=True)
    
    # Sensitivity
    sensitivity = st.slider("Alert Sensitivity", 1, 10, 5)
    
    st.divider()
    
    # Stats
    st.header("📊 Session Stats")
    stats = alert_sys.get_statistics()
    st.metric("Total Events", stats['total_events'])
    st.metric("Critical Alerts", stats['critical_events'])
    st.metric("Safe Driving %", f"{stats['safe_percentage']:.1f}%")
    
    if st.button("🔄 Reset Stats"):
        alert_sys.reset()
        drowsy_det.reset()
        distract_det.reset()
        st.rerun()

# ================================
# MAIN CONTENT
# ================================

if mode == "📹 Live Camera":
    st.warning("⚠️ Live camera only works locally. Use 'Upload Videos' mode for cloud deployment.")
    
    camera_idx = st.number_input("Camera Index", 0, 5, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Front Camera (Drowsiness)")
        front_frame = st.empty()
    
    with col2:
        st.subheader("Side Camera (Distraction)")
        side_frame = st.empty()
    
    alert_placeholder = st.empty()
    
    start = st.button("▶️ Start Monitoring")
    stop = st.button("⏹️ Stop Monitoring")
    
    if start:
        st.session_state.processing = True
    if stop:
        st.session_state.processing = False
    
    if st.session_state.processing:
        cap = cv2.VideoCapture(camera_idx)
        
        while st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process both
            drowsy_status, drowsy_sev, drowsy_frame = drowsy_det.detect(frame)
            distract_act, distract_sev, distract_frame = distract_det.detect(frame)
            
            # Get alert
            alert_level, alert_msg, combined_sev = alert_sys.evaluate(
                drowsy_status, drowsy_sev, distract_act, distract_sev
            )
            
            # Display
            front_frame.image(cv2.cvtColor(drowsy_frame, cv2.COLOR_BGR2RGB))
            side_frame.image(cv2.cvtColor(distract_frame, cv2.COLOR_BGR2RGB))
            
            # Alert
            color = alert_sys.get_color_for_level(alert_level)
            alert_placeholder.markdown(
                f'<div style="background:{color}; padding:15px; border-radius:8px; color:white; font-weight:bold;">'
                f'{alert_msg}</div>',
                unsafe_allow_html=True
            )
            
            time.sleep(0.03)
        
        cap.release()

elif mode == "📁 Upload Videos (Dual)":
    st.subheader("📹 Upload Two Videos for Complete Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Front Camera (Drowsiness Detection)**")
        front_video = st.file_uploader("Upload front video", type=['mp4', 'avi', 'mov'], key='front')
    
    with col2:
        st.write("**Side Camera (Distraction Detection)**")
        side_video = st.file_uploader("Upload side video", type=['mp4', 'avi', 'mov'], key='side')
    
    if st.button("🎬 Analyze Videos", use_container_width=True):
        if not front_video and not side_video:
            st.error("⚠️ Please upload at least one video")
        else:
            # Save uploaded files
            front_path = None
            side_path = None
            
            if front_video:
                front_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                front_temp.write(front_video.read())
                front_path = front_temp.name
            
            if side_video:
                side_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                side_temp.write(side_video.read())
                side_path = side_temp.name
            
            # Process videos
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Front Camera Analysis")
                front_frame_placeholder = st.empty()
            
            with col2:
                st.subheader("Side Camera Analysis")
                side_frame_placeholder = st.empty()
            
            alert_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Open videos
            front_cap = cv2.VideoCapture(front_path) if front_path else None
            side_cap = cv2.VideoCapture(side_path) if side_path else None
            
            # Get frame count
            total_frames = 0
            if front_cap:
                total_frames = max(total_frames, int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if side_cap:
                total_frames = max(total_frames, int(side_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            
            current_frame = 0
            alert_sys.reset()
            
            while True:
                front_ret = front_cap.read() if front_cap else (False, None)
                side_ret = side_cap.read() if side_cap else (False, None)
                
                if not front_ret[0] and not side_ret[0]:
                    break
                
                current_frame += 1
                
                # Process frames
                drowsy_status, drowsy_sev, drowsy_frame = ('safe', 0, front_ret[1]) if front_ret[0] else ('unknown', 0, None)
                distract_act, distract_sev, distract_frame = ('safe_driving', 0, side_ret[1]) if side_ret[0] else ('unknown', 0, None)
                
                if front_ret[0]:
                    drowsy_status, drowsy_sev, drowsy_frame = drowsy_det.detect(front_ret[1])
                    front_frame_placeholder.image(cv2.cvtColor(drowsy_frame, cv2.COLOR_BGR2RGB))
                
                if side_ret[0]:
                    distract_act, distract_sev, distract_frame = distract_det.detect(side_ret[1])
                    side_frame_placeholder.image(cv2.cvtColor(distract_frame, cv2.COLOR_BGR2RGB))
                
                # Alert
                alert_level, alert_msg, combined_sev = alert_sys.evaluate(
                    drowsy_status, drowsy_sev, distract_act, distract_sev
                )
                
                color = alert_sys.get_color_for_level(alert_level)
                alert_placeholder.markdown(
                    f'<div style="background:{color}; padding:15px; border-radius:8px; color:white; font-weight:bold;">'
                    f'{alert_msg}</div>',
                    unsafe_allow_html=True
                )
                
                progress_bar.progress(min(current_frame / total_frames, 1.0))
                time.sleep(0.01)
            
            # Cleanup
            if front_cap:
                front_cap.release()
            if side_cap:
                side_cap.release()
            
            if front_path:
                os.unlink(front_path)
            if side_path:
                os.unlink(side_path)
            
            # Final stats
            st.success("✅ Analysis Complete!")
            st.divider()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", stats['total_events'])
            with col2:
                st.metric("Critical Alerts", stats['critical_events'])
            with col3:
                st.metric("High Alerts", stats['high_events'])
            with col4:
                st.metric("Safe %", f"{stats['safe_percentage']:.1f}%")

else:  # Images
    st.subheader("🖼️ Upload Images for Analysis")
    
    uploaded_files = st.file_uploader(
        "Upload driver images", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🔍 Analyze Images"):
        cols = st.columns(2)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Detect
            drowsy_status, drowsy_sev, drowsy_frame = drowsy_det.detect(frame)
            distract_act, distract_sev, distract_frame = distract_det.detect(frame)
            
            # Combined display
            combined = np.hstack([drowsy_frame, distract_frame])
            
            with cols[idx % 2]:
                st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), 
                        caption=f"Image {idx+1}", use_container_width=True)
                
                alert_level, alert_msg, _ = alert_sys.evaluate(
                    drowsy_status, drowsy_sev, distract_act, distract_sev
                )
                
                color = alert_sys.get_color_for_level(alert_level)
                st.markdown(
                    f'<div style="background:{color}; padding:10px; border-radius:5px; color:white;">'
                    f'{alert_msg}</div>',
                    unsafe_allow_html=True
                )

# Footer
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.9rem;">
    <p>🚗 Driver Safety Monitoring System | Powered by TensorFlow & OpenCV</p>
</div>
""", unsafe_allow_html=True)
