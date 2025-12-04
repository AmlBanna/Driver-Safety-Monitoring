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

# Import custom modules
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from utils.alert_system import AlertSystem

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Driver Safety Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #dc3545;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# INITIALIZE SESSION STATE
# ================================
if 'drowsy_detector' not in st.session_state:
    st.session_state.drowsy_detector = None
if 'distraction_detector' not in st.session_state:
    st.session_state.distraction_detector = None
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_detectors():
    st.info("🔄 Loading AI models... This may take a moment.")
    drowsy = DrowsinessDetector()
    distract = DistractionDetector()
    st.success("✅ Models loaded successfully!")
    return drowsy, distract

# ================================
# HELPER FUNCTIONS
# ================================
def create_metrics_dashboard(stats):
    """Create beautiful metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", stats['total_events'])
    with col2:
        st.metric("Critical Alerts", stats['critical_events'], 
                 delta=f"-{stats['critical_events']}" if stats['critical_events'] > 0 else None,
                 delta_color="inverse")
    with col3:
        st.metric("Safe Driving %", f"{stats['safe_percentage']:.1f}%",
                 delta=f"+{stats['safe_percentage']:.1f}%" if stats['safe_percentage'] > 70 else None)
    with col4:
        st.metric("Avg Risk Level", f"{stats['avg_severity']:.1f}/10",
                 delta_color="inverse")

def create_severity_chart(alert_system):
    """Create real-time severity chart"""
    if not alert_system.alert_history:
        return None
    
    history = alert_system.alert_history[-50:]  # Last 50 events
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Drowsiness Severity', 'Distraction Severity'),
        vertical_spacing=0.15
    )
    
    # Drowsiness
    fig.add_trace(
        go.Scatter(
            y=[h['drowsy_severity'] for h in history],
            mode='lines+markers',
            name='Drowsiness',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )
    
    # Distraction
    fig.add_trace(
        go.Scatter(
            y=[h['distraction_severity'] for h in history],
            mode='lines+markers',
            name='Distraction',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    fig.update_yaxes(range=[0, 10], title_text="Severity", row=1, col=1)
    fig.update_yaxes(range=[0, 10], title_text="Severity", row=2, col=1)
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def process_frame_combined(frame, drowsy_det, distract_det, alert_sys, camera_type):
    """Process frame with both detectors"""
    if camera_type == 'Front (Drowsiness)':
        status, severity, annotated = drowsy_det.detect(frame)
        alert_level, alert_msg, _ = alert_sys.evaluate(status, severity, 'safe_driving', 0)
        return annotated, alert_level, alert_msg, severity
    
    elif camera_type == 'Side (Distraction)':
        activity, severity, annotated = distract_det.detect(frame)
        alert_level, alert_msg, _ = alert_sys.evaluate('safe', 0, activity, severity)
        return annotated, alert_level, alert_msg, severity
    
    else:  # Both cameras
        # Split frame horizontally
        h, w = frame.shape[:2]
        left_frame = frame[:, :w//2]
        right_frame = frame[:, w//2:]
        
        # Process both
        drowsy_status, drowsy_sev, drowsy_anno = drowsy_det.detect(left_frame)
        distract_act, distract_sev, distract_anno = distract_det.detect(right_frame)
        
        # Combine frames
        combined = np.hstack([drowsy_anno, distract_anno])
        
        # Get alert
        alert_level, alert_msg, combined_sev = alert_sys.evaluate(
            drowsy_status, drowsy_sev, distract_act, distract_sev
        )
        
        return combined, alert_level, alert_msg, combined_sev

# ================================
# MAIN APP
# ================================
st.markdown('<div class="main-header">🚗 Driver Safety Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Drowsiness & Distraction Detection</div>', unsafe_allow_html=True)

# Load models
if st.session_state.drowsy_detector is None:
    st.session_state.drowsy_detector, st.session_state.distraction_detector = load_detectors()

drowsy_det = st.session_state.drowsy_detector
distract_det = st.session_state.distraction_detector
alert_sys = st.session_state.alert_system

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    input_type = st.radio(
        "Input Source",
        ["📹 Live Camera", "📁 Upload Video", "🖼️ Upload Images"],
        key="input_type"
    )
    
    if input_type == "📹 Live Camera":
        camera_type = st.selectbox(
            "Camera Type",
            ["Front (Drowsiness)", "Side (Distraction)", "Both Cameras"],
            key="camera_type"
        )
    else:
        camera_type = st.selectbox(
            "Analysis Type",
            ["Front (Drowsiness)", "Side (Distraction)", "Both Cameras"],
            key="analysis_type"
        )
    
    st.divider()
    
    # Statistics
    st.header("📊 Live Statistics")
    stats = alert_sys.get_statistics()
    st.metric("Total Frames", stats['total_events'])
    st.metric("Critical Alerts", stats['critical_events'])
    st.metric("Safe %", f"{stats['safe_percentage']:.1f}%")
    
    if st.button("🔄 Reset Statistics"):
        alert_sys.reset()
        drowsy_det.reset()
        distract_det.reset()
        st.rerun()

# ================================
# MAIN CONTENT
# ================================

# === LIVE CAMERA ===
if input_type == "📹 Live Camera":
    st.subheader("🔴 Live Camera Feed")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        alert_placeholder = st.empty()
        severity_placeholder = st.empty()
    
    chart_placeholder = st.empty()
    
    start_btn = st.button("▶️ Start Monitoring", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Monitoring", use_container_width=True)
    
    if start_btn:
        st.session_state.processing = True
    if stop_btn:
        st.session_state.processing = False
    
    if st.session_state.processing:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check permissions.")
            st.session_state.processing = False
        else:
            fps_time = time.time()
            fps = 0
            
            while st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame")
                    break
                
                # Process frame
                annotated, alert_level, alert_msg, severity = process_frame_combined(
                    frame, drowsy_det, distract_det, alert_sys, camera_type
                )
                
                # Calculate FPS
                fps = 1 / (time.time() - fps_time + 1e-6)
                fps_time = time.time()
                
                # Add FPS overlay
                cv2.putText(annotated, f"FPS: {fps:.1f}", (annotated.shape[1]-120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                      use_container_width=True)
                
                # Alert display
                alert_color = alert_sys.get_color_for_level(alert_level)
                alert_placeholder.markdown(
                    f'<div style="background-color: {alert_color}; padding: 15px; '
                    f'border-radius: 8px; color: white; font-weight: bold;">'
                    f'{alert_msg}</div>',
                    unsafe_allow_html=True
                )
                
                severity_placeholder.progress(severity / 10, text=f"Risk: {severity}/10")
                
                # Update chart every 10 frames
                if stats['total_events'] % 10 == 0:
                    fig = create_severity_chart(alert_sys)
                    if fig:
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.01)
            
            cap.release()

# === UPLOAD VIDEO ===
elif input_type == "📁 Upload Video":
    st.subheader("📹 Video Analysis")
    
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            frame_placeholder = st.empty()
        with col2:
            alert_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        chart_placeholder = st.empty()
        
        if st.button("▶️ Analyze Video", use_container_width=True):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            alert_sys.reset()
            drowsy_det.reset()
            distract_det.reset()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame += 1
                
                # Process every 3rd frame for speed
                if current_frame % 3 != 0:
                    continue
                
                annotated, alert_level, alert_msg, severity = process_frame_combined(
                    frame, drowsy_det, distract_det, alert_sys, camera_type
                )
                
                # Display
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                      use_container_width=True)
                
                alert_color = alert_sys.get_color_for_level(alert_level)
                alert_placeholder.markdown(
                    f'<div style="background-color: {alert_color}; padding: 15px; '
                    f'border-radius: 8px; color: white; font-weight: bold;">'
                    f'{alert_msg}</div>',
                    unsafe_allow_html=True
                )
                
                progress_bar.progress(current_frame / total_frames)
                
                time.sleep(0.01)
            
            cap.release()
            
            # Final statistics
            st.success("✅ Video analysis complete!")
            st.divider()
            st.subheader("📊 Analysis Summary")
            create_metrics_dashboard(alert_sys.get_statistics())
            
            fig = create_severity_chart(alert_sys)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# === UPLOAD IMAGES ===
else:
    st.subheader("🖼️ Image Analysis")
    
    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🔍 Analyze Images", use_container_width=True):
            alert_sys.reset()
            drowsy_det.reset()
            distract_det.reset()
            
            cols = st.columns(2)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                annotated, alert_level, alert_msg, severity = process_frame_combined(
                    frame, drowsy_det, distract_det, alert_sys, camera_type
                )
                
                with cols[idx % 2]:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"Image {idx+1}", use_container_width=True)
                    
                    alert_color = alert_sys.get_color_for_level(alert_level)
                    st.markdown(
                        f'<div style="background-color: {alert_color}; padding: 10px; '
                        f'border-radius: 5px; color: white; font-size: 0.9rem;">'
                        f'{alert_msg}</div>',
                        unsafe_allow_html=True
                    )
            
            st.divider()
            st.subheader("📊 Overall Analysis")
            create_metrics_dashboard(alert_sys.get_statistics())

# ================================
# FOOTER
# ================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>🚗 Driver Safety Monitoring System v1.0</p>
    <p>Powered by TensorFlow & OpenCV | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
