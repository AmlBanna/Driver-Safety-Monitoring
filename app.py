#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Driver Safety Dashboard – Drowsiness + Distraction
Uses only: improved_cnn_best.keras + driver_distraction_model.keras
No TFLite | No Caffe | No Extra Files
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from pathlib import Path
import time
import json
from collections import Counter, deque
import base64
import tempfile
import urllib.request
import threading
import queue

# -------------------------- CONFIG --------------------------
BASE_DIR = Path(__file__).parent
DROWSINESS_DIR = BASE_DIR / "drowsiness"
DISTRACTION_DIR = BASE_DIR / "distraction"

# Models
EYE_MODEL_PATH = BASE_DIR / "eye_model"   # ← مجلد من export
DISTRACTION_MODEL_PATH = DISTRACTION_DIR / "driver_distraction_model.keras"
CLASS_IDX_PATH = DISTRACTION_DIR / "class_indices.json"

# Download distraction model if missing
GDRIVE_URL = "https://drive.google.com/uc?id=1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z&export=download"

# -------------------------- HELPERS --------------------------
def download_model():
    if not DISTRACTION_MODEL_PATH.exists():
        with st.spinner("Downloading distraction model (~120 MB)..."):
            urllib.request.urlretrieve(GDRIVE_URL, DISTRACTION_MODEL_PATH)
        st.success("Model downloaded!")

# -------------------------- DROWSINESS (Keras مباشرة) --------------------------
class DrowsinessDetector:
    def __init__(self):
        if not EYE_MODEL_PATH.exists():
            st.error("Folder 'eye_model' not found! Upload it to GitHub.")
            st.stop()
        
        # تحميل SavedModel
        self.model = tf.saved_model.load(str(EYE_MODEL_PATH))
        self.predict_fn = self.model.signatures["serving_default"]
        st.success("Drowsiness model loaded (SavedModel)")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.closed_cnt = 0
        self.THRESH = 5
        self.INPUT_SIZE = (48, 48)

    def preprocess_eye(self, eye):
        eye = cv2.resize(eye, self.INPUT_SIZE)
        eye = eye.astype(np.float32) / 255.0
        return np.expand_dims(eye, axis=-1)

    def detect(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        eye_boxes = []
        eye_imgs = []

        for (x, y, fw, fh) in faces:
            roi = gray[y:y+int(fh*0.65), x:x+fw]
            eyes = self.eye_cascade.detectMultiScale(roi, 1.05, 4, minSize=(20,20))
            for (ex, ey, ew, eh) in eyes:
                if ey > roi.shape[0]*0.55: continue
                eye = roi[ey:ey+eh, ex:ex+ew]
                if min(ew, eh) < 18: continue
                eye_imgs.append(self.preprocess_eye(eye))
                sx = w / small.shape[1]
                sy = h / small.shape[0]
                eye_boxes.append((int((x+ex)*sx), int((y+ey)*sy), int(ew*sx), int(eh*sy)))

        preds = np.array([])
        if eye_imgs:
            batch = np.stack(eye_imgs)
            # تنبؤ باستخدام SavedModel
            input_tensor = tf.constant(batch)
            output = self.predict_fn(input_tensor)
            # المخرج عادةً يكون في مفتاح زي 'output_0' أو 'dense'
            pred_key = list(output.keys())[0]
            preds = output[pred_key].numpy().flatten()

        drowsy = False
        for i, (pred, (x, y, w, h)) in enumerate(zip(preds, eye_boxes)):
            is_open = pred > 0.5
            conf = pred if is_open else 1-pred
            color = (0, 255, 0) if is_open else (0, 0, 255)
            label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if not is_open:
                drowsy = True

        if drowsy:
            self.closed_cnt += 1
        elif eye_boxes:
            self.closed_cnt = max(0, self.closed_cnt - 1)

        alert = self.closed_cnt >= self.THRESH
        return frame, alert, self.closed_cnt
# -------------------------- DISTRACTION --------------------------
class DistractionClassifier:
    def __init__(self):
        download_model()
        self.model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
        with open(CLASS_IDX_PATH) as f:
            idx = json.load(f)
        self.idx2cls = {v: k for k, v in idx.items()}

        self.history = deque(maxlen=8)
        self.frame_cnt = 0

    def _preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        return np.expand_dims(img.astype(np.float32) / 255.0, 0)

    def _final_label(self, cls, conf):
        if cls == 'c6' and conf > 0.30: return 'drinking'
        if cls in ['c1','c2','c3','c4','c9'] and conf > 0.28: return 'using_phone'
        if cls == 'c0' and conf > 0.5: return 'safe_driving'
        if cls == 'c7' and conf > 0.7: return 'turning_back'
        if cls == 'c8' and conf > 0.8: return 'hair_makeup'
        if cls == 'c5' and conf > 0.6: return 'radio'
        return 'other'

    def predict(self, frame):
        self.frame_cnt += 1
        if self.frame_cnt % 2 != 0:
            return self.history[-1] if self.history else ('safe_driving', 0.7)

        x = tf.convert_to_tensor(self._preprocess(frame))
        pred = self.model(x, training=False)[0].numpy()
        idx = np.argmax(pred)
        label = self._final_label(self.idx2cls[idx], pred[idx])
        self.history.append(label)

        if len(self.history) >= 3:
            return Counter(self.history).most_common(1)[0][0], 0.96
        return label, pred[idx]

# -------------------------- LOAD MODELS --------------------------
@st.cache_resource
def load_models():
    d = DrowsinessDetector()
    c = DistractionClassifier()
    return d, c

drowsiness, distraction = load_models()

# -------------------------- THREADS & QUEUES --------------------------
q_d = queue.Queue(maxsize=2)
q_c = queue.Queue(maxsize=2)
result_q = queue.Queue(maxsize=2)

def cam_d_thread():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, f = cap.read()
        if not ret: break
        try: q_d.put_nowait(f)
        except: pass
    cap.release()

def cam_c_thread():
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, f = cap.read()
        if not ret: break
        try: q_c.put_nowait(f)
        except: pass
    cap.release()

def processor():
    while True:
        if not q_d.empty():
            f = q_d.get()
            f, alert, cnt = drowsiness.detect(f.copy())
            result_q.put_nowait(('d', f, alert))
        if not q_c.empty():
            f = q_c.get()
            label, _ = distraction.predict(f.copy())
            cv2.putText(f, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            result_q.put_nowait(('c', f, label))

# -------------------------- UI --------------------------
st.set_page_config(page_title="Driver Safety Dashboard", layout="wide")
st.title("Driver Safety Dashboard")
st.markdown("**Real-time analysis using your models only**")

tab1, tab2 = st.tabs(["Live Cameras", "Upload Media"])

# ========= LIVE =========
with tab1:
    c1, c2 = st.columns(2)
    with c1: cam_d = st.checkbox("Front (Drowsiness)", True)
    with c2: cam_c = st.checkbox("Side (Distraction)", True)

    start = st.button("Start Live", type="primary")
    stop = st.button("Stop")

    ph_d = st.empty()
    ph_c = st.empty()
    alert = st.empty()
    score = st.empty()

    if start:
        if cam_d: threading.Thread(target=cam_d_thread, daemon=True).start()
        if cam_c: threading.Thread(target=cam_c_thread, daemon=True).start()
        threading.Thread(target=processor, daemon=True).start()
        st.session_state.live = True

    if stop and st.session_state.get("live"):
        st.session_state.live = False
        st.rerun()

    if st.session_state.get("live"):
        d_frame, c_frame = None, None
        d_alert, c_label = False, "safe_driving"

        while not result_q.empty():
            typ, f, data = result_q.get()
            if typ == 'd': d_frame, d_alert = f, data
            else: c_frame, c_label = f, data

        if cam_d and d_frame is not None:
            ph_d.image(d_frame, channels="BGR", caption="Front – Drowsiness")
        if cam_c and c_frame is not None:
            ph_c.image(c_frame, channels="BGR", caption="Side – Distraction")

        # Alerts
        msg = ""
        if d_alert:
            msg += "<h2 style='color:red;text-align:center;'>DROWSY!</h2>"
        if c_label == 'turning_back':
            msg += "<h2 style='color:orange;text-align:center;'>LOOKING BACK!</h2>"
        elif c_label in ['using_phone', 'drinking']:
            msg += f"<h3 style='color:#FF4500;text-align:center;'>ALERT: {c_label.upper()}</h3>"
        alert.markdown(msg, unsafe_allow_html=True)

        safe = not d_alert and c_label == 'safe_driving'
        score.metric("Safety Score", f"{100 if safe else 0} %")

# ========= UPLOAD =========
with tab2:
    col1, col2 = st.columns(2)
    with col1: f_d = st.file_uploader("Front Video/Image", ["mp4","avi","jpg","png"])
    with col2: f_c = st.file_uploader("Side Video/Image", ["mp4","avi","jpg","png"])

    if st.button("Analyze", type="primary") and (f_d or f_c):
        tmp_d = tmp_c = None
        if f_d:
            tmp_d = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f_d.name).suffix)
            tmp_d.write(f_d.getvalue()); tmp_d.close()
        if f_c:
            tmp_c = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f_c.name).suffix)
            tmp_c.write(f_c.getvalue()); tmp_c.close()

        cap_d = cv2.VideoCapture(tmp_d.name) if tmp_d else None
        cap_c = cv2.VideoCapture(tmp_c.name) if tmp_c else None

        ph_d = st.empty()
        ph_c = st.empty()
        alert_ph = st.empty()

        drowsy_count = 0
        events = Counter()

        while (cap_d and cap_d.isOpened()) or (cap_c and cap_c.isOpened()):
            if cap_d:
                ret, frame = cap_d.read()
                if ret:
                    frame, alert, _ = drowsiness.detect(frame.copy())
                    if alert: drowsy_count += 1
                    ph_d.image(frame, channels="BGR")
            if cap_c:
                ret, frame = cap_c.read()
                if ret:
                    label, _ = distraction.predict(frame.copy())
                    events[label] += 1
                    cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                    ph_c.image(frame, channels="BGR")

            time.sleep(0.03)

        st.success(f"Analysis Done!\nDrowsy events: {drowsy_count}\nDistractions: {dict(events)}")

        for tmp in (tmp_d, tmp_c):
            if tmp: os.unlink(tmp.name)
        for cap in (cap_d, cap_c):
            if cap: cap.release()
