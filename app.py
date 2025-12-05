#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Driver Safety Dashboard – Drowsiness + Distraction
Streamlit | Real-time | Audio alerts | Summary
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
from PIL import Image
import queue

# -------------------------- CONFIG --------------------------
BASE_DIR = Path(__file__).parent
DROWSINESS_DIR = BASE_DIR / "drowsiness"
DISTRACTION_DIR = BASE_DIR / "distraction"
ASSETS_DIR = BASE_DIR / "assets"

# Eye model (TFLite first, Keras fallback)
TFLITE_PATH = DROWSINESS_DIR / "model.tflite"
KERAS_EYE_PATH = DROWSINESS_DIR / "improved_cnn_best.keras"

# Distraction model (download if missing)
DISTRACTION_MODEL_PATH = DISTRACTION_DIR / "driver_distraction_model.keras"
CLASS_IDX_PATH = DISTRACTION_DIR / "class_indices.json"
GDRIVE_URL = "https://drive.google.com/uc?id=1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z&export=download"

ALERT_SOUND = ASSETS_DIR / "alert.wav"

# -------------------------- HELPERS --------------------------
def download_file(url, dest):
    if dest.exists():
        return
    with st.spinner(f"Downloading large model (~120 MB)…"):
        urllib.request.urlretrieve(url, dest)

def load_sound():
    if ALERT_SOUND.exists():
        with open(ALERT_SOUND, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio autoplay="true"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    return ""

# -------------------------- DROWSINESS ENGINE --------------------------
class DrowsinessDetector:
    def __init__(self):
        # ---- TFLite (fast) ----
        if TFLITE_PATH.exists():
            self.interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH), num_threads=4)
            self.interpreter.allocate_tensors()
            self.input_idx = self.interpreter.get_input_details()[0]['index']
            self.output_idx = self.interpreter.get_output_details()[0]['index']
            self.predict = self._predict_tflite
            st.success("Drowsiness: TFLite loaded")
        else:
            model = tf.keras.models.load_model(str(KERAS_EYE_PATH))
            self.predict = lambda batch: model(batch, training=False).numpy().flatten()
            st.warning("Drowsiness: Keras fallback")

        # ---- Face detector (Haar only – no Caffe needed) ----
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.closed_cnt = 0
        self.THRESH = 5
        self.INPUT_SIZE = (48, 48)

    def _predict_tflite(self, batch):
        if batch.size == 0:
            return np.array([])
        self.interpreter.set_tensor(self.input_idx, batch.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_idx).flatten()

    def preprocess_eye(self, eye):
        eye = cv2.resize(eye, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        eye = eye.astype(np.float32) / 255.0
        return np.expand_dims(eye, axis=-1)

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = 0.7
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ---- faces (Haar) ----
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        eye_boxes = []      # original coordinates
        eye_imgs = []
        for (x, y, fw, fh) in faces:
            roi = gray[y:y+int(fh*0.65), x:x+fw]
            eyes = self.eye_cascade.detectMultiScale(
                roi, 1.05, 4, minSize=(20,20), maxSize=(80,80))
            for (ex, ey, ew, eh) in eyes:
                if ey > roi.shape[0]*0.55: continue
                eye = roi[ey:ey+eh, ex:ex+ew]
                if eye.size == 0 or min(ew, eh) < 18: continue
                eye_imgs.append(self.preprocess_eye(eye))
                sx = w / small.shape[1]
                sy = h / small.shape[0]
                eye_boxes.append((
                    int((x + ex) * sx),
                    int((y + ey) * sy),
                    int(ew * sx),
                    int(eh * sy)
                ))

        preds = np.array([])
        if eye_imgs:
            batch = np.stack(eye_imgs)
            preds = self.predict(batch)

        drowsy = False
        for i, (pred, box) in enumerate(zip(preds, eye_boxes)):
            open_eye = pred > 0.5
            conf = pred if open_eye else 1-pred
            color = (0, 255, 0) if open_eye else (0, 0, 255)
            label = f"{'OPEN' if open_eye else 'CLOSED'} {conf:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if not open_eye:
                drowsy = True

        if drowsy:
            self.closed_cnt += 1
        elif eye_boxes:
            self.closed_cnt = max(0, self.closed_cnt - 1)

        alert = self.closed_cnt >= self.THRESH
        return frame, alert, self.closed_cnt

# -------------------------- DISTRACTION ENGINE --------------------------
class DistractionClassifier:
    def __init__(self):
        if not DISTRACTION_MODEL_PATH.exists():
            download_file(GDRIVE_URL, DISTRACTION_MODEL_PATH)

        self.model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
        with open(CLASS_IDX_PATH) as f:
            idx = json.load(f)
        self.idx2cls = {v: k for k, v in idx.items()}

        self.history = deque(maxlen=8)
        self.frame_skip = 1
        self.frame_cnt = 0

    def _preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

    def _final_label(self, cls, conf):
        if cls == 'c6' and conf > 0.30: return 'drinking'
        if cls in ['c1','c2','c3','c4','c9'] and conf > 0.28: return 'using_phone'
        if cls == 'c0' and conf > 0.5: return 'safe_driving'
        if cls == 'c7' and conf > 0.7: return 'turning_back'   # <-- big alert
        if cls == 'c8' and conf > 0.8: return 'hair_makeup'
        if cls == 'c5' and conf > 0.6: return 'radio'
        return 'other'

    def predict(self, frame):
        self.frame_cnt += 1
        if self.frame_cnt % (self.frame_skip + 1) != 0:
            if self.history:
                return Counter(self.history).most_common(1)[0][0], 0.95
            return 'safe_driving', 0.7

        x = tf.convert_to_tensor(self._preprocess(frame))
        pred = self.model(x, training=False)[0].numpy()
        idx = np.argmax(pred)
        cls = self.idx2cls[idx]
        conf = pred[idx]

        label = self._final_label(cls, conf)
        self.history.append(label)

        if len(self.history) >= 3:
            return Counter(self.history).most_common(1)[0][0], 0.96
        return label, conf

# -------------------------- GLOBAL STATE --------------------------
@st.cache_resource
def get_detectors():
    d = DrowsinessDetector()
    c = DistractionClassifier()
    return d, c

drowsiness_det, distraction_clf = get_detectors()

# queues for live threads
frame_q_d = queue.Queue(maxsize=2)
frame_q_c = queue.Queue(maxsize=2)
result_q = queue.Queue(maxsize=2)

# -------------------------- THREADS --------------------------
def cam_thread(cap_idx, q):
    cap = cv2.VideoCapture(cap_idx)
    while cap.isOpened():
        ret, f = cap.read()
        if not ret: break
        try:
            q.put_nowait(f)
        except queue.Full:
            pass
    cap.release()

def processor_thread():
    while True:
        if not frame_q_d.empty():
            f = frame_q_d.get()
            f, alert, cnt = drowsiness_det.detect(f.copy())
            result_q.put_nowait(('drowsy', f, alert, cnt))
        if not frame_q_c.empty():
            f = frame_q_c.get()
            label, _ = distraction_clf.predict(f.copy())
            cv2.putText(f, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            result_q.put_nowait(('dist', f, label))

# -------------------------- UI --------------------------
st.set_page_config(page_title="Driver Safety Dashboard", layout="wide")
st.title("Driver Safety Dashboard")
st.markdown("**Real-time drowsiness + distraction detection** – live cams, uploaded media, audio alerts")

tab_live, tab_upload = st.tabs(["Live Cameras", "Upload Media"])

# ====================== LIVE ======================
with tab_live:
    col1, col2 = st.columns(2)
    with col1:
        cam_d = st.checkbox("Front camera (drowsiness)", value=True)
    with col2:
        cam_c = st.checkbox("Side camera (distraction)", value=True)

    start_btn = st.button("Start Live Analysis", type="primary")
    stop_btn = st.button("Stop", type="secondary")

    ph_d = st.empty()
    ph_c = st.empty()
    alert_ph = st.empty()
    score_ph = st.empty()

    if start_btn:
        if cam_d: threading.Thread(target=cam_thread, args=(0, frame_q_d), daemon=True).start()
        if cam_c: threading.Thread(target=cam_thread, args=(1, frame_q_c), daemon=True).start()
        threading.Thread(target=processor_thread, daemon=True).start()
        st.session_state.running = True

    if stop_btn and st.session_state.get("running"):
        st.session_state.running = False
        st.rerun()

    if st.session_state.get("running"):
        drowsy_f = dist_f = None
        drowsy_alert = False
        dist_label = "safe_driving"

        while not result_q.empty():
            typ, f, *rest = result_q.get()
            if typ == 'drowsy':
                drowsy_f, drowsy_alert, _ = f, *rest
            else:
                dist_f, dist_label = f, rest[0]

        if cam_d and drowsy_f is not None:
            ph_d.image(drowsy_f, channels="BGR", caption="Front – Drowsiness")
        if cam_c and dist_f is not None:
            ph_c.image(dist_f, channels="BGR", caption="Side – Distraction")

        # ----- ALERTS -----
        html = ""
        if drowsy_alert:
            html += load_sound() + "<h2 style='color:red;text-align:center;'>DROWSY – WAKE UP!</h2>"
        if dist_label == 'turning_back':
            html += load_sound() + "<h2 style='color:orange;text-align:center;'>LOOKING BACK – DANGER!</h2>"
        elif dist_label in ['using_phone', 'drinking']:
            html += load_sound() + f"<h3 style='color:#FF4500;text-align:center;'>DISTRACTION: {dist_label.upper()}</h3>"
        elif dist_label not in ['safe_driving', 'other']:
            html += f"<p style='color:#FFD700;text-align:center;'>Warning: {dist_label.replace('_',' ').title()}</p>"

        alert_ph.markdown(html, unsafe_allow_html=True)

        safe = 1 if not drowsy_alert and dist_label == 'safe_driving' else 0
        score_ph.metric("Safety Score", f"{int(safe*100)} %")

# ====================== UPLOAD ======================
with tab_upload:
    st.subheader("Upload one or two videos / images")
    colA, colB = st.columns(2)
    with colA:
        file_d = st.file_uploader("Front (drowsiness)", type=["mp4","avi","mov","jpg","png"])
    with colB:
        file_c = st.file_uploader("Side (distraction)", type=["mp4","avi","mov","jpg","png"])

    analyse_btn = st.button("Analyse Uploaded Media", type="primary")
    out_ph = st.empty()

    if analyse_btn and (file_d or file_c):
        tmp_d = tmp_c = None
        if file_d:
            tmp_d = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_d.name).suffix)
            tmp_d.write(file_d.getvalue())
            tmp_d.close()
        if file_c:
            tmp_c = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_c.name).suffix)
            tmp_c.write(file_c.getvalue())
            tmp_c.close()

        caps = {}
        if tmp_d: caps['d'] = cv2.VideoCapture(tmp_d.name)
        if tmp_c: caps['c'] = cv2.VideoCapture(tmp_c.name)

        ph_d = st.empty()
        ph_c = st.empty()
        alert_ph = st.empty()

        drowsy_total = 0
        dist_events = Counter()

        while any(c.isOpened() for c in caps.values()):
            for key, cap in caps.items():
                ret, frame = cap.read()
                if not ret: continue

                if key == 'd':
                    frame, alert, _ = drowsiness_det.detect(frame.copy())
                    if alert: drowsy_total += 1
                    ph_d.image(frame, channels="BGR")
                else:
                    label, _ = distraction_clf.predict(frame.copy())
                    dist_events[label] += 1
                    cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                    ph_c.image(frame, channels="BGR")

                if alert:
                    alert_ph.markdown("<h2 style='color:red;'>DROWSY</h2>", unsafe_allow_html=True)
                if label == 'turning_back':
                    alert_ph.markdown("<h2 style='color:orange;'>LOOKING BACK</h2>", unsafe_allow_html=True)

            time.sleep(0.03)

        report = f"""
        ### Summary
        - **Drowsiness events**: {drowsy_total}
        - **Distraction breakdown**: {dict(dist_events)}
        """
        out_ph.markdown(report)

        for tmp in (tmp_d, tmp_c):
            if tmp: os.unlink(tmp.name)
        for c in caps.values():
            c.release()

st.success("Dashboard ready – use Live or Upload tabs")
