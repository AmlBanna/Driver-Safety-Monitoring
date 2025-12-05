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
from io import BytesIO
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

# TFLite eye model (fast)
TFLITE_PATH = DROWSINESS_DIR / "model.tflite"
KERAS_EYE_PATH = DROWSINESS_DIR / "improved_cnn_best.keras"

# Distraction model (large – download if missing)
DISTRACTION_MODEL_PATH = DISTRACTION_DIR / "driver_distraction_model.keras"
CLASS_IDX_PATH = DISTRACTION_DIR / "class_indices.json"
GDRIVE_URL = "https://drive.google.com/uc?id=1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z&export=download"

# DNN face detector
PROTO = DROWSINESS_DIR / "models" / "deploy.prototxt"
WEIGHTS = DROWSINESS_DIR / "models" / "res10_300x300_ssd_iter_140000.caffemodel"

ALERT_SOUND = ASSETS_DIR / "alert.wav"

# -------------------------- HELPERS --------------------------
def download_file(url, dest):
    if dest.exists():
        return
    st.info(f"Downloading large model (~120 MB)…")
    urllib.request.urlretrieve(url, dest)
    st.success("Download finished")

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
        # TFLite first
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

        # DNN face detector
        if PROTO.exists() and WEIGHTS.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(PROTO), str(WEIGHTS))
            self.use_dnn = True
        else:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.use_dnn = False

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

        # ---- face ----
        faces = []
        if self.use_dnn:
            blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            dets = self.net.forward()
            for i in range(dets.shape[2]):
                conf = dets[0, 0, i, 2]
                if conf < 0.5: continue
                box = dets[0, 0, i, 3:7] * np.array([small.shape[1], small.shape[0]] * 2)
                x1, y1, x2, y2 = box.astype(int)
                if min(x2-x1, y2-y1) < 80: continue
                faces.append((x1, y1, x2-x1, y2-y1))
        else:
            faces = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))

        eye_boxes = []      # original size
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
                # back to original
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

        # ---- draw + logic ----
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
        # download if missing
        if not DISTRACTION_MODEL_PATH.exists():
            download_file(GDRIVE_URL, DISTRACTION_MODEL_PATH)

        self.model = tf.keras.models.load_model(str(DISTRACTION_MODEL_PATH))
        with open(CLASS_IDX_PATH) as f:
            idx = json.load(f)
        self.idx2cls = {v: k for k, v in idx.items()}

        self.history = deque(maxlen=8)
        self.frame_skip = 1   # process every 2nd frame
        self.frame_cnt = 0

    def _preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

    def _final_label(self, cls, conf):
        # priority rules (same as your original code)
        if cls == 'c6' and conf > 0.30: return 'drinking'
        if cls in ['c1','c2','c3','c4','c9'] and conf > 0.28: return 'using_phone'
        if cls == 'c0' and conf > 0.5: return 'safe_driving'
        if cls == 'c7' and conf > 0.7: return 'turning_back'      # <-- our "looking back"
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
frame_queue_d = queue.Queue(maxsize=2)
frame_queue_c = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# -------------------------- PROCESSING THREADS --------------------------
def drowsiness_thread(cap_idx):
    cap = cv2.VideoCapture(cap_idx)
    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            frame_queue_d.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

def distraction_thread(cap_idx):
    cap = cv2.VideoCapture(cap_idx)
    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            frame_queue_c.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

def processor_thread():
    while True:
        # ---- drowsiness ----
        if not frame_queue_d.empty():
            f = frame_queue_d.get()
            f, alert, cnt = drowsiness_det.detect(f.copy())
            result_queue.put_nowait(('drowsy', f, alert, cnt))

        # ---- distraction ----
        if not frame_queue_c.empty():
            f = frame_queue_c.get()
            label, conf = distraction_clf.predict(f.copy())
            result_queue.put_nowait(('dist', f, label, conf))

# -------------------------- UI --------------------------
st.set_page_config(page_title="Driver Safety Dashboard", layout="wide")
st.title("Driver Safety Dashboard")
st.markdown("**Real-time drowsiness + distraction detection** – live cams, uploaded media, audio alerts")

tab_live, tab_upload = st.tabs(["Live Cameras", "Upload Media"])

# ====================== LIVE TAB ======================
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
    summary_ph = st.empty()

    if start_btn:
        # launch threads
        if cam_d:
            threading.Thread(target=drowsiness_thread, args=(0,), daemon=True).start()
        if cam_c:
            threading.Thread(target=distraction_thread, args=(1,), daemon=True).start()
        threading.Thread(target=processor_thread, daemon=True).start()

        st.session_state.running = True

    if stop_btn and st.session_state.get("running"):
        st.session_state.running = False
        st.rerun()

    if st.session_state.get("running"):
        # pull latest results
        drowsy_frame = None
        dist_frame = None
        drowsy_alert = False
        dist_label = "safe_driving"

        while not result_queue.empty():
            typ, frm, *rest = result_queue.get()
            if typ == 'drowsy':
                drowsy_frame, drowsy_alert, closed_cnt = frm, *rest
            else:
                dist_frame, dist_label, _ = frm, *rest

        # ---- display ----
        if cam_d and drowsy_frame is not None:
            ph_d.image(drowsy_frame, channels="BGR", caption="Front – Drowsiness")
        if cam_c and dist_frame is not None:
            ph_c.image(dist_frame, channels="BGR", caption="Side – Distraction")

        # ---- alerts ----
        alert_html = ""
        if drowsy_alert:
            alert_html += load_sound()
            alert_html += "<h2 style='color:red;text-align:center;'>DROWSY – WAKE UP!</h2>"
        if dist_label == 'turning_back':
            alert_html += load_sound()
            alert_html += "<h2 style='color:orange;text-align:center;'>LOOKING BACK – DANGER!</h2>"
        elif dist_label in ['using_phone', 'drinking']:
            alert_html += load_sound()
            alert_html += f"<h3 style='color:#FF4500;text-align:center;'>DISTRACTION: {dist_label.upper()}</h3>"
        elif dist_label not in ['safe_driving', 'other']:
            alert_html += f"<p style='color:#FFD700;text-align:center;'>Warning: {dist_label.replace('_',' ').title()}</p>"

        alert_ph.markdown(alert_html, unsafe_allow_html=True)

        # ---- summary ----
        safe = 1 if not drowsy_alert and dist_label == 'safe_driving' else 0
        summary_ph.metric("Driver Safety Score", f"{int(safe*100)} %")

# ====================== UPLOAD TAB ======================
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
        # ---- save temp files ----
        tmp_d = None
        tmp_c = None
        if file_d:
            tmp_d = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_d.name).suffix)
            tmp_d.write(file_d.getvalue())
            tmp_d.close()
        if file_c:
            tmp_c = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_c.name).suffix)
            tmp_c.write(file_c.getvalue())
            tmp_c.close()

        # ---- process video(s) ----
        caps = {}
        if tmp_d: caps['d'] = cv2.VideoCapture(tmp_d.name)
        if tmp_c: caps['c'] = cv2.VideoCapture(tmp_c.name)

        frame_d_placeholder = st.empty()
        frame_c_placeholder = st.empty()
        alert_placeholder = st.empty()

        drowsy_cnt_total = 0
        dist_events = Counter()

        while any(cap.isOpened() for cap in caps.values()):
            for key, cap in caps.items():
                ret, frame = cap.read()
                if not ret: continue

                if key == 'd':
                    frame, alert, cnt = drowsiness_det.detect(frame.copy())
                    if alert: drowsy_cnt_total += 1
                    frame_d_placeholder.image(frame, channels="BGR", caption="Front")
                else:
                    label, _ = distraction_clf.predict(frame.copy())
                    dist_events[label] += 1
                    cv2.putText(frame, label, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                    frame_c_placeholder.image(frame, channels="BGR", caption="Side")

                # alerts (same logic as live)
                alert_html = ""
                if alert:
                    alert_html += "<h2 style='color:red;'>DROWSY</h2>"
                if label == 'turning_back':
                    alert_html += "<h2 style='color:orange;'>LOOKING BACK</h2>"
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

            time.sleep(0.03)   # ~30 FPS

        # ---- final report ----
        report = f"""
        ### Analysis Summary
        - **Drowsiness events**: {drowsy_cnt_total}
        - **Distraction breakdown**: {dict(dist_events)}
        """
        out_ph.markdown(report)

        # cleanup
        for tmp in (tmp_d, tmp_c):
            if tmp: os.unlink(tmp.name)
        for cap in caps.values():
            cap.release()

st.success("Dashboard ready – use Live or Upload tabs")
