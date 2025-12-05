#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from pathlib import Path
import urllib.request
import json
from collections import Counter, deque
import threading
import queue
import time

# -------------------------- CONFIG --------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive IDs (أنا حوّلت النعاس لـ TFLite وحطيته)
EYE_TFLITE_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"           # eye_model.tflite (محوّل مني)
DISTRACTION_ID = "1QE5Z84JU4b0N0MlZtaLsdFt60nIXzt3Z"        # driver_distraction_model.keras
CLASS_JSON_ID = "1zDv7V4iQri7lC-e8BLsUJPLMuLlA8Ylg"         # class_indices.json

TFLITE_URL = f"https://drive.google.com/uc?id={EYE_TFLITE_ID}&export=download"
KERAS_URL = f"https://drive.google.com/uc?id={DISTRACTION_ID}&export=download"
JSON_URL = f"https://drive.google.com/uc?id={CLASS_JSON_ID}&export=download"

TFLITE_PATH = MODELS_DIR / "eye_model.tflite"
KERAS_PATH = MODELS_DIR / "driver_distraction_model.keras"
JSON_PATH = MODELS_DIR / "class_indices.json"

# -------------------------- DOWNLOAD --------------------------
def download(url, path):
    if path.exists():
        return
    with st.spinner(f"Downloading {path.name}..."):
        urllib.request.urlretrieve(url, path)

# -------------------------- DROWSINESS (TFLite) --------------------------
class Drowsiness:
    def __init__(self):
        download(TFLITE_URL, TFLITE_PATH)
        self.interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.output_idx = self.interpreter.get_output_details()[0]['index']
        st.success("Drowsiness Model Loaded (TFLite)")
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.cnt = 0
        self.THRESH = 5

    def detect(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (0,0), fx=0.7, fy=0.7), cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        eyes, boxes = [], []
        h, w = frame.shape[:2]
        for (x,y,fw,fh) in faces:
            roi = gray[y:y+int(fh*0.65), x:x+fw]
            es = self.eye.detectMultiScale(roi, 1.05, 4, minSize=(20,20))
            for (ex,ey,ew,eh) in es:
                if ey > roi.shape[0]*0.55: continue
                eye_img = cv2.resize(roi[ey:ey+eh, ex:ex+ew], (48,48)) / 255.0
                eyes.append(np.expand_dims(eye_img, -1).astype(np.float32))
                sx = w / gray.shape[1]; sy = h / gray.shape[0]
                boxes.append((int((x+ex)*sx), int((y+ey)*sy), int(ew*sx), int(eh*sy)))
        if eyes:
            batch = np.stack(eyes)
            self.interpreter.set_tensor(self.input_idx, batch)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self.output_idx).flatten()
            for i, (x,y,w,h) in enumerate(boxes):
                open_eye = scores0
                col = (0,255,0) if open_eye else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
                cv2.putText(frame, f"{'OPEN' if open_eye else 'CLOSED'} {scores[i]:.2f}", (x,y-5), 0, 0.4, col, 1)
                if not open_eye: self.cnt += 1
                else: self.cnt = max(0, self.cnt-1)
        return frame, self.cnt >= self.THRESH

# -------------------------- DISTRACTION (Keras) --------------------------
class Distraction:
    def __init__(self):
        download(KERAS_URL, KERAS_PATH)
        download(JSON_URL, JSON_PATH)
        self.model = tf.keras.models.load_model(str(KERAS_PATH))
        with open(JSON_PATH) as f: idx = json.load(f)
        self.idx2cls = {v:k for k,v in idx.items()}
        self.hist = deque(maxlen=8)
        self.skip = 0
        st.success("Distraction Model Loaded")

    def predict(self, frame):
        self.skip = (self.skip + 1) % 2
        if self.skip: return self.hist[-1] if self.hist else "safe_driving"
        x = tf.convert_to_tensor(np.expand_dims(cv2.resize(frame, (224,224))/255.0, 0).astype(np.float32))
        p = self.model(x, training=False)[0].numpy()
        cls = self.idx2cls[np.argmax(p)]
        conf = p.max()
        label = (
            'drinking' if cls=='c6' and conf>0.3 else
            'using_phone' if cls in ['c1','c2','c3','c4','c9'] and conf>0.28 else
            'safe_driving' if cls=='c0' and conf>0.5 else
            'turning_back' if cls=='c7' and conf>0.7 else
            'hair_makeup' if cls=='c8' and conf>0.8 else
            'radio' if cls=='c5' and conf>0.6 else 'other'
        )
        self.hist.append(label)
        return Counter(self.hist).most_common(1)[0][0] if len(self.hist)>=3 else label

# -------------------------- LOAD --------------------------
@st.cache_resource
def load():
    return Drowsiness(), Distraction()

drowsy, distract = load()

# -------------------------- UI --------------------------
st.set_page_config(page_title="Driver Safety", layout="wide")
st.title("Driver Safety Dashboard")
tab1, tab2 = st.tabs(["Live", "Upload"])

# LIVE
with tab1:
    col1, col2 = st.columns(2)
    with col1: cam1 = st.checkbox("Front Cam", True)
    with col2: cam2 = st.checkbox("Side Cam", True)
    start = st.button("Start", type="primary")
    stop = st.button("Stop")
    ph1 = st.empty(); ph2 = st.empty(); alert = st.empty()

    q1, q2, res = queue.Queue(2), queue.Queue(2), queue.Queue(2)

    def cam_thread(idx, q):
        cap = cv2.VideoCapture(idx)
        while cap.isOpened():
            r, f = cap.read()
            if not r: break
            try: q.put_nowait(f)
            except: pass
        cap.release()

    def proc():
        while True:
            if not q1.empty():
                f = q1.get()
                f, a = drowsy.detect(f.copy())
                res.put_nowait(('d', f, a))
            if not q2.empty():
                f = q2.get()
                l = distract.predict(f.copy())
                cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 3)
                res.put_nowait(('c', f, l))

    if start:
        if cam1: threading.Thread(target=cam_thread, args=(0, q1), daemon=True).start()
        if cam2: threading.Thread(target=cam_thread, args=(1, q2), daemon=True).start()
        threading.Thread(target=proc, daemon=True).start()
        st.session_state.run = True

    if stop: st.session_state.run = False; st.rerun()

    if st.session_state.get("run"):
        df, cf, da, cl = None, None, False, "safe_driving"
        while not res.empty():
            t, f, d = res.get()
            if t=='d': df, da = f, d
            else: cf, cl = f, d
        if cam1 and df is not None: ph1.image(df, channels="BGR")
        if cam2 and cf is not None: ph2.image(cf, channels="BGR")
        msg = ""
        if da: msg += "<h2 style='color:red'>DROWSY!</h2>"
        if cl == 'turning_back': msg += "<h2 style='color:orange'>LOOKING BACK!</h2>"
        elif cl in ['using_phone','drinking']: msg += f"<h3 style='color:#FF4500'>{cl.upper()}</h3>"
        alert.markdown(msg, unsafe_allow_html=True)

# UPLOAD
with tab2:
    f1 = st.file_uploader("Front", ["mp4","jpg"])
    f2 = st.file_uploader("Side", ["mp4","jpg"])
    if st.button("Analyze") and (f1 or f2):
        t1 = t2 = None
        if f1: t1 = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f1.name).suffix); t1.write(f1.getvalue()); t1.close()
        if f2: t2 = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f2.name).suffix); t2.write(f2.getvalue()); t2.close()
        c1 = cv2.VideoCapture(t1.name) if t1 else None
        c2 = cv2.VideoCapture(t2.name) if t2 else None
        p1 = st.empty(); p2 = st.empty()
        dc = 0; ev = Counter()
        while (c1 and c1.isOpened()) or (c2 and c2.isOpened()):
            if c1:
                r, f = c1.read()
                if r:
                    f, a = drowsy.detect(f.copy())
                    if a: dc += 1
                    p1.image(f, channels="BGR")
            if c2:
                r, f = c2.read()
                if r:
                    l = distract.predict(f.copy())
                    ev[l] += 1
                    cv2.putText(f, l, (10,70), 0, 2, (0,0,255), 3)
                    p2.image(f, channels="BGR")
            time.sleep(0.03)
        st.success(f"Drowsy: {dc} | Events: {dict(ev)}")
        for t in (t1,t2): if t: os.unlink(t.name)
        for c in (c1,c2): if c: c.release()
