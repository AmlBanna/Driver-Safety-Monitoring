#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from pathlib import Path
import urllib.request
import zipfile
import json
from collections import Counter, deque
import threading
import queue
import time

# -------------------------- CONFIG --------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Google Drive IDs
EYE_TFLITE_ID = "1m-6tjfX46a82wxmrMTclBXrVAf_XmJlj"          # eye_model.tflite (حوّلي الموديل لـ .tflite)
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
    if path.exists(): return
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
        st.success("Drowsiness (TFLite) Loaded")
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
                open_eye = scores[i] > 0.5
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

# -------------------------- UI (نفس اللي فات) --------------------------
st.set_page_config(page_title="Driver Safety", layout="wide")
st.title("Driver Safety Dashboard")
tab1, tab2 = st.tabs(["Live", "Upload"])

# باقي الكود (Live + Upload) نفس اللي في الرسالة السابقة
# (انسخيه من السطر "with tab1:" لحد النهاية)
