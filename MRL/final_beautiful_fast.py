#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BALANCED EYE DETECTION: Fast + Accurate
- 70–100 FPS
- High accuracy
- Smart frame skipping
- Adaptive scaling
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
import argparse

# ==================== CONFIG ====================
MODEL_PATH = Path('improved_cnn_best.keras')
USE_TFLITE = True
INPUT_SIZE = (48, 48)          # متوازن (مش صغير أوي)
CLOSED_THRESHOLD = 5
BASE_SCALE = 0.7              # 70% من الحجم الأصلي
MIN_FACE_SIZE = 80            # تجاهل الوجوه الصغيرة
FRAME_SKIP_MAX = 2            # أقصى تخطي فريم
# ===============================================

# Load TFLite
if USE_TFLITE and Path('model.tflite').exists():
    interpreter = tf.lite.Interpreter(model_path='model.tflite', num_threads=4)
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    print("TFLite model loaded (balanced)")
    def predict_batch(eyes):
        if len(eyes) == 0: return np.array([])
        interpreter.set_tensor(input_idx, eyes.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_idx).flatten()
else:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    def predict_batch(eyes):
        if len(eyes) == 0: return np.array([])
        return model.predict(eyes, verbose=0).flatten()

# DNN Face Detector
MODEL_DIR = Path("models")
proto = MODEL_DIR / "deploy.prototxt"
weights = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

net = None
use_dnn = proto.exists() and weights.exists()
if use_dnn:
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    print("DNN Face Detector: ON")
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Haar Cascade: ON")

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    eye = eye.astype(np.float32) / 255.0
    return np.expand_dims(eye, axis=-1)

def analyze_video(video_path, output_path=None, duration=0):
    cap = cv2.VideoCapture(str(video_path) if video_path != 0 else 0)
    if not cap.isOpened():
        print("Cannot open video")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if output_path:
        w = int(cap.get(3) * BASE_SCALE)
        h = int(cap.get(4) * BASE_SCALE)
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

    closed_counter = 0
    frame_count = 0
    skip_counter = 0
    last_eyes_state = None
    start_time = time.time()
    fps = 0

    print("\n" + "="*60)
    print("BALANCED DETECTION: 70–100 FPS + HIGH ACCURACY")
    print("="*60)

    while True:
        ret, frame = cap.read()
        if not ret: break

        elapsed = time.time() - start_time
        if duration > 0 and elapsed > duration: break

        frame_count += 1
        if frame_count % 20 == 0:
            fps = 20 / (time.time() - start_time + 1e-6)
            start_time = time.time()

        # === Adaptive Frame Skipping ===
        # لو العيون مفتوحة من زمان → نسمح بتخطي فريم
        if last_eyes_state == 'open' and closed_counter == 0:
            skip_counter += 1
            if skip_counter <= FRAME_SKIP_MAX:
                if out:
                    small = cv2.resize(frame, (0,0), fx=BASE_SCALE, fy=BASE_SCALE)
                    out.write(small)
                continue
        skip_counter = 0

        # === Downscale Smart ===
        h_orig, w_orig = frame.shape[:2]
        scale = BASE_SCALE
        if min(w_orig, h_orig) < 400:
            scale = 1.0  # لو الفيديو صغير، ما نصغرش
        small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        eyes_batch = []
        eye_boxes = []
        eyes_closed = False
        eyes_detected = False

        # === Face Detection ===
        faces = []
        if use_dnn and net:
            h, w = small_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf < 0.5: continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                fw, fh = x2-x1, y2-y1
                if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE: continue
                faces.append((x1, y1, fw, fh))
        else:
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

        # === Eye Detection (دقيق) ===
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+int(h*0.65), x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(20, 20),
                maxSize=(80, 80)
            )
            for (ex, ey, ew, eh) in eyes:
                if ey > roi_gray.shape[0] * 0.55: continue
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                if eye_img.size == 0 or min(ew, eh) < 18: continue
                eyes_detected = True
                eyes_batch.append(preprocess_eye(eye_img))
                # تحويل الإحداثيات للحجم الأصلي (لو عايزة ترسمي على الأصلي)
                sx, sy = w_orig / small_frame.shape[1], h_orig / small_frame.shape[0]
                ex_full = int((x + ex) * sx)
                ey_full = int((y + ey) * sy)
                ew_full = int(ew * sx)
                eh_full = int(eh * sy)
                eye_boxes.append((ex_full, ey_full, ew_full, eh_full))

        # === Prediction ===
        preds = np.array([])
        if eyes_batch:
            batch = np.array(eyes_batch)
            preds = predict_batch(batch)

        # === Results ===
        display_frame = frame.copy()
        current_state = 'unknown'
        for i, (pred, (ex, ey, ew, eh)) in enumerate(zip(preds, eye_boxes)):
            is_open = pred > 0.5
            conf = pred if is_open else 1 - pred
            color = (0, 255, 0) if is_open else (0, 0, 255)
            label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
            cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(display_frame, label, (ex, ey-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if not is_open:
                eyes_closed = True
                current_state = 'closed'
            else:
                current_state = 'open'

        if not eyes_detected and faces:
            cv2.putText(display_frame, "Eyes not visible", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # === Drowsiness Logic ===
        if eyes_closed:
            closed_counter += 1
        elif eyes_detected:
            closed_counter = max(0, closed_counter - 1)

        if closed_counter >= CLOSED_THRESHOLD:
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 100), (0, 0, 255), -1)
            cv2.putText(display_frame, "DROWSINESS ALERT!", (60, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
            cv2.putText(display_frame, "WAKE UP!", (60, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        last_eyes_state = current_state

        # === Info ===
        status = "CLOSED" if eyes_closed else "OPEN"
        color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        cv2.putText(display_frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if closed_counter > 0:
            cv2.putText(display_frame, f"Closed: {closed_counter}", (10, display_frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Balanced Detection (70–100 FPS)', display_frame)
        if out:
            out.write(display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'):
            cv2.imwrite(f"screenshot_{int(time.time())}.jpg", display_frame)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print(f"Finished. Avg FPS: {fps:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=0)
    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--duration", type=int, default=0)
    args = parser.parse_args()
    analyze_video(args.video, args.output, args.duration)