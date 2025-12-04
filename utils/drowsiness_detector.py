import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

class DrowsinessDetector:
    def __init__(self, model_path='models/improved_cnn_best.keras'):
        self.INPUT_SIZE = (48, 48)
        self.CLOSED_THRESHOLD = 5
        self.MIN_FACE_SIZE = 80
        self.closed_counter = 0
        self.last_state = None
        
        # Load model
        self.model = None
        self.use_tflite = False
        
        try:
            if Path('models/model.tflite').exists():
                self.interpreter = tf.lite.Interpreter(
                    model_path='models/model.tflite', 
                    num_threads=4
                )
                self.interpreter.allocate_tensors()
                self.input_idx = self.interpreter.get_input_details()[0]['index']
                self.output_idx = self.interpreter.get_output_details()[0]['index']
                self.use_tflite = True
                print("✅ TFLite drowsiness model loaded")
            elif Path(model_path).exists():
                self.model = tf.keras.models.load_model(model_path)
                self.use_tflite = False
                print("✅ Keras drowsiness model loaded")
            else:
                print(f"⚠️ Drowsiness model not found at: {model_path}")
                print("⚠️ Drowsiness detection will be disabled")
        except Exception as e:
            print(f"⚠️ Could not load drowsiness model: {e}")
            print("⚠️ Drowsiness detection will be disabled")
        
        # Face and eye cascades
        proto = Path("models/deploy.prototxt")
        weights = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
        
        if proto.exists() and weights.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
            self.use_dnn = True
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_dnn = False
        
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def preprocess_eye(self, eye_img):
        eye = cv2.resize(eye_img, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        eye = eye.astype(np.float32) / 255.0
        return np.expand_dims(eye, axis=-1)
    
    def predict_batch(self, eyes):
        if len(eyes) == 0 or self.model is None:
            return np.array([])
        
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_idx, eyes.astype(np.float32))
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_idx).flatten()
        else:
            return self.model.predict(eyes, verbose=0).flatten()
    
    def detect(self, frame):
        """
        Returns: (status, severity, annotated_frame)
        status: 'safe', 'drowsy', 'alert'
        severity: 0-10 (0=safe, 10=critical)
        """
        if self.model is None:
            return 'unknown', 0, frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        # Detect faces
        faces = []
        if self.use_dnn and hasattr(self, 'net'):
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf < 0.5: continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                fw, fh = x2-x1, y2-y1
                if fw < self.MIN_FACE_SIZE or fh < self.MIN_FACE_SIZE: continue
                faces.append((x1, y1, fw, fh))
        else:
            faces = self.face_cascade.detectMultiScale(
                gray, 1.1, 4, minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE)
            )
        
        # Detect eyes
        eyes_batch = []
        eye_boxes = []
        eyes_closed = False
        eyes_detected = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+int(h*0.65), x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
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
                eyes_batch.append(self.preprocess_eye(eye_img))
                eye_boxes.append((x+ex, y+ey, ew, eh))
        
        # Predict
        if eyes_batch:
            batch = np.array(eyes_batch)
            preds = self.predict_batch(batch)
            
            for i, (pred, (ex, ey, ew, eh)) in enumerate(zip(preds, eye_boxes)):
                is_open = pred > 0.5
                conf = pred if is_open else 1 - pred
                color = (0, 255, 0) if is_open else (0, 0, 255)
                label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
                cv2.rectangle(display_frame, (ex, ey), (ex+ew, ey+eh), color, 2)
                cv2.putText(display_frame, label, (ex, ey-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if not is_open:
                    eyes_closed = True
        
        # Update counter
        if eyes_closed:
            self.closed_counter += 1
        elif eyes_detected:
            self.closed_counter = max(0, self.closed_counter - 1)
        
        # Determine status and severity
        status = 'safe'
        severity = 0
        
        if self.closed_counter >= self.CLOSED_THRESHOLD:
            status = 'alert'
            severity = 10
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
            cv2.putText(display_frame, "DROWSINESS ALERT!", (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        elif eyes_closed and self.closed_counter > 0:
            status = 'drowsy'
            severity = min(9, self.closed_counter * 2)
        
        # Status text
        color = (0, 255, 0) if status == 'safe' else (0, 165, 255) if status == 'drowsy' else (0, 0, 255)
        cv2.putText(display_frame, f"Eyes: {'CLOSED' if eyes_closed else 'OPEN'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.closed_counter > 0:
            cv2.putText(display_frame, f"Closed Frames: {self.closed_counter}", 
                       (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        return status, severity, display_frame
    
    def reset(self):
        """Reset detection state"""
        self.closed_counter = 0
        self.last_state = None
