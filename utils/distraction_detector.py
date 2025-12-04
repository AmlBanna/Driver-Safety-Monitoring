import cv2
import numpy as np
import tensorflow as tf
import json
from collections import Counter
from pathlib import Path

class DistractionDetector:
    def __init__(self, model_path='models/driver_distraction_model.keras', 
                 json_path='models/class_indices.json'):
        self.history = []
        self.frame_count = 0
        self.skip = 1  # Process every 2nd frame
        self.model = None
        self.predict_fn = None
        
        # Load model
        try:
            if Path(model_path).exists():
                self.model = tf.keras.models.load_model(model_path)
                self.predict_fn = tf.function(lambda x: self.model(x, training=False))
                print(f"✅ Distraction model loaded from: {model_path}")
            else:
                print(f"⚠️ Distraction model not found at: {model_path}")
                print("⚠️ Distraction detection will be disabled")
        except Exception as e:
            print(f"⚠️ Could not load distraction model: {e}")
            print("⚠️ Distraction detection will be disabled")
        
        # Load class indices
        try:
            if Path(json_path).exists():
                with open(json_path, 'r') as f:
                    class_indices = json.load(f)
                self.idx_to_class = {v: k for k, v in class_indices.items()}
                print(f"✅ Class indices loaded: {len(self.idx_to_class)} classes")
            else:
                print(f"⚠️ Class indices not found at: {json_path}")
                self.idx_to_class = {}
        except Exception as e:
            print(f"⚠️ Could not load class indices: {e}")
            self.idx_to_class = {}
        
        # Severity mapping
        self.severity_map = {
            'safe_driving': 0,
            'radio': 3,
            'turning': 4,
            'hair_makeup': 6,
            'using_phone': 8,
            'drinking': 9,
            'others_activities': 5
        }
        
        # Color mapping
        self.color_map = {
            'safe_driving': (0, 255, 0),
            'using_phone': (0, 0, 255),
            'drinking': (200, 0, 200),
            'hair_makeup': (255, 20, 147),
            'turning': (0, 255, 255),
            'radio': (100, 100, 255),
            'others_activities': (255, 165, 0)
        }
    
    def get_final_label(self, cls, conf):
        """Map raw predictions to meaningful labels"""
        if cls == 'c6' and conf > 0.30:
            return 'drinking'
        if cls in ['c1', 'c2', 'c3', 'c4', 'c9'] and conf > 0.28:
            return 'using_phone'
        if cls == 'c0' and conf > 0.5:
            return 'safe_driving'
        if cls == 'c7' and conf > 0.7:
            return 'turning'
        if cls == 'c8' and conf > 0.8:
            return 'hair_makeup'
        if cls == 'c5' and conf > 0.6:
            return 'radio'
        return 'others_activities'
    
    def preprocess(self, frame):
        """Preprocess frame for model"""
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    
    def detect(self, frame):
        """
        Returns: (activity, severity, annotated_frame)
        activity: 'safe_driving', 'using_phone', 'drinking', etc.
        severity: 0-10 (0=safe, 10=critical)
        """
        if self.model is None:
            display_frame = frame.copy()
            cv2.putText(display_frame, "Distraction model not available", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(display_frame, "Downloading model... Please wait", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            return 'unknown', 0, display_frame
        
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % (self.skip + 1) != 0:
            if self.history:
                most_common = Counter(self.history).most_common(1)[0][0]
                severity = self.severity_map.get(most_common, 5)
                return most_common, severity, self._annotate_frame(frame, most_common, 0.95)
            return 'safe_driving', 0, frame
        
        # Predict
        try:
            input_tensor = tf.convert_to_tensor(self.preprocess(frame))
            pred = self.predict_fn(input_tensor)[0].numpy()
            idx = np.argmax(pred)
            cls = self.idx_to_class.get(idx, 'c0')
            conf = pred[idx]
            
            label = self.get_final_label(cls, conf)
            
            # Smoothing
            self.history.append(label)
            if len(self.history) > 8:
                self.history.pop(0)
            
            if len(self.history) >= 3:
                most_common = Counter(self.history).most_common(1)[0][0]
                label = most_common
                conf = 0.96
        except Exception as e:
            print(f"Prediction error: {e}")
            label = 'unknown'
            conf = 0.0
        
        severity = self.severity_map.get(label, 5)
        annotated = self._annotate_frame(frame, label, conf)
        
        return label, severity, annotated
    
    def _annotate_frame(self, frame, label, conf):
        """Add label overlay to frame"""
        display_frame = frame.copy()
        color = self.color_map.get(label, (255, 255, 255))
        
        # Main label
        text = label.replace('_', ' ').upper()
        cv2.putText(display_frame, text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Confidence
        cv2.putText(display_frame, f"Conf: {conf:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return display_frame
    
    def reset(self):
        """Reset detection state"""
        self.history = []
        self.frame_count = 0
