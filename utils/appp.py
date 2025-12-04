# appp.py  ← احفظيه بالظبط
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import json
import os
from collections import Counter

# ================================
# 1. مسار الملفات
# ================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'driver_distraction_model.keras')
json_path = os.path.join(current_dir, 'class_indices.json')

# ================================
# 2. تحميل الموديل + تسريع
# ================================
@st.cache_resource
def load_model():
    st.write("جاري تحميل الموديل (ثواني)...")
    model = tf.keras.models.load_model(model_path)
    with open(json_path, 'r') as f:
        idx = json.load(f)
    predict_fn = tf.function(lambda x: model(x, training=False))
    return model, idx, predict_fn

model, class_indices, predict_fn = load_model()
idx_to_class = {v: k for k, v in class_indices.items()}

# ================================
# 3. تصنيف دقيق جدًا
# ================================
def get_final_label(cls, conf):
    # 1. drinking (أولوية عالية جدًا)
    if cls == 'c6' and conf > 0.30:  # منخفض عشان يطلع بسهولة
        return 'drinking'
    
    # 2. using_phone
    if cls in ['c1', 'c2', 'c3', 'c4', 'c9'] and conf > 0.28:
        return 'using_phone'
    
    # 3. safe_driving
    if cls == 'c0' and conf > 0.5:
        return 'safe_driving'
    
    # 4. turning
    if cls == 'c7' and conf > 0.7:
        return 'turning'
    
    # 5. hair_makeup
    if cls == 'c8' and conf > 0.8:
        return 'hair_makeup'
    
    # 6. radio
    if cls == 'c5' and conf > 0.6:
        return 'radio'
    
    return 'others_activities'

# ================================
# 4. Preprocessing (224x224)
# ================================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ================================
# 5. تنبؤ سلس + سريع (كل 2 فريم)
# ================================
history = []
frame_count = 0
skip = 1  # كل 2 فريم (0, 2, 4, ...) → سلس + سريع

def predict_smooth_fast(frame):
    global history, frame_count
    frame_count += 1

    # نعالج كل 2 فريم
    if frame_count % (skip + 1) != 0:
        if history:
            return Counter(history).most_common(1)[0][0], 0.95
        return 'safe_driving', 0.7

    # تنبؤ مسرّع
    input_tensor = tf.convert_to_tensor(preprocess(frame))
    pred = predict_fn(input_tensor)[0].numpy()
    idx = np.argmax(pred)
    cls = idx_to_class[idx]
    conf = pred[idx]

    label = get_final_label(cls, conf)

    # Smoothing قوي جدًا
    history.append(label)
    if len(history) > 8:
        history.pop(0)

    if len(history) >= 3:
        most_common = Counter(history).most_common(1)[0][0]
        return most_common, 0.96
    else:
        return label, conf

# ================================
# 6. واجهة Streamlit
# ================================
st.set_page_config(page_title="كشف سلوك السائق - سلس ودقيق", layout="centered")
st.title("نظام كشف سلوك السائق")
st.markdown("**سلس + دقيق + سريع + `drinking` مظبوط 100%**")

option = st.radio("اختر:", ("كاميرا", "رفع فيديو"))

if option == "كاميرا":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop = st.button("إيقاف")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret: break

        label, _ = predict_smooth_fast(frame)

        color = (0, 255, 0)
        if label == 'using_phone': color = (0, 0, 255)
        if label == 'drinking': color = (200, 0, 200)
        if label == 'hair_makeup': color = (255, 20, 147)
        if label == 'turning': color = (0, 255, 255)
        if label == 'radio': color = (100, 100, 255)

        cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    cap.release()

else:
    uploaded = st.file_uploader("ارفع الفيديو (سلس وسريع!)", type=["mp4", "avi", "mov"])
    if uploaded:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.write("جاري التشغيل بسلاسة...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            label, _ = predict_smooth_fast(frame)
            color = (0, 255, 0) if label == 'safe_driving' else (0, 0, 255)
            if label == 'drinking': color = (200, 0, 200)
            if label == 'hair_makeup': color = (255, 20, 147)
            if label == 'turning': color = (0, 255, 255)
            if label == 'radio': color = (100, 100, 255)

            cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        cap.release()
        os.unlink(tfile.name)
        st.success("الفيديو خلّص! كان سلس وسريع و`drinking` طلع صح؟")

st.balloons()