import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==== الإعداد الأساسي ====
st.set_page_config(page_title="🧠 Custom Smart Classifier", layout="centered")

# تحميل النموذج الجديد
model_path = "models/mobilenet_finetuned.h5"
model = tf.keras.models.load_model(model_path)

# قراءة أسماء الفئات من JSON
labels_path = "models/class_names.json"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        class_names = json.load(f)
else:
    st.error("⚠️ Could not load class names.")
    class_names = [f"class_{i}" for i in range(model.output_shape[-1])]

# سجل التاريخ
if "history" not in st.session_state:
    st.session_state.history = []

# ==== تنسيق CSS ====
st.markdown("""
<style>
body { background-color: #0f1117; color: #ffffff; }
.stApp { font-family: 'Segoe UI', sans-serif; }
.title { font-size: 32px; color: #fbbf24; text-align:center; margin-bottom: 10px; }
.subtitle { font-size: 18px; color: #a3e635; text-align:center; margin-bottom: 30px; }
.card {
    background-color: #1f2937;
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 12px;
}
.stDownloadButton>button {
    background-color: #10b981;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1.2em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==== العنوان ====
st.markdown('<div class="title">🧠 Custom Smart Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or paste a URL — get an instant prediction!</div>', unsafe_allow_html=True)

# ==== Tabs: رفع صورة أو من رابط ====
tab1, tab2 = st.tabs(["📁 Upload Image", "🌐 From URL"])
image = None

with tab1:
    uploaded_file = st.file_uploader("Upload a real-world image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

with tab2:
    url = st.text_input("Paste image URL:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("❌ Could not load image from URL.")

# ==== إذا وُجدت صورة، نفّذ التنبؤ ====
if image:
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_preprocessed = preprocess_input(image_array)
    input_array = np.expand_dims(image_preprocessed, axis=0)

    # التنبؤ
    predictions = model.predict(input_array)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    # عرض النتيجة
    st.markdown(f"### ✅ Prediction: **{predicted_class}**")
    st.progress(int(confidence * 100))

    # رسم بياني لجميع الفئات
    st.markdown("### 📊 All Class Probabilities")
    fig, ax = plt.subplots(figsize=(6, len(class_names) * 0.35))
    ax.barh(class_names, predictions, color=plt.cm.viridis(predictions))
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    ax.set_xlabel("Confidence")
    st.pyplot(fig)

    # Top-3 نتائج
    st.markdown("### 🥇 Top 3 Predictions")
    top_indices = predictions.argsort()[-3:][::-1]
    for i in top_indices:
        st.markdown(f"🔹 **{class_names[i]}** — {predictions[i]*100:.2f}%")

    # تنزيل النتائج كـ JSON
    result_dict = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": {
            class_names[i]: float(prob) for i, prob in enumerate(predictions)
        }
    }
    result_json = json.dumps(result_dict, indent=2)
    st.download_button("⬇️ Download Result as JSON", result_json, file_name="result.json", mime="application/json")

    # حفظ في التاريخ
    st.session_state.history.append({
        "image": image.copy(),
        "prediction": predicted_class,
        "confidence": confidence
    })
else:
    st.info("Upload or paste an image to start.")

# ==== سجل التنبؤات ====
if st.session_state.history:
    st.markdown("## 🕓 Prediction History")
    for item in reversed(st.session_state.history[-5:]):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(item["image"].resize((64, 64)), use_container_width=True)
        with cols[1]:
            st.markdown(f"**Prediction:** {item['prediction']}")
            st.markdown(f"Confidence: {item['confidence']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
