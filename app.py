import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD LOCAL BACKGROUND IMAGE ----------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_img = get_base64("Brain_Img.png")

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>

/* Background Image */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 0;
}}

/* Glass container */
.main, .block-container {{
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    position: relative;
    z-index: 1;
}}

/* Title */
.title {{
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}}

/* Subtitle */
.subtitle {{
    text-align: center;
    font-size: 18px;
    color: #f1f1f1;
    margin-bottom: 25px;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="title">🧠 AI Brain Tumor Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an MRI image and get instant AI prediction</p>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    file_id = "1H-Aal7xMKoDw7r_Qx12NDVc28xWpGhen"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "Tumour_Detection.h5"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    model = tf.keras.models.load_model(output)
    return model

model = load_model()
IMG_SIZE = 150

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded MRI", use_column_width=True)

    with col2:
        st.write("### 🔍 AI Analysis in Progress")

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # ---------- PREPROCESS ----------
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- PREDICTION ----------
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        st.write("### 🧾 Prediction Result")

        if confidence > 0.5:
            st.error(f"⚠️ Tumor Detected\n\nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f"✅ No Tumor Detected\n\nConfidence: {(1-confidence)*100:.2f}%")
            st.balloons()

        # ---------- CONFIDENCE BAR ----------
        st.write("### 📊 Model Confidence")
        st.progress(confidence if confidence > 0.5 else 1-confidence)

        st.info(f"Raw Model Output: {confidence:.4f}")

st.write("---")
st.caption("Deep Learning Medical Imaging System | Developed by Shivkumar Nevhal")

