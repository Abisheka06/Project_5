import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Solar Panel Defect Detection", layout="centered")

st.sidebar.title("🔍 About the App")
st.sidebar.markdown("""
This application uses a trained **CNN model** to detect defects in solar panel images.

📁 **Supported Classes**:
- Clean
- Dusty
- Bird-Drop
- Electrical-Damage
- Physical-Damage
- Snow-Covered

📷 Upload a solar panel image to get a prediction.
""")

st.title("☀️ Solar Panel Defect Detection")
st.write("Upload an image of a solar panel, and the AI will classify the type of defect (if any).")

st.caption(f"Current directory: `{os.getcwd()}`")
if not os.path.exists('solar_panel_model.h5'):
    st.error("❌ Model file not found! Please ensure `solar_panel_model.h5` is in the app directory.")
else:
    st.success("✅ Model loaded successfully.")
    model = load_model('solar_panel_model.h5')

   
    class_names = ['Clean', 'Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

   
    uploaded_file = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

       
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

       
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

       
        st.markdown("---")
        st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
