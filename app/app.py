import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
from src.predict import predict_vehicle

st.title("Emergency Vehicle Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image")

    label, probability = predict_vehicle(img)

    st.success(f"Prediction: {label}")

    st.write(f"Confidence: {probability:.2%}")
