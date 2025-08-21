import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import json
from PIL import Image
import cv2

# Load model and class mapping
model = load_model("models/sign_model.h5")
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

st.title("Sign Language Prediction App")

# Radio for input mode
mode = st.radio("Select Input Mode:", ("Webcam", "Upload Single Image", "Upload Multiple Images"))

def preprocess_image(image, target_size=(64,64)):
    frame = np.array(image)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

# ---------------- Webcam Mode ----------------
if mode == "Webcam":
    uploaded_file = st.camera_input("Take a picture")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Captured Image", use_column_width=True)

        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = idx_to_class[predicted_class]
        st.write(f"Predicted Class: {predicted_label}")

# ---------------- Single Image Upload ----------------
elif mode == "Upload Single Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = idx_to_class[predicted_class]
        st.write(f"Predicted Class: {predicted_label}")

# ---------------- Multiple Image Upload ----------------
else:
    uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image {i+1}", use_column_width=True)

            input_image = preprocess_image(image)
            prediction = model.predict(input_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = idx_to_class[predicted_class]
            st.write(f"Predicted Class for Image {i+1}: {predicted_label}")
