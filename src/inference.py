import cv2
import numpy as np
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model("models/sign_model.h5")

# Load class mapping
with open("models/class_indices.json") as f:
    class_indices = json.load(f)

CLASS_NAMES = {v: k for k, v in class_indices.items()}

# Preprocess image
def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Prediction
frame = cv2.imread("data/raw/asl_alphabet_train/D/D4.jpg")
img = preprocess_image(frame)
pred = model.predict(img)
class_idx = np.argmax(pred, axis=1)[0]

print(f"[RESULT] Predicted Sign: {CLASS_NAMES[class_idx]}")
