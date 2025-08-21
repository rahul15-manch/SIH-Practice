import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# -------------------------------
# Global Config
# -------------------------------
IMG_SIZE = (64, 64)  # Change as per your dataset
CLASS_NAMES = [chr(i) for i in range(65, 91)]  # ['A', 'B', 'C', ..., 'Z']

# -------------------------------
# Image Utilities
# -------------------------------
def load_and_preprocess_img(img_path, target_size=IMG_SIZE):
    """
    Load and preprocess image for model prediction.
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"[ERROR] Could not load {img_path}: {e}")
        return None

def predict_image(model, img_path):
    """
    Predict the class of a given image using the trained model.
    """
    img_array = load_and_preprocess_img(img_path)
    if img_array is None:
        return None

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    return predicted_class

# -------------------------------
# Model Utilities
# -------------------------------
def save_trained_model(model, path="models/sign_lang_model.h5"):
    """
    Save trained model to given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[INFO] Model saved at {path}")

def load_trained_model(path="models/sign_lang_model.h5"):
    """
    Load trained model from path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    model = load_model(path)
    print(f"[INFO] Loaded model from {path}")
    return model

# -------------------------------
# Visualization Utilities
# -------------------------------
def plot_training(history):
    """
    Plot training & validation accuracy/loss.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Acc")
    plt.plot(val_acc, label="Validation Acc")
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.show()
