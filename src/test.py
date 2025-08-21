# src/test.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helper import load_trained_model
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Paths
# -----------------------------
MODEL_PATH = "models/sign_model.h5"  
TEST_DIR = "data/asl_alphabet_test"  # Path to your test dataset

# -----------------------------
# 2. Load trained model
# -----------------------------
print("[INFO] Loading model...")
model = load_trained_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# -----------------------------
# 3. Prepare test data
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# 4. Evaluate model on test set
# -----------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"[RESULT] Test Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 5. Detailed classification report
# -----------------------------
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Optional: Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

