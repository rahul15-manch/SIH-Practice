# src/train.py
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import create_cnn_model  # import the modular model

# ---------------------------
# Paths
# ---------------------------
DATA_DIR = "data/raw/asl_alphabet_train/"  # path to your unzipped dataset
MODEL_PATH = "models/sign_model.h5"

# ---------------------------
# Hyperparameters
# ---------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# ---------------------------
# Data Generators
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ---------------------------
# Model
# ---------------------------
model = create_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                         num_classes=train_generator.num_classes)

# ---------------------------
# Train
# ---------------------------
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ---------------------------
# Save Model
# ---------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ---------------------------
# Save Class Mapping
# ---------------------------
class_indices = train_generator.class_indices
with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f)
print("Class mapping saved to models/class_indices.json")
