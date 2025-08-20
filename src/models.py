# src/models.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape=(64, 64, 3), num_classes=29):
    """
    Creates and returns a CNN model for ASL sign classification.

    Parameters:
    - input_shape: Tuple of input image dimensions (height, width, channels)
    - num_classes: Number of output classes (letters in ASL)

    Returns:
    - model: Compiled Keras Sequential model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Example usage:
# from models import create_cnn_model
# model = create_cnn_model(input_shape=(64,64,3), num_classes=29)
