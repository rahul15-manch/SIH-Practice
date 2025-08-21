# preprocessing.py
import numpy as np
import cv2
from PIL import Image

def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess a PIL Image or numpy array for model prediction.
    
    Args:
        image (PIL.Image.Image or np.ndarray): Input image
        target_size (tuple): Target size (width, height) for resizing
    
    Returns:
        np.ndarray: Preprocessed image with shape (1, target_size[0], target_size[1], 3)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        frame = np.array(image)
    else:
        frame = image

    # If grayscale, convert to RGB
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Resize image
    frame_resized = cv2.resize(frame, target_size)

    # Normalize pixel values to [0,1]
    frame_normalized = frame_resized / 255.0

    # Add batch dimension
    frame_array = np.expand_dims(frame_normalized, axis=0)

    return frame_array


def preprocess_frame(frame, target_size=(64, 64)):
    """
    Preprocess a cv2 frame (BGR) from webcam or OpenCV for model prediction.
    
    Args:
        frame (np.ndarray): Input frame in BGR format
        target_size (tuple): Target size (width, height)
    
    Returns:
        np.ndarray: Preprocessed image with shape (1, target_size[0], target_size[1], 3)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    frame_resized = cv2.resize(frame_rgb, target_size)

    # Normalize
    frame_normalized = frame_resized / 255.0

    # Add batch dimension
    frame_array = np.expand_dims(frame_normalized, axis=0)

    return frame_array
