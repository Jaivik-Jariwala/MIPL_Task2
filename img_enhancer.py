import cv2
import numpy as np
from PIL import Image

def enhance_image(image_path):
    """
    Enhance the input image to improve facial feature detection.

    Parameters:
    image_path (str): Path to the image to enhance.

    Returns:
    enhanced_image (numpy array): The enhanced image.
    """
    # Load the image using PIL
    image = Image.open(image_path)

    # Convert PIL image to OpenCV format (numpy array)
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Histogram Equalization
    gray = cv2.equalizeHist(gray)

    # Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)

    # Convert back to color (RGB) from grayscale
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    # Apply Gaussian Blur to reduce noise
    enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    # Additional: Sharpen the image to emphasize features
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    return enhanced_image
