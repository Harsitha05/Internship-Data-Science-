import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/deepfake_model.h5")

# Image settings
IMG_SIZE = 128

# Test image path
image_path = "test.jpg"

# Read image
img = cv2.imread(image_path)

# Check image loaded
if img is None:
    print("Image not found!")
    exit()

# Resize image
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Normalize image
img = img / 255.0

# Reshape image
img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

# Prediction
prediction = model.predict(img)

confidence = prediction[0][0]

if confidence > 0.5:
    print(f"\nFake Image Detected")
    print(f"Confidence: {confidence:.2f}")
else:
    print(f"\nReal Image Detected")
    print(f"Confidence: {1-confidence:.2f}")