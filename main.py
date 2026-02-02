# ---------------- STREAMLIT LIVE WEBCAM ---------------- #

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = "artifacts/model/face_mask_detector.h5"

# Use custom_objects to handle 'mse' deserialization
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"mse": tf.keras.losses.MeanSquaredError}
)

CLASS_MAP = {0: "With Mask 😷", 1: "Without Mask 😡", 2: "Mask Worn Incorrectly 🤔"}

IMG_SIZE = 224

# ---------------- UTILITY FUNCTIONS ---------------- #
def preprocess_frame(frame):
    """Resize and normalize frame for the model."""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_bbox(frame, bbox, label, color=(0, 255, 0)):
    """Draw bounding box and label on frame."""
    h, w, _ = frame.shape
    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(
        frame,
        label,
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return frame

# ---------------- STREAMLIT APP ---------------- #
st.title("🟢 Live Face Mask Detection")

# Checkbox to start/stop webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

# Open webcam
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("❌ Failed to grab frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess and predict
    input_frame = preprocess_frame(rgb_frame)
    class_pred, bbox_pred = model.predict(input_frame)
    class_id = np.argmax(class_pred[0])
    bbox = bbox_pred[0]

    # Draw bounding box and label
    frame = draw_bbox(rgb_frame, bbox, CLASS_MAP[class_id])

    # Show frame in Streamlit
    FRAME_WINDOW.image(frame)

# Release webcam when stopped
cap.release()
st.write("🛑 Webcam stopped")
