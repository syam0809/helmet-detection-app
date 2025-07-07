import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import db  # your db.py module in the same folder

# --- App title ---
st.title("🚦 Helmet Detection System with Database")

# --- Ensure captured_images folder exists ---
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# --- Load YOLO model ---
model = YOLO('yolov8n.pt')  # Replace with your own trained model if needed

# --- Create DB table on app start ---
db.create_table()

# --- Function to process a single frame ---
def process_frame(frame):
    results = model(frame)

    for result in results:
        boxes = result.boxes
        classes = result.boxes.cls

        for box, cls in zip(boxes.xyxy, classes):
            cls = int(cls)

            if cls == 0:  # Helmet class
                continue
            elif cls == 2:  # Number plate or No Helmet class (adjust based on your model)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"captured_images/violation_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                plate_text = "Unknown"  # You can integrate OCR later
                db.add_fine(plate_text, "No Helmet", timestamp)

    return frame

# --- Sidebar inputs ---
st.sidebar.header("📷 Input Options")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
rtsp_url = st.sidebar.text_input(
    "RTSP CCTV Stream URL",
    placeholder="rtsp://user:pass@IP:port/Streaming/Channels/101"
)

# --- Process image ---
if image_file is not None:
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    processed_frame = process_frame(frame)
    st.image(processed_frame, channels="BGR", caption="Processed Image")

# --- Process uploaded video ---
elif video_file is not None:
    file_path = os.path.join("captured_images", video_file.name)
    with open(file_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(file_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        stframe.image(processed_frame, channels="BGR", use_container_width=True)
    cap.release()

# --- Process RTSP CCTV Stream ---
elif rtsp_url:
    st.info(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to read frame from stream. Check your RTSP URL.")
            break
        processed_frame = process_frame(frame)
        stframe.image(processed_frame, channels="BGR", use_container_width=True)
    cap.release()

# --- Violation dashboard ---
st.header("📊 Violation Dashboard (Permanent)")

fines = db.get_fines()

if fines:
    df = pd.DataFrame(fines, columns=["Plate", "Violation", "Timestamp"])
    st.dataframe(df)

    st.download_button(
        label="⬇️ Download Violations as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="violations.csv",
        mime="text/csv"
    )
else:
    st.info("✅ No violations detected yet.")

