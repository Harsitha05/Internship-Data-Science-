import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b, #312e81);
        background-size: cover;
        background-attachment: fixed;
    }

    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 60px;
        font-weight: 800;
        letter-spacing: 2px;
        font-family: 'Trebuchet MS', sans-serif;
        text-shadow: 2px 2px 12px rgba(0,0,0,0.4);
    }

    .subtext {
        text-align: center;
        color: #e2e8f0;
        font-size: 22px;
        margin-bottom: 10px;
        font-family: Georgia, serif;
    }

    .quote {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 35px;
        font-style: italic;
    }

    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border-radius: 12px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1>🤖 AI Object Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtext'>Upload an image and detect your favorite objects instantly using powerful AI technology</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='quote'>✨ Smart Detection • Beautiful Interface • Real-Time AI Experience ✨</div>",
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- OBJECT OPTIONS ----------------
object_options = [
    "all",
    "person",
    "bottle",
    "chair",
    "laptop",
    "cell phone",
    "book",
    "cup",
    "tv",
    "mouse",
    "keyboard"
]

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Detection Settings")

selected_object = st.sidebar.selectbox(
    "Choose object to detect",
    object_options
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Convert RGBA to RGB
    image = image.convert("RGB")

    image_np = np.array(image)

    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🚀 Start Detection"):

        with st.spinner("Detecting objects..."):

            results = model(image_np)

            detected_image = image_np.copy()

            object_count = 0

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    cls = int(box.cls[0])

                    class_name = model.names[cls]

                    confidence = float(box.conf[0])

                    # Detect selected object
                    if selected_object == "all" or class_name == selected_object:

                        object_count += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw rectangle
                        cv2.rectangle(
                            detected_image,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                        # Label
                        label = f"{class_name} {confidence:.2f}"

                        cv2.putText(
                            detected_image,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

            # ---------------- SHOW COUNT ----------------
            st.markdown("## 🔍 Detection Summary")

            if selected_object == "all":
                st.success(f"Total Objects Detected: {object_count}")
            else:
                st.success(f"Total '{selected_object}' Detected: {object_count}")

            # ---------------- SHOW RESULT ----------------
            st.subheader("✅ Detection Result")

            st.image(detected_image, use_container_width=True)