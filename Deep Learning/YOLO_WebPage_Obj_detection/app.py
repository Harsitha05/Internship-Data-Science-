import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VisionX",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main Background */

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.95);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Titles */

h1, h2, h3 {
    color: white;
    font-weight: 700;
}

/* Cards */

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(10px);
}

/* Buttons */

.stButton>button {
    background: linear-gradient(to right, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
}

/* Upload Box */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Select Box */

.stSelectbox label,
.stMultiSelect label {
    color: white !important;
    font-size: 17px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = YOLO("yolov8n.pt")

# ---------------- SIDEBAR ----------------

st.sidebar.title("VisionX")

page = st.sidebar.radio(
    "",
    [
        "Overview",
        "Image Analysis",
        "Video Analysis",
        "Live Monitoring"
    ]
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.title("Intelligent Visual Analytics")

    st.markdown("### Advanced Real-Time Monitoring & Recognition")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h2>Real-Time</h2>
        <p>Instant Processing</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h2>Multi-Object</h2>
        <p>Simultaneous Recognition</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
        <h2>Smart Insights</h2>
        <p>Visual Analytics</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.image(
        "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5",
        use_container_width=True
    )

# ---------------- IMAGE ANALYSIS ----------------

elif page == "Image Analysis":

    st.title("Image Analysis")

    selected_objects = st.multiselect(
        "Recognition Filters",
        [
            "person",
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "cell phone",
            "bottle",
            "chair",
            "dog",
            "cat"
        ]
    )

    uploaded_image = st.file_uploader(
        "Upload Visual",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        img_array = np.array(image)

        results = model.predict(img_array)

        result = results[0]

        names = result.names

        filtered_objects = []

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = names[cls]

            if selected_objects:

                if label not in selected_objects:
                    continue

            filtered_objects.append({
                "Object": label,
                "Confidence": round(conf * 100, 2)
            })

        annotated_frame = result.plot()

        st.image(
            annotated_frame,
            use_container_width=True
        )

        if filtered_objects:

            df = pd.DataFrame(filtered_objects)

            st.markdown("### Detection Summary")

            st.dataframe(df)

            st.success(f"{len(df)} Objects Identified")

            st.markdown("### Visual Insights")

            counts = df["Object"].value_counts()

            fig, ax = plt.subplots(figsize=(8, 4))

            counts.plot(kind="bar", ax=ax)

            ax.set_xlabel("Objects")
            ax.set_ylabel("Count")

            st.pyplot(fig)

        else:

            st.warning("No matching objects identified.")

# ---------------- VIDEO ANALYSIS ----------------

elif page == "Video Analysis":

    st.title("Video Analysis")

    selected_objects = st.multiselect(
        "Recognition Filters",
        [
            "person",
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "cell phone",
            "bottle"
        ]
    )

    uploaded_video = st.file_uploader(
        "Upload Recording",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(delete=False)

        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        stframe = st.empty()

        object_count = {}

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(frame)

            result = results[0]

            names = result.names

            for box in result.boxes:

                cls = int(box.cls[0])

                label = names[cls]

                if selected_objects:

                    if label not in selected_objects:
                        continue

                object_count[label] = object_count.get(label, 0) + 1

            annotated_frame = result.plot()

            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        if object_count:

            df = pd.DataFrame(
                list(object_count.items()),
                columns=["Object", "Count"]
            )

            st.markdown("### Session Summary")

            st.dataframe(df)

# ---------------- LIVE MONITORING ----------------

elif page == "Live Monitoring":

    st.title("Live Monitoring")

    selected_objects = st.multiselect(
        "Recognition Filters",
        [
            "person",
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "cell phone",
            "bottle"
        ]
    )

    run = st.checkbox("Enable Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera unavailable")
            break

        results = model.predict(frame)

        result = results[0]

        names = result.names

        for box in result.boxes:

            cls = int(box.cls[0])

            label = names[cls]

            if selected_objects:

                if label not in selected_objects:
                    continue

        annotated_frame = result.plot()

        FRAME_WINDOW.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )

    camera.release()