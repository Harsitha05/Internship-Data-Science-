import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(
    page_title="DeepFake Vision AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = load_model("model/deepfake_model.h5")

MODEL_SIZE = 224

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

.main-title{
    text-align:center;
    font-size:55px;
    font-weight:800;
    color:white;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.card{
    background:#111827;
    padding:20px;
    border-radius:15px;
    border:1px solid #374151;
}

.footer{
    text-align:center;
    color:#cbd5e1;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    '<div class="main-title">🛡️ DeepFake Vision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Advanced AI-Based DeepFake Image Detection System</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# FEATURES
# -----------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("🤖 AI Detection")

with c2:
    st.info("⚡ Fast Analysis")

with c3:
    st.info("📊 Confidence Score")

with c4:
    st.info("🔒 Media Verification")

st.markdown("---")

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# PREDICTION
# -----------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        # Convert image to RGB
        image = image.convert("RGB")

        # Convert to numpy
        img = np.array(image)

        # Resize to model size
        img = cv2.resize(
            img,
            (MODEL_SIZE, MODEL_SIZE)
        )

        # Normalize
        img = img.astype("float32") / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)

        confidence = float(prediction[0][0])

    st.markdown("---")
    st.subheader("🔍 Analysis Report")

    if confidence > 0.5:

        score = confidence * 100

        st.error("⚠️ FAKE IMAGE DETECTED")

        st.progress(int(score))

        st.metric(
            "Confidence Score",
            f"{score:.2f}%"
        )

    else:

        score = (1 - confidence) * 100

        st.success("✅ REAL IMAGE DETECTED")

        st.progress(int(score))

        st.metric(
            "Confidence Score",
            f"{score:.2f}%"
        )

# -----------------------------------
# ABOUT
# -----------------------------------
st.markdown("---")

st.subheader("About")

st.write(
    """
    DeepFake Vision AI is a CNN and MobileNetV2 based
    image verification system developed to identify
    manipulated facial images.

    The system automatically analyzes uploaded images
    and predicts whether they are Real or Fake using
    Artificial Intelligence techniques.
    """
)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    """
    <div class="footer">
    DeepFake Vision AI | Powered by TensorFlow, OpenCV & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)