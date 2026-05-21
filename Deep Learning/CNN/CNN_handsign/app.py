import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
model = load_model("sign_language_cnn_model.keras", compile=False)

# Labels
labels = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E',
    5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N',
    14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V',
    22:'W', 23:'X', 24:'Y'
}

# Page Config
st.set_page_config(
    page_title="AI Sign Language",
    layout="centered"
)

# Background Styling
st.markdown("""
<style>

.stApp{

background-image:url("https://images.unsplash.com/photo-1515879218367-8466d910aaa4?q=80&w=1920&auto=format&fit=crop");

background-size:cover;

background-position:center;
}

.main-box{

background:rgba(0,0,0,0.65);

padding:40px;

border-radius:20px;

margin-top:40px;
}

.title{

text-align:center;

font-size:55px;

font-weight:bold;

color:white;
}

.sub{

text-align:center;

font-size:22px;

color:white;

margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# Main Container
st.markdown("<div class='main-box'>", unsafe_allow_html=True)

st.markdown(
'<p class="title">🤟 AI Sign Language</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="sub">✨ Upload hand sign image and predict instantly 🚀</p>',
unsafe_allow_html=True
)

# Upload
uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    # Display Image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=300
    )

    # Preprocess Image
    image = Image.open(uploaded_file).convert("L")

    image = image.resize((28,28))

    image = np.array(image) / 255.0

    image = image.reshape(1,28,28,1)

    # Prediction
    pred = model.predict(image, verbose=0)

    result = labels[np.argmax(pred)]

    # Output
    st.success(f"🎯 Predicted Sign : {result}")

st.markdown("</div>", unsafe_allow_html=True)
