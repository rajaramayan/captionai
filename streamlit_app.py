import streamlit as st
import os
import tempfile
import time
from utils.predict import generate_caption, load_models
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="📸",
    layout="centered"
)

# --- STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .caption-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border-left: 5px solid #e05252;
        margin-top: 20px;
        font-size: 20px;
        font-style: italic;
        color: #1a1a2e;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("📸 AI Image Caption Generator")
st.write("Upload an image and our Deep Learning model (InceptionV3 + LSTM) will describe what's inside!")

# --- LOAD MODELS ---
@st.cache_resource(show_spinner=False)
def init_models():
    """Load models once and cache them in memory"""
    load_models()
    return True

with st.spinner("Loading AI Models (this takes a moment on first boot)..."):
    init_models()

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate Button
    if st.button("✨ Generate Caption"):
        with st.spinner("Analyzing image..."):
            start_time = time.time()
            
            # Save uploaded file temporarily for the model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Run prediction
                caption, confidence = generate_caption(tmp_path)
                elapsed_time = time.time() - start_time
                
                # Display Results
                st.markdown(f'<div class="caption-box">"{caption.capitalize()}"</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with col2:
                    st.metric("Generation Time", f"{elapsed_time:.2f}s")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

st.markdown("---")
st.markdown("*Built with TensorFlow, Keras, and Streamlit*")
