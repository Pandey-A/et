import streamlit as st
import requests
import tempfile
import os

API_URL = "https://deepfake-api-916h.onrender.com/predict"  # Updated backend

st.title("Deepfake Video Detection")

st.markdown("Upload a video file to check if it's REAL or FAKE.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Show file size
    st.write(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.video(tmp_path)

    if st.button("Analyze Video"):
        with open(tmp_path, "rb") as f:
            files = {"video": (uploaded_file.name, f, "video/mp4")}
            with st.spinner("Analyzing... This may take a while for large videos."):
                try:
                    response = requests.post(API_URL, files=files, timeout=300)  # increase timeout for large files
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: **{result['prediction']}**")
                        st.write(f"Score: {result['score']:.4f}")
                    else:
                        st.error("Error analyzing video.")
                        st.write(response.json())
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Video may be too large or server is busy.")

    # Clean up temporary file
    os.remove(tmp_path)
