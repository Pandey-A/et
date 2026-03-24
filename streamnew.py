import streamlit as st
import requests
import tempfile
import os
import pandas as pd

# Backend API URLs
VIDEO_API_URL = "http://localhost:5007/predict/video"
IMAGE_API_URL = "http://localhost:5007/predict/image"

st.set_page_config(page_title="Deepfake Detection System", layout="wide")

# Custom CSS for styled tables and layout matching the reference images
st.markdown("""
<style>
    .video-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .fake-label {
        color: #e74c3c;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .real-label {
        color: #27ae60;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .stats-text {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.2rem;
    }
    .section-card {
        background: #fff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e8e8e8;
    }
    .audio-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        border-bottom: 3px solid #4a90d9;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
    }
    .donut-container {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    /* Style dataframes */
    .stDataFrame thead tr th {
        background-color: #4a90d9 !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: center !important;
    }
    .stDataFrame tbody tr td {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Deepfake Detection System")

st.markdown("Choose whether you want to analyze a **video** or an **image** for deepfake detection.")

option = st.radio("Select input type:", ["Video", "Image"])

if option == "Video":
    st.header("🎬 Video Deepfake Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        st.write(f"📁 File size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.video(tmp_path)

        if st.button("🔬 Analyze Video"):
            with open(tmp_path, "rb") as f:
                files = {"video": (uploaded_file.name, f, "video/mp4")}
                with st.spinner("🔄 Analyzing video... This may take some time for large files."):
                    try:
                        response = requests.post(VIDEO_API_URL, files=files, timeout=600)
                        if response.status_code == 200:
                            result = response.json()

                            # ===========================
                            # VIDEO ANALYSIS RESULTS
                            # ===========================
                            st.markdown("---")
                            prediction = result.get("prediction", "Unknown")
                            overall_score = result.get("overall_score", 0)
                            fake_clips = result.get("fake_clips_detected", 0)
                            avg_lips = result.get("avg_lips_manipulation", 0)
                            avg_face = result.get("avg_face_manipulation", 0)
                            video_segments = result.get("video_segments", [])

                            is_fake = "Fake" in prediction

                            # Header with verdict
                            if is_fake:
                                st.markdown(f"""
                                <div class="section-card">
                                    <div class="video-header">Video Analysis Results: &nbsp;
                                        <span class="fake-label">🚨 {prediction} ({overall_score}%)</span>
                                    </div>
                                    <p class="stats-text">Fake clips detected: <strong>{fake_clips}</strong></p>
                                    <p class="stats-text">Average Lips Manipulation: <strong>{avg_lips}%</strong></p>
                                    <p class="stats-text">Average Face Manipulation: <strong>{avg_face}%</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="section-card">
                                    <div class="video-header">Video Analysis Results: &nbsp;
                                        <span class="real-label">✅ {prediction} ({overall_score}%)</span>
                                    </div>
                                    <p class="stats-text">Fake clips detected: <strong>{fake_clips}</strong></p>
                                    <p class="stats-text">Average Lips Manipulation: <strong>{avg_lips}%</strong></p>
                                    <p class="stats-text">Average Face Manipulation: <strong>{avg_face}%</strong></p>
                                </div>
                                """, unsafe_allow_html=True)

                            # Video segments table
                            if video_segments:
                                df_video = pd.DataFrame(video_segments)[["time_range", "lips_manipulation", "face_manipulation"]]
                                df_video.columns = ["Time Range (s)", "Lips Manipulation (%)", "Face Manipulation (%)"]
                                st.dataframe(
                                    df_video.style.set_properties(**{
                                        'text-align': 'center',
                                        'font-size': '14px'
                                    }).set_table_styles([
                                        {'selector': 'th', 'props': [
                                            ('background-color', '#4a90d9'),
                                            ('color', 'white'),
                                            ('font-weight', '600'),
                                            ('text-align', 'center'),
                                            ('padding', '10px')
                                        ]},
                                        {'selector': 'td', 'props': [
                                            ('text-align', 'center'),
                                            ('padding', '8px')
                                        ]}
                                    ]),
                                    use_container_width=True,
                                    hide_index=True
                                )

                            # ===========================
                            # AUDIO ANALYSIS RESULTS
                            # ===========================
                            audio = result.get("audio_analysis", {})
                            if audio:
                                audio_prediction = audio.get("prediction", "Unknown")
                                audio_score = audio.get("overall_score", 0)
                                total_segments = audio.get("total_segments", 0)
                                fake_seg_count = audio.get("fake_segments_count", 0)
                                audio_segments = audio.get("segments", [])

                                audio_is_fake = "Fake" in audio_prediction

                                st.markdown(f"""
                                <div class="section-card">
                                    <div class="audio-header">Audio Analysis Results</div>
                                    <div style="display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;">
                                        <div style="flex: 1; min-width: 200px;">
                                            <p style="font-weight: 600; color: #555; margin-bottom: 0.5rem;">Overall Analysis</p>
                                            <p class="{'fake-label' if audio_is_fake else 'real-label'}">{audio_prediction}({audio_score}%)</p>
                                        </div>
                                        <div style="flex: 0; min-width: 120px; text-align: center;">
                                            <div style="
                                                width: 100px; height: 100px; border-radius: 50%;
                                                background: conic-gradient(
                                                    {'#e74c3c' if audio_is_fake else '#27ae60'} {audio_score * 3.6}deg,
                                                    #e8e8e8 {audio_score * 3.6}deg
                                                );
                                                display: flex; align-items: center; justify-content: center;
                                                margin: 0 auto;
                                            ">
                                                <div style="
                                                    width: 70px; height: 70px; border-radius: 50%;
                                                    background: white;
                                                    display: flex; align-items: center; justify-content: center;
                                                    font-weight: 700; font-size: 1.1rem; color: #333;
                                                ">{audio_score}%</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                                        <div class="metric-box">
                                            <div class="metric-value">{total_segments}</div>
                                            <div class="metric-label">Total Segments</div>
                                        </div>
                                        <div class="metric-box">
                                            <div class="metric-value" style="color: #e74c3c;">{fake_seg_count}</div>
                                            <div class="metric-label">Fake Segments</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Audio segment details table
                                if audio_segments:
                                    st.markdown("**Segment Details**")
                                    # Create dataframe mapping from raw array
                                    df_audio = pd.DataFrame([
                                        {"Segment": s["segment"], "Prediction": s["prediction"], "Confidence": f'{s["confidence"]}%'}
                                        for s in audio_segments
                                    ])

                                    st.dataframe(
                                        df_audio.style.applymap(
                                            lambda v: 'color: #e74c3c; font-weight: 700' if v == 'Fake' else '',
                                            subset=['Prediction']
                                        ).set_properties(**{
                                            'text-align': 'center',
                                            'font-size': '14px'
                                        }).set_table_styles([
                                            {'selector': 'th', 'props': [
                                                ('background-color', '#4a90d9'),
                                                ('color', 'white'),
                                                ('font-weight', '600'),
                                                ('text-align', 'center'),
                                                ('padding', '10px')
                                            ]},
                                            {'selector': 'td', 'props': [
                                                ('text-align', 'center'),
                                                ('padding', '8px')
                                            ]}
                                        ]),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.info("✅ No fake audio segments detected.")

                        else:
                            st.error("❌ Error analyzing video.")
                            st.write(response.json())
                    except requests.exceptions.Timeout:
                        st.error("⏳ Request timed out. Video may be too large or the server is busy.")
        
        # Clean up temporary file
        os.remove(tmp_path)


elif option == "Image":
    st.header("🖼️ Image Deepfake Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        st.write(f"📁 File size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

        if st.button("🔬 Analyze Image"):
            with open(tmp_path, "rb") as f:
                files = {"image": (uploaded_file.name, f, "image/png")}
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        response = requests.post(IMAGE_API_URL, files=files, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Prediction: **{result['prediction']}**")
                            st.write(f"Score: {result['score']:.4f}")
                        else:
                            st.error("❌ Error analyzing image.")
                            st.write(response.json())
                    except requests.exceptions.Timeout:
                        st.error("⏳ Request timed out. Try with a smaller image.")
        
        # Clean up temporary file
        os.remove(tmp_path)
