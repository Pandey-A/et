# Deepfake Detection API

A Flask-based API for detecting deepfake content in videos and images using machine learning models.

---

## 🚀 Features

- ✅ Health check endpoint to ensure the service and models are loaded.
- ✅ Video upload endpoint to predict if the video is fake or real.
- ✅ Image upload endpoint to predict if the image is fake or real.
- ✅ File format validation and error handling.
- ✅ Temporary file handling to ensure storage efficiency.

---
To run the docker file:
Step 1: Build the Docker image (because Docker caches layers):
docker build -t elevate-trust-flask .

Step 2: Run the container:
docker run -p 5000:5000 elevate-trust-flask

## 📦 Requirements

- Python 3.10 or later
- Flask
- Required ML libraries (install as per your environment setup)

---

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2.Install dependencies:

pip install -r requirements.txt


3.Set environment variables if needed:

export PORT=5000


4.Run the API:

python app3.py

📡 API Endpoints
🔹 GET /health

Check if the API and models are ready.

Request

GET /health


Response

{
    "status": "healthy",
    "models_loaded": true,
    "upload_folder_exists": true
}


Possible Responses

"healthy" – All models are loaded and ready.

"models_not_loaded" – Models are not initialized.

"error" – An internal error occurred.

🔹 POST /predict/video

Upload a video file for deepfake detection.

Request

Method: POST

Content-Type: multipart/form-data

Field: video → Video file.

Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm

Example

curl -X POST -F "video=@/path/to/video.mp4" http://localhost:5000/predict/video


Response

{
    "filename": "video.mp4",
    "score": 0.78,
    "prediction": "FAKE",
    "confidence": 0.78
}


Error Cases

400 → Missing or invalid file.

413 → File too large (max 100MB).

500 → Internal server error.

🔹 POST /predict/image

Upload an image file for deepfake detection.

Request

Method: POST

Content-Type: multipart/form-data

Field: image → Image file.

Supported formats: .jpg, .jpeg, .png, .bmp

Example

curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/predict/image


Response

{
    "filename": "image.jpg",
    "score": 0.34,
    "prediction": "REAL"
}


Error Cases

400 → Missing or invalid file.

413 → File too large (max 100MB).

500 → Internal server error.

⚙ Error Handling

413 Payload Too Large: Files over 100MB are rejected.

500 Internal Server Error: Unexpected issues are logged and reported.

📂 Temporary File Handling

Uploaded files are stored temporarily in the configured upload directory and automatically deleted after processing.

📌 Notes

Ensure that required models are properly loaded at startup.

Use version-specific pip commands if running in an environment with multiple Python versions.

Validate inputs before using the API in production.

Use curl, Postman, or other tools to test endpoints.
# et
