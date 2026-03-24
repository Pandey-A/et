from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import logging
from functools import lru_cache

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Paths (update as needed) ===
VIDEO_MODEL_PATH = "video_deepfake.h5"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Constants ===
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# === Initialize Flask app ===
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins; restrict later in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# === Global variables for models ===
feature_extractor = None
video_model = None

def crop_center_square(frame):
    """Crop frame to center square"""
    try:
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]
    except Exception as e:
        logger.error(f"Error in crop_center_square: {e}")
        raise

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """Load and preprocess video frames"""
    logger.info(f"Loading video from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame is None or frame.size == 0:
                logger.warning(f"Empty frame at position {frame_count}")
                continue
                
            try:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
                frames.append(frame)
                frame_count += 1
                
                if max_frames and len(frames) >= max_frames:
                    break
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error reading video: {e}")
        raise
    finally:
        cap.release()
    
    if len(frames) == 0:
        raise ValueError("No valid frames found in video")
    
    logger.info(f"Loaded {len(frames)} frames from video")
    return np.array(frames)

@lru_cache(maxsize=1)
def build_feature_extractor():
    """Build and cache feature extractor model"""
    logger.info("Building feature extractor...")
    try:
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input
        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)
        outputs = feature_extractor(preprocessed)
        model = keras.Model(inputs, outputs, name="feature_extractor")
        logger.info("Feature extractor built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building feature extractor: {e}")
        raise

def build_video_model(max_seq_length, num_features):
    """Build video classification model"""
    logger.info("Building video model...")
    try:
        frame_features_input = keras.Input((max_seq_length, num_features))
        mask_input = keras.Input((max_seq_length,), dtype="bool")
        x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(8, activation="relu")(x)
        output = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model([frame_features_input, mask_input], output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        logger.info("Video model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building video model: {e}")
        raise

def prepare_single_video(frames):
    """Prepare video frames for prediction"""
    logger.info(f"Preparing video with {len(frames)} frames")
    try:
        # Limit frames to MAX_SEQ_LENGTH to prevent memory issues
        if len(frames) > MAX_SEQ_LENGTH:
            # Sample frames evenly across the video
            indices = np.linspace(0, len(frames) - 1, MAX_SEQ_LENGTH, dtype=int)
            frames = frames[indices]
            logger.info(f"Downsampled to {len(frames)} frames")
        
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            logger.info(f"Processing {length} frames for feature extraction")
            
            # Process frames in smaller batches to avoid memory issues
            batch_size = 5  # Process 5 frames at a time
            for start_idx in range(0, length, batch_size):
                end_idx = min(start_idx + batch_size, length)
                logger.info(f"Processing frames {start_idx} to {end_idx-1}")
                
                for j in range(start_idx, end_idx):
                    try:
                        # Add batch dimension for single frame prediction
                        frame_batch = np.expand_dims(batch[j], axis=0)
                        features = feature_extractor.predict(frame_batch, verbose=0)
                        frame_features[i, j, :] = features[0]  # Remove batch dimension
                    except Exception as e:
                        logger.error(f"Error extracting features for frame {j}: {e}")
                        # Use zero features for failed frames
                        frame_features[i, j, :] = np.zeros(NUM_FEATURES)
                        
            frame_mask[i, :length] = 1
            
        logger.info("Video preparation completed")
        return frame_features, frame_mask
    except Exception as e:
        logger.error(f"Error in prepare_single_video: {e}")
        raise

def sequence_prediction(video_path):
    """Predict if video is deepfake"""
    logger.info(f"Starting prediction for: {video_path}")
    try:
        frames = load_video(video_path)
        frame_features, frame_mask = prepare_single_video(frames)
        pred = video_model.predict([frame_features, frame_mask], verbose=0)
        result = float(pred[0][0])
        logger.info(f"Prediction completed. Score: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in sequence_prediction: {e}")
        raise

def initialize_models():
    """Initialize models on startup"""
    global feature_extractor, video_model
    
    logger.info("Initializing models...")
    
    # Check if model file exists
    if not os.path.exists(VIDEO_MODEL_PATH):
        logger.error(f"Model file not found: {VIDEO_MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {VIDEO_MODEL_PATH}")
    
    try:
        # Build models
        feature_extractor = build_feature_extractor()
        video_model = build_video_model(MAX_SEQ_LENGTH, NUM_FEATURES)
        
        # Load weights
        video_model.load_weights(VIDEO_MODEL_PATH)
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

# === Flask routes ===

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "Deepfake Detection API is running!",
        "models_loaded": feature_extractor is not None and video_model is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        models_ready = feature_extractor is not None and video_model is not None
        return jsonify({
            "status": "healthy" if models_ready else "models_not_loaded",
            "models_loaded": models_ready,
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_video():
    """Predict if uploaded video is deepfake"""
    try:
        # Initialize models on first request if not already loaded
        if feature_extractor is None or video_model is None:
            try:
                initialize_models()
            except Exception as e:
                logger.error(f"Failed to initialize models: {e}")
                return jsonify({"error": f"Model initialization failed: {str(e)}"}), 503
        
        # Check if file is uploaded
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded"}), 400
        
        video = request.files['video']
        
        if video.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if not any(video.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({"error": "Unsupported video format"}), 400
        
        filename = secure_filename(video.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save uploaded file
        video.save(file_path)
        logger.info(f"Video saved to: {file_path}")
        
        try:
            # Process video
            score = sequence_prediction(file_path)
            label = "FAKE" if score >= 0.5 else "REAL"
            confidence = score if score >= 0.5 else 1 - score
            
            return jsonify({
                "filename": filename,
                "score": score,
                "prediction": label,
                "confidence": confidence
            })
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up file {file_path}: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error in predict_video: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": "Internal server error"}), 500

# Initialize models before starting the app
try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    # You might want to exit here or handle this differently
    # sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


