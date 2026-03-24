import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("xception_deepfake_image.h5")

# Function to preprocess and predict
def predict_fake_or_real(img_path):
    # Load and preprocess the image (Xception expects 299x299)
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    pred = model.predict(img_array)[0][0]

    # Threshold 0.5 (since sigmoid output)
    if pred >= 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    return label, float(pred)

# Example usage
if __name__ == "__main__":
    img_path = "Screenshot 2025-09-08 104051.png"  # replace with your image path
    label, score = predict_fake_or_real(img_path)
    print(f"Prediction: {label} (score: {score:.4f})")
