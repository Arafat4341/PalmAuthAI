from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import cv2
import mediapipe as mp
import io
import joblib
from skimage.feature import hog

app = Flask(__name__)

# Load trained model
model = joblib.load("models/palm_recognition_model.pkl")

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def extract_palm_landmarks(image):
    """Extracts palm landmark coordinates and normalizes them."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])  # Store x, y, z coordinates
            
            # Normalize features
            features = np.array(features)
            features -= features.mean()  # Center around 0
            features /= features.std()   # Scale to unit variance
            
            return features  # Return as a NumPy array
    return None

def extract_hog_features(image):
    """Extracts HOG features from a grayscale palm image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))  # Resize for consistency

    # Compute HOG features
    hog_features, _ = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    return hog_features

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handles authentication requests."""
    try:
        # Try reading file from both form-data and binary input
        file = request.files.get('image') or request.data

        if not file:
            return jsonify({"success": False, "message": "No image provided!"}), 400

        # Convert raw data to numpy array (Handles both form-data and binary uploads)
        if isinstance(file, bytes):  # If sent as binary
            image_np = np.frombuffer(file, np.uint8)
        else:  # If sent as form-data
            image_np = np.frombuffer(file.read(), np.uint8)

        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "message": "Invalid image format!"}), 400

        # Extract features
        palm_landmarks = extract_palm_landmarks(image)
        hog_features = extract_hog_features(image)

        if palm_landmarks is None or hog_features is None:
            return jsonify({"success": False, "message": "No palm detected!"}), 400

        combined_features = np.concatenate((palm_landmarks, hog_features)).reshape(1, -1)

        # Get prediction & confidence scores
        probabilities = model.predict_proba(combined_features)
        max_prob = np.max(probabilities)
        predicted_user = model.predict(combined_features)[0]  # Get predicted user ID

        # Confidence threshold (e.g., 70%)
        if max_prob < 0.7:
            return jsonify({"success": False, "message": "Authentication Failed: Unknown User"}), 401

        # Check if predicted user exists in DB
        conn = sqlite3.connect("data/palm_auth.db")
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE user_id = ?)", (predicted_user,))
        user_exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        if user_exists:
            return jsonify({"success": True, "user_id": int(predicted_user), "message": "Authentication Successful"}), 200
        else:
            return jsonify({"success": False, "message": "Authentication Failed: User Not Found"}), 401

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
