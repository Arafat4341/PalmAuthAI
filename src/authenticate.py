import sqlite3
import numpy as np
import cv2
import mediapipe as mp
import io
import joblib
from skimage.feature import hog

# Load trained model & user mapping
model = joblib.load("models/palm_recognition_model.pkl")

# Connect to SQLite
conn = sqlite3.connect("data/palm_auth.db")
cursor = conn.cursor()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def extract_palm_landmarks(frame):
    """Extracts palm landmark coordinates and normalizes them."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

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

# Start webcam for authentication
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features
    palm_landmarks = extract_palm_landmarks(frame)
    hog_features = extract_hog_features(frame)

    if palm_landmarks is not None and hog_features is not None:
        combined_features = np.concatenate((palm_landmarks, hog_features))  # Merge both
        combined_features = combined_features.reshape(1, -1)  # Reshape for model

        # Get prediction & confidence scores
        probabilities = model.predict_proba(combined_features)
        max_prob = np.max(probabilities)
        predicted_user = model.predict(combined_features)[0]  # Get predicted user ID

        print('predicted user result from SVM', predicted_user)
        print('Probability score:', max_prob)

        # Confidence threshold (e.g., 70%)
        if max_prob < 0.7:
            print("❌ Authentication Failed: Unknown User")
            cv2.putText(frame, "Authentication Failed", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Check if predicted user exists in DB
            cursor.execute("SELECT EXISTS(SELECT 1 FROM users WHERE user_id = ?)", (predicted_user,))
            user_exists = cursor.fetchone()[0]

            if user_exists:
                print(f"✅ Welcome, User ID {predicted_user}!")
                cv2.putText(frame, f"Welcome, User {predicted_user}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("❌ Authentication Failed: User Not Found in Database")
                cv2.putText(frame, "User Not Found", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Palm Authentication", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Quit authentication

cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
print("✅ Authentication session ended.")
