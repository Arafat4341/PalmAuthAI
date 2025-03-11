import cv2
import mediapipe as mp
import numpy as np
from skimage.feature import hog

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def extract_palm_landmarks(frame):
    """Extracts 21 palm landmark coordinates and normalizes them."""
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

# Capture palm features and combine them
cap = cv2.VideoCapture(0)
saved_features = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract features
    palm_landmarks = extract_palm_landmarks(frame)
    hog_features = extract_hog_features(frame)

    if palm_landmarks is not None and hog_features is not None:
        combined_features = np.concatenate((palm_landmarks, hog_features))  # Merge both

        cv2.putText(frame, "Press 'S' to Save Features", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Palm Feature Extraction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and palm_landmarks is not None and hog_features is not None:
        saved_features = combined_features
        print("✅ Combined Palm Features Saved:", saved_features)
        break  # Exit after saving
    elif key == ord('q'):
        break  # Quit without saving

cap.release()
cv2.destroyAllWindows()

# Save the features for training
if saved_features is not None:
    np.save("combined_palm_features.npy", saved_features)
    print("✅ Features saved to combined_palm_features.npy")
