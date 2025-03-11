import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Set data directory
data_dir = "data/"
X = []  # Features
y = []  # Labels
user_names = {}  # Dictionary to store user-to-label mapping
user_counter = 0  # Unique label counter

# Dynamically identify users based on filenames
for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        user_name = file.split("_")[0]  # Extract user name (before "_")
        
        # Assign a unique numeric label to each user
        if user_name not in user_names:
            user_names[user_name] = user_counter
            user_counter += 1  # Increment counter for the next user
        
        # Load the feature data
        features = np.load(os.path.join(data_dir, file))
        X.append(features)
        y.append(user_names[user_name])  # Assign numeric label

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
model_path = "models/palm_recognition_model.pkl"
joblib.dump(svm_model, model_path)
print(f"✅ Model saved at {model_path}")

# Print user-label mapping for reference
print("\n🔹 User-to-Label Mapping:")
for name, label in user_names.items():
    print(f"  {label}: {name}")