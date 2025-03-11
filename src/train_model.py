# import sqlite3
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib
# import io

# # Connect to SQLite database
# conn = sqlite3.connect("data/palm_auth.db")
# cursor = conn.cursor()

# # Load training data from SQLite
# cursor.execute("SELECT name, features FROM users")
# rows = cursor.fetchall()

# # Close the connection
# cursor.close()
# conn.close()

# # Check if we have any data
# if not rows:
#     print("❌ No training data found in the database! Please run `store_features.py` first.")
#     exit()

# X = []  # Feature vectors
# y = []  # Labels (user names)
# user_mapping = {}  # User-to-label mapping
# label_counter = 0  # Numeric label counter

# # Process retrieved rows
# for name, features_blob in rows:
#     # Convert binary data back to NumPy array
#     features_bytes = io.BytesIO(features_blob)
#     features = np.load(features_bytes)

#     # Assign a unique label to each user dynamically
#     if name not in user_mapping:
#         user_mapping[name] = label_counter
#         label_counter += 1

#     X.append(features)
#     y.append(user_mapping[name])

# # Convert lists to NumPy arrays
# X = np.array(X)
# y = np.array(y)

# # Split into training (80%) and testing (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train an SVM classifier
# svm_model = SVC(kernel="linear", probability=True)
# svm_model.fit(X_train, y_train)

# # Evaluate accuracy
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# # Save trained model
# model_path = "models/palm_recognition_model.pkl"
# joblib.dump(svm_model, model_path)
# print(f"✅ Model saved at {model_path}")

# # Save user-to-label mapping
# mapping_path = "models/user_mapping.pkl"
# joblib.dump(user_mapping, mapping_path)
# print(f"✅ User mapping saved at {mapping_path}")

# # Print user-label mapping for reference
# print("\n🔹 User-to-Label Mapping:")
# for name, label in user_mapping.items():
#     print(f"  {label}: {name}")

import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import io

# Connect to SQLite database
conn = sqlite3.connect("data/palm_auth.db")
cursor = conn.cursor()

# Load training data from SQLite
cursor.execute("SELECT user_id, features FROM users")
rows = cursor.fetchall()

# Close the connection
cursor.close()
conn.close()

# Check if we have any data
if not rows:
    print("❌ No training data found in the database! Please run `store_features.py` first.")
    exit()

X = []  # Feature vectors
y = []  # Labels (user IDs)
user_ids = set()  # Track unique user IDs

# Process retrieved rows
for user_id, features_blob in rows:
    # Convert binary data back to NumPy array
    features_bytes = io.BytesIO(features_blob)
    features = np.load(features_bytes)

    X.append(features)
    y.append(int(user_id))  # Ensure user_id is treated as an integer
    user_ids.add(user_id)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# Save trained model
model_path = "models/palm_recognition_model.pkl"
joblib.dump(svm_model, model_path)
print(f"✅ Model saved at {model_path}")

print("\n🔹 Users Trained:")
for user_id in sorted(user_ids):
    print(f"  User ID: {user_id}")
