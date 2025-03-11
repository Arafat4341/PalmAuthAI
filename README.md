## **🖐️ PalmAuthAI – AI-Based Palm Recognition System**  
**PalmAuthAI** is a real-time biometric authentication system that recognizes users based on **palm landmarks and texture analysis** using **Machine Learning & Computer Vision**.  

🚀 **Built With:**  
✅ **MediaPipe** – Extracts 21 key palm landmarks  
✅ **HOG (Histogram of Oriented Gradients)** – Captures palm texture  
✅ **SVM (Support Vector Machine)** – Trained to classify palm features  
✅ **SQLite** – Stores user feature data for authentication  
✅ **Flask API** – Allows authentication through HTTP requests  

---

## **📌 Project Structure**  
```
PalmAuthAI/               
│── data/                       # Stores SQLite database
│   ├── palm_auth.db            # Database file storing user features
│── models/                     # Trained ML model & mappings
│   ├── palm_recognition_model.pkl  # Trained SVM model
│── src/                        # Source code for feature extraction, training, & API
│   ├── db_setup.py             # Sets up SQLite database
│   ├── store_features.py       # Captures palm features & saves to database
│   ├── train_model.py          # Trains SVM model using stored features
│   ├── authenticate.py         # Performs real-time authentication
│   ├── api.py                  # Flask API for authentication
│── requirements.txt            # Python dependencies
│── README.md                   # Documentation
```

---

## **📌 Installation & Setup**  

### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Set Up the Database**  
Run this once to initialize SQLite:  
```bash
python src/db_setup.py
```

---

## **📌 Data Collection & Training**  

### **3️⃣ Store Palm Features in Database**  
Run this script to **capture palm features** (press `'s'` to save, `'q'` to quit):  
```bash
python src/store_features.py
```
🔹 **What happens?**  
✅ Opens webcam → detects palm → extracts features → saves to database  

---

### **4️⃣ Train the Machine Learning Model**  
Once features are stored, train the model:  
```bash
python src/train_model.py
```
🔹 **What happens?**  
✅ Loads features from SQLite → Trains an SVM model → Saves the trained model  

---

## **📌 Authentication Methods**  

### **5️⃣ Real-Time Authentication (Using Webcam)**  
Run the following script to **authenticate users in real time**:  
```bash
python src/authenticate.py
```
🔹 **What happens?**  
✅ Captures palm → Extracts features → Predicts user → Checks DB → Authenticates  

---

### **6️⃣ Authentication via API (External Apps/Web Services)**  
Start the Flask API server:  
```bash
python src/api.py
```
🔹 **API Endpoint:**  
- **URL:** `http://localhost:5000/authenticate`  
- **Method:** `POST`  
- **Body:** Upload an image (`image` key)  

#### **🔹 Example API Request (Using cURL)**
```bash
curl -X POST -F "image=@test_palm.jpg" http://localhost:5000/authenticate
```

#### **🔹 Expected API Responses**
| **Scenario** | **Response** |
|-------------|-------------|
| ✅ **Successful Authentication** | `{"success": true, "user_id": 1, "message": "Authentication Successful"}` |
| ❌ **Palm Not Detected** | `{"success": false, "message": "No palm detected!"}` |
| ❌ **Unknown User (Low Confidence)** | `{"success": false, "message": "Authentication Failed: Unknown User"}` |
| ❌ **User Not Found in DB** | `{"success": false, "message": "Authentication Failed: User Not Found"}` |

---

## **📌 Viewing Database Entries**  
To manually inspect the database via SQLite shell:  
```bash
sqlite3 data/palm_auth.db
```
Run SQL queries like:
```sql
SELECT * FROM users;
```

---

## **📌 Next Steps & Future Improvements**  
✅ Improve Model Training with More Data  
✅ Fine-Tune SVM Parameters for Higher Accuracy  
✅ Add a User Registration API for Seamless Enrollment  
✅ Support Cloud Database for Multi-Device Authentication  

---

### **🎯 Conclusion**  
PalmAuthAI is a powerful **proof-of-concept** biometric system. With further improvements, it can be used in **secure access control, identity verification, and AI-driven authentication.** 🚀  

💡 **Want to contribute or improve accuracy? Let’s discuss next steps!**  