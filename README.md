# Image-Based Crack Prediction System

This project implements an end-to-end web-based system for automated crack detection in infrastructure images using deep learning (MobileNet with transfer learning) and a robust Flask backend. Below is a concise overview of the system's architecture, features, and workflow, illustrated with UML and system diagrams.

---

##  System Overview

- **Purpose:** Automated detection of cracks in structural images, enabling efficient, accurate, and scalable infrastructure monitoring.
- **Core Technologies:** Python, Flask, TensorFlow/Keras, SQLite, HTML/CSS/JS.

---

##  Key Features

- **User Registration & Login:** Secure authentication for clients and admins.
- **Image Upload & Prediction:** Users upload images; the system preprocesses and predicts crack status using a trained deep learning model.
- **Record Management:** Add, edit, delete, and list prediction records in a persistent SQLite database.
- **Model Training Interface:** Admins can trigger model retraining on new datasets directly from the web UI.
- **Role-Based Access:** Both Admin and Client roles with different privileges.

---

## System Architecture

![System Architecture](https://pplx-res.cloudinary.com/image/private/user_uploads/54085632/37224ed3-103a-4109-8154-208f83cff255/System_Architecture.jpg)

- **Frontend:** HTML/CSS/JS for user interaction.
- **Flask Server:** Handles routing, business logic, and API endpoints.
- **Image Processing Module:** Handles preprocessing and augmentation.
- **Machine Learning Model:** MobileNet-based CNN (TensorFlow/Keras) for prediction.
- **SQLite Database:** Stores user data and prediction records.

---

##  Use Case Diagram

![Use Case Diagram](https://pplx-res.cloudinary.com/image/private/user_uploads/54085632/d6d536aa-30c2-403a-9ede-d843ab0a93fc/usecasediagram.drawio.jpg)

- **Actors:** Admin, Client
- **Major Use Cases:**
  - View Home Page
  - User Registration & Login
  - Upload Image for Prediction
  - Predict Crack Status
  - Add/Edit/Delete/View Records
  - Train Model

---

##  Workflow

1. **User Registration/Login:** Users create accounts and log in.
2. **Image Upload:** Users upload images for crack analysis.
3. **Prediction:** Uploaded images are preprocessed and passed to the MobileNet model for prediction (cracked/uncracked).
4. **Record Management:** All predictions are stored; users/admins can view, add, edit, or delete records.
5. **Model Training:** Admin can retrain the model with new data from the UI.

---

##  Technical Stack

- **Backend:** Python, Flask, SQLite
- **ML/DL:** TensorFlow, Keras, OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **Dependencies:** See `requirements.txt` for full list

---

##  How to Run

```
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```


---

##  Model Highlights

- **Architecture:** MobileNet + custom dense layers, transfer learning
- **Augmentation:** Shear, zoom, flip, normalization
- **Performance:** ~99% accuracy, precision, and recall on benchmark datasets

---

##  Project Structure

```
├── app.py
├── requirements.txt
├── media/
│ ├── dataset/
│ └── my_model123.h5
├── static/uploads/
├── templates/
│ ├── user/
│ └── admin/
├── database.db
```



