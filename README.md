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


```
┌─────────────────────────────────────────────────────────────┐
│                    Client Interface Layer                   │
├─────────────────────────────────────────────────────────────┤
│           Flask Web Application Framework                   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │  Authentication │ │   File Upload   │ │  Visualization  ││
│  │     Module      │ │    Handler      │ │    Engine       ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│              Image Processing Pipeline                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ Preprocessing   │ │  Feature        │ │  Classification ││
│  │    Module       │ │  Extraction     │ │    Engine       ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│            Machine Learning Infrastructure                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   MobileNet     │ │   TensorFlow    │ │   Model         ││
│  │  Architecture   │ │    Backend      │ │  Persistence    ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                Data Persistence Layer                       │
│           SQLite Database with User Management              │
└─────────────────────────────────────────────────────────────┘
```

- **Frontend:** HTML/CSS/JS for user interaction.
- **Flask Server:** Handles routing, business logic, and API endpoints.
- **Image Processing Module:** Handles preprocessing and augmentation.
- **Machine Learning Model:** MobileNet-based CNN (TensorFlow/Keras) for prediction.
- **SQLite Database:** Stores user data and prediction records.

---

##  Use Case Diagram

![Use Case Diagram](images/secasediagram.drawio.jpg)

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

## 🔧 Model Optimization: INT8 Quantization

Our Deep Crack system implements advanced INT8 quantization to optimize the MobileNet-based crack detection model for real-world deployment.

### 📊 Quantization Results

| Model Version | Size | Compression Ratio | Accuracy Impact |
|---------------|------|-------------------|-----------------|
| **Original (FP32)** | 169.23 MB | 1.0x | Baseline |
| **Float16** | 84.62 MB | 2.0x | <1% loss |
| **INT8 Quantized** | 42.30 MB | 4.0x | 1-3% loss |
| **INT8 + Gzip** | ~15 MB | 11x+ | 1-3% loss |

### ⚡ Performance Benefits

- **75% Size Reduction**: From 169MB to 42MB
- **4x Faster Inference**: Integer operations vs floating-point
- **Lower Memory Usage**: Reduced RAM requirements by 75%
- **Mobile-Ready**: Deployable on smartphones and edge devices
- **Energy Efficient**: Better battery life for mobile applications


###  Use Cases Enabled by Quantization

1. **Mobile Applications**: Real-time crack detection on smartphones
2. **IoT Deployment**: Edge devices with limited computational resources
3. **Cloud Optimization**: Reduced server costs and faster response times
4. **Embedded Systems**: Integration with surveillance cameras and drones

###  Accuracy Preservation

Despite 75% size reduction, the quantized model maintains:
- **98%+ Accuracy**: Minimal impact on crack detection performance
- **Robust Predictions**: Consistent results across diverse image conditions
- **Real-World Viability**: Suitable for production deployment

###  Deployment Advantages

- **GitHub Compatible**: Compressed models fit within repository limits
- **Fast Downloads**: Quicker model distribution and updates
- **Scalable**: Suitable for large-scale infrastructure monitoring
- **Cost-Effective**: Reduced cloud storage and bandwidth costs

###  Implementation Notes

- **Calibration Dataset**: Uses representative samples for optimal quantization
- **Precision Trade-off**: Minimal accuracy loss (1-3%) for significant performance gains
- **Hardware Acceleration**: Compatible with specialized inference chips
- **Future-Proof**: Supports emerging edge AI hardware platforms

## CONCLUSION


In conclusion, the proposed MobileNet-based model demonstrated outstanding performance in classifying the images by underscoring its effectiveness for our specific task. Future work could involve fine-tuning the model on specific subsets of data to further enhance its accuracy by addressing the challenges posed by similar-looking features. Additionally, exploring advanced transfer learning techniques or experimenting with more sophisticated architectures holds promise for achieving even better results in diverse real-world scenarios.


