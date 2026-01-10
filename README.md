# 🧠 Skin Disease Detection Using Deep Ensemble Learning

![Project](https://img.shields.io/badge/Project-Skin%20Disease%20Detection-blue)
![Python](https://img.shields.io/badge/Backend-Python%203.12-blue)
![React](https://img.shields.io/badge/Frontend-React%2018-blue)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

A web-based system to detect skin diseases from images using deep learning. The project uses **ensemble learning with CNN architectures** to improve diagnostic accuracy.

Users can upload a skin image, and the backend uses three trained CNNs (EfficientNetB3, ResNet101, DenseNet121) to classify it. The predictions are averaged to determine the final result.

---

## ✨ Features

- 🖼️ Upload dermatoscopic images for prediction  
- 🧠 Ensemble of EfficientNetB3, ResNet101, and DenseNet121  
- ✅ JWT-based authentication system (register/login)
- 📊 Confidence score for each prediction
- 📖 Description of predicted skin condition
- 📱 Responsive UI using React 18

---

## 🛠️ Tech Stack

### 🔙 Backend
- Python 3.12
- Flask + Flask-JWT-Extended
- TensorFlow / Keras
- MongoDB (via PyMongo)
- OpenCV & Albumentations for preprocessing
- dotenv for environment configs

### 🔜 Frontend
- React 18
- Axios
- React Router DOM
- Styled Components
- React Icons

---

## 📁 Project Structure (Simplified)

```
skin_disease_detection/
├── backend/
│ ├── app/ # Backend app logic
│ ├── main.py # Flask app entry
│ ├── requirements.txt # Python dependencies
│ └── uploads/ # Uploaded images
├── frontend/
│ └── src/ # React app
├── trained_models/ # Pretrained .h5 models
└── README.md
```


---

## 📊 Dataset

- Dataset: [HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- 10,000 labeled images across 7 classes:
  - `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

To prepare:
- Extract dataset into:  
  `backend/data/skin_disease_dataset/base_dir/`  
  with subdirectories: `train_dir/`, `val_dir/`, `test_dir/`

---

## 🧠 Model Details

- Ensemble of:
  - ✅ DenseNet121
  - ✅ EfficientNetB3
  - ✅ ResNet101
- Each model is trained independently.
- Final prediction: average of softmax scores from all 3 models.

### 📥 Download Trained Models

Download the pre-trained models from Google Drive:

🔗 [Download Models](https://drive.google.com/file/d/1RdJgiNy94sM2OfU3N6RqFbDBD-BawNYY/view)

After downloading, extract and place the `trained_models` folder in the project root directory:

```
skin_disease_detection/
├── backend/
├── frontend/
├── trained_models/      ← Place here
│   ├── densenet121.h5
│   ├── efficientnetb3.h5
│   └── resnet101.h5
└── README.md
```


---

## 🔐 Authentication

- JWT-based login/register system
- Tokens must be passed in `Authorization` header for prediction requests.

---

## 🧪 API Endpoints

| Endpoint            | Method | Auth | Description                  |
|---------------------|--------|------|------------------------------|
| `/auth/register`    | POST   | ❌   | Register a new user         |
| `/auth/login`       | POST   | ❌   | Login, receive JWT token    |
| `/predict`          | POST   | ✅   | Upload image & get results  |

---

## 🚀 Getting Started

### 🧰 Prerequisites

- Python 3.12+
- Node.js v18+
- MongoDB (local or cloud)
- Git

---

### ⚙️ Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

###  Create a .env file in backend/ with:

```bash
FLASK_SECRET_KEY=YourSecretKey
JWT_SECRET_KEY=YourJWTSecretKey
MONGO_URI=mongodb://localhost:27017/skin_disease_db
```

### (Optional) Train Your Own Models

If you want to train the models yourself instead of using the pre-trained ones, download and extract the HAM10000 dataset into:

```bash 
backend/data/skin_disease_dataset/base_dir/
├── train_dir/
├── val_dir/
└── test_dir/
```
### Then run:

```bash
python main.py
```

### 🌐 Frontend Setup
```bash 
cd frontend
npm install
npm start
```

### 🧪 Prediction Flow
```bash
Login → Receive JWT token

Upload skin image

Backend runs predictions using all 3 models

Softmax probabilities are averaged

Highest scoring class is selected

Response includes:

Predicted disease

Confidence score

Disease name + description
```

### 📄 License

MIT License See [LICENSE]() file.


### 🙋 Contact

[@Purvesh-PJ](https://github.com/Purvesh-PJ) 


### 🙏 Acknowledgements

- [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [TensorFlow](https://www.tensorflow.org/)
- [React](https://react.dev/)
- [Flask](https://flask.palletsprojects.com/en/stable/)
