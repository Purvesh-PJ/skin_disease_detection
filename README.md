<<<<<<< HEAD
# Skin Disease Detection using Ensemble Learning

## Overview
This project implements an advanced machine learning system for detecting skin diseases using ensemble learning techniques. The system analyzes skin condition images to accurately classify various dermatological conditions, aiding in medical diagnosis. By leveraging ensemble learning, which combines multiple deep learning models, we achieve higher predictive accuracy and robustness in handling diverse medical image data.

## Features
- Multi-model ensemble learning approach
- Support for various skin disease classifications
- User-friendly web interface
- Secure authentication system
- Real-time image processing and prediction
- High accuracy through combined model predictions

## Tech Stack
- **Frontend**: React.js
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Database**: MongoDB
- **Authentication**: JWT

## Prerequisites

Before installation, ensure you have the following installed:
1. [Node.js](https://nodejs.org/) (v14 or higher)
2. [Git](https://git-scm.com/downloads)
3. [Anaconda](https://www.anaconda.com/download/success) (Python 3.12.0)
4. MongoDB (if using local database)
=======
# Skin Disease Detection Using Ensemble Learning

![Skin Disease Detection](https://img.shields.io/badge/Project-Skin%20Disease%20Detection-blue)
![Python](https://img.shields.io/badge/Backend-Python%203.12-blue)
![React](https://img.shields.io/badge/Frontend-React%2018-blue)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)
>>>>>>> aa4cef4f1a0223fa67e061da4e4c12d8a6b91bff

## 📋 Overview

<<<<<<< HEAD
### 1. Clone the Repository
```bash
git clone https://github.com/Purvesh-PJ/skin_disease_detection.git
cd skin_disease_detection
```

### 2. Backend Setup

#### Set up Python Environment
```bash
cd backend
conda create --name skin_disease python=3.12.0
conda activate skin_disease
pip install -r requirements.txt
```

#### Dataset Installation
1. Create a `data` folder in the `backend/app/` directory
2. Download the [Skin Cancer MNIST: HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
3. Extract the downloaded file to the `data` folder
4. Rename the extracted folder to `Ham10000`
5. Remove the downloaded zip file

#### Start the Backend Server
```bash
python main.py
```
The backend server will run on http://localhost:5000

### 3. Frontend Setup

```bash
cd frontend
npm install
npm start
```
The frontend application will be available at http://localhost:3000

## Usage
1. Register a new account or login with existing credentials
2. Navigate to the disease prediction tool
3. Upload a skin image for analysis
4. View the prediction results and confidence scores

## Project Structure
- `frontend/`: React application files
- `backend/`: Flask server and ML models
  - `app/ai_models/`: Contains all ML model implementations
  - `app/routes/`: API endpoints
  - `app/services/`: Business logic and services
  - `app/utils/`: Utility functions and helpers

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset: HAM10000 dataset from Kaggle
- Base models: DenseNet121, EfficientNet, ResNet50, ResNet101
=======
This project focuses on developing a machine learning-based system to detect skin diseases using ensemble learning techniques. The system analyzes images of skin conditions to classify various diseases, helping improve diagnosis accuracy. Ensemble learning combines multiple algorithms to enhance predictive accuracy, offering a robust approach to handling complex and diverse medical image data.

![Project Screenshot](https://via.placeholder.com/800x400?text=Skin+Disease+Detection+Screenshot)

## ✨ Features

- **Image Upload & Analysis**: Upload skin images for instant disease detection
- **Ensemble Learning Model**: Utilizes multiple ML algorithms for improved accuracy
- **Disease Classification**: Identifies various skin conditions from uploaded images
- **Confidence Scoring**: Provides confidence level for each prediction
- **Detailed Results**: Displays comprehensive information about detected conditions
- **User Authentication**: Secure login system to protect user data
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## 🛠️ Technologies Used

### Backend
- **Python 3.12**: Core programming language
- **Flask**: Web framework for API development
- **TensorFlow**: Machine learning library for model development
- **Scikit-learn**: Machine learning tools for model evaluation and ensemble techniques
- **Flask-JWT-Extended**: Authentication using JSON Web Tokens
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **MongoDB**: Database for storing user information and results

### Frontend
- **React 18**: JavaScript library for building the user interface
- **React Router**: Navigation and routing
- **Axios**: HTTP client for API requests
- **Styled Components**: Component-level styling
- **React Icons**: Icon library

## 🔧 Installation

### Prerequisites

1. [Node.js](https://nodejs.org/) (v14 or higher)
2. [Git](https://git-scm.com/downloads)
3. [Anaconda](https://www.anaconda.com/download/success) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
4. [MongoDB](https://www.mongodb.com/try/download/community) (optional if using cloud MongoDB)

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/Purvesh-PJ/skin_disease_detection.git
cd skin_disease_detection
```

#### 2. Backend Setup (Python/Flask)

```bash
# Navigate to backend directory
cd backend

# Create and activate Conda environment
conda create --name skin_disease_env python=3.12.0
conda activate skin_disease_env

# Install required packages
pip install -r requirements.txt

# Set up dataset
mkdir -p app/data
# Download the HAM10000 dataset from Kaggle:
# https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# Extract in app/data folder and rename to Ham10000

# Start the Flask server
python main.py
```

The backend server will run on http://localhost:5000

#### 3. Frontend Setup (React)

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The React application will run on http://localhost:3000

## 📊 Dataset

This project uses the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), which contains 10,000 dermatoscopic images of pigmented skin lesions across seven different disease categories:

- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

## 🧠 Model Architecture

The project implements an ensemble learning approach combining:

- Convolutional Neural Networks (CNN)
- Support Vector Machines (SVM)
- Random Forests
- Gradient Boosting

This ensemble approach improves prediction accuracy by leveraging the strengths of multiple algorithms.

## 🔒 Authentication

The application uses JWT (JSON Web Tokens) for secure authentication. Users need to register and log in to access the disease prediction functionality.

## 🚀 Usage

1. Register/Login to the application
2. Navigate to the disease prediction tool
3. Upload an image of the skin condition
4. Wait for the analysis to complete
5. View the detailed results including:
   - Predicted disease
   - Confidence score
   - Disease information and recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Purvesh PJ - [GitHub Profile](https://github.com/Purvesh-PJ)

Project Link: [https://github.com/Purvesh-PJ/skin_disease_detection](https://github.com/Purvesh-PJ/skin_disease_detection)

## 🙏 Acknowledgements

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
>>>>>>> aa4cef4f1a0223fa67e061da4e4c12d8a6b91bff

