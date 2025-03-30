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

## Installation

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

