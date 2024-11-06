# Skin disease detection using ensenble learning

This project focuses on developing a machine learning-based system to detect `Skin Diseases` using `Ensemble learning technique`.The system will analyze images of skin conditions to classify various diseases, helping improve diagnosis. Ensemble learning combines multiple algorithms to improve predictive accuracy, offering a robust approach to handling complex and diverse medical image data.

## Installation

### STEP 1 : **Install prerequisites**

1. [Node.js](https://nodejs.org/)
2. [Git](https://git-scm.com/downloads)
3. [Python required version 3.10.0](https://www.python.org/downloads/release/python-3100/)
4. **Virtualenv** Install via `pip` once Python is installed.
   - `python -m pip install --upgrade pip`
   - `pip install virtualenv`

### STEP 2 : **Setup project frontend and backend**

1. **Clone the Repository**

   - `git clone` https://github.com/Purvesh-PJ/skin_disease_detection.git
   - `cd skin-disease-detection`

2. **Backend Setup (Flask) :**
   - Navigate to the backend directory
      - `cd backend` 
   - Create a virtual environment
      - `python -m venv venv`
   - Activate the virtual environment
      - `venv\Scripts\activate`
   - Install the required Python packages
      - `pip install -r requirements.txt`
   - Install `tensorflow`required python version 3.10.0
      - `pip install tensorflow --force`
   - Install dataset from kaggle
      - Create `data` folder inside `backend/app/` 
      - Download [skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 
      - Extract in `data` folder and `rename to Ham10000`
      - `Delete` skin-cancer-mnist-ham10000.zip
   - Run the Flask server
      - `flask run`

The backend server should now be running on http://localhost:5000

3. **Frontend Setup (React) :**

   - Navigate to the frontend directory
      - `cd frontend`
   - Install dependencies
      - `npm install`
   - Start the React development server:
      - `npm start`

The React app should now be running on http://localhost:3000

