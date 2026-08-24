# 🚀 Local Setup & Installation Guide

> **Step-by-step developer guide for cloning, configuring, and running the Skin Disease Detection platform locally on Windows, macOS, or Linux.**

---

## 📋 System Prerequisites

Ensure the following runtimes and tools are installed on your workstation before starting:

| Tool / Runtime | Minimum Version | Recommended Version | Verify Command |
| :--- | :--- | :--- | :--- |
| **Python** | `3.10` | `3.12.x` | `python --version` |
| **Node.js** | `18.x` | `20.x LTS` | `node --version` |
| **npm** | `9.x` | `10.x` | `npm --version` |
| **Git** | `2.x` | Latest | `git --version` |
| **MongoDB** | `6.x` | `7.x` or MongoDB Atlas | `mongod --version` |

---

## ⚡ Quick Start (One-Click Windows Launchers)

For rapid local testing on Windows:

1. **Start Backend**: Double-click [`run_backend.bat`](file:///d:/repos/skin_disease_detection/run_backend.bat) (Starts Flask server on port 5000).
2. **Start Frontend**: Double-click [`run_frontend.bat`](file:///d:/repos/skin_disease_detection/run_frontend.bat) (Starts React dev server on port 3000).
3. Open `http://localhost:3000` in your browser.

---

## ⚙️ Manual Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Purvesh-PJ/skin_disease_detection.git
cd skin_disease_detection
```

---

### Step 2: Configure Environment Variables

Create the root `.env` file by copying from the provided `.env.example`:

```bash
# On Linux / macOS
cp .env.example .env

# On Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` to configure your MongoDB connection and secrets:
```env
# Production & Security Secrets
FLASK_SECRET_KEY=your_custom_flask_secret_key
JWT_SECRET_KEY=your_custom_jwt_secret_key

# Database Connection (Local MongoDB or MongoDB Atlas)
MONGO_DB_URI=mongodb://localhost:27017/skin_disease_db
MONGO_DB_NAME=skin_disease_db

# CORS Whitelist (allow React frontend)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Directory containing .h5 model files
MODEL_DIR=trained_models
```

---

### Step 3: Backend Setup (Python & Flask)

1. Navigate to the backend directory and create a virtual environment:
```bash
cd backend
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows (PowerShell / CMD)
.\venv\Scripts\activate

# On Linux / macOS
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Verify backend installation by running the test suite or starting the server:
```bash
python main.py
```
> The backend server will initialize on `http://localhost:5000`.

---

### Step 4: Machine Learning Model Weights

The platform requires 3 trained CNN weight files in the `trained_models/` directory:

```
skin_disease_detection/
└── trained_models/
    ├── densenet121.h5
    ├── efficientnetb3.h5
    └── resnet101.h5
```

**Option A: Automated Cloud Download (Recommended)**
If `MODEL_ZIP_URL` is configured in your `.env`, the backend will automatically download and unpack the models upon your first scan request.

**Option B: Manual Download**
* Download the pre-trained weights archive from [Google Drive Storage](https://drive.google.com/file/d/1RdJgiNy94sM2OfU3N6RqFbDBD-BawNYY/view).
* Extract the `.h5` files into the `trained_models/` directory in the repository root.

---

### Step 5: Frontend Setup (React 18)

1. Open a new terminal and navigate to the `frontend/` directory:
```bash
cd frontend
```

2. Install npm packages:
```bash
npm install
```

3. Verify environment configuration in `frontend/.env.development`:
```env
REACT_APP_API_URL=http://localhost:5000
```

4. Start the React development server:
```bash
npm start
```
> The browser will automatically open `http://localhost:3000`.

---

## 🐳 Running with Docker Compose

If you have Docker and Docker Compose installed, launch the entire stack in one command:

```bash
docker-compose up --build
```

* **Frontend UI**: `http://localhost:3000`
* **Backend API**: `http://localhost:5000`
* **MongoDB**: `localhost:27017`

---

## ✅ Verifying the Installation

1. **Check Backend Health**: Visit `http://localhost:5000/health`. Expected response:
   ```json
   { "status": "healthy", "service": "Skin Disease Detection API", "version": "1.0.0" }
   ```
2. **Check Frontend & 1-Click Demo**:
   * Navigate to `http://localhost:3000/login`.
   * Click **"1-Click Demo Login"**.
   * Select a lesion sample and click **"Analyze Image"**.

---

## 🔗 Cross References
* [5-Minute Quickstart Guide (`quickstart.md`)](./quickstart.md)
* [Troubleshooting Diagnostic Playbook (`../troubleshooting/diagnostic-playbook.md`)](../troubleshooting/diagnostic-playbook.md)
* [REST API Reference (`../api-reference/endpoints.md`)](../api-reference/endpoints.md)
