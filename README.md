# 🧠 AI Skin Disease & Cancer Detection Platform

<div align="center">

### 🔬 Fast, AI-powered skin lesion analysis & cancer screening in seconds

Upload a photo of a skin mole or lesion &mdash; our 3-AI deep learning system analyzes the image, predicts the exact skin condition across 7 disease types, and gives a clear confidence score with medical guidance.

<br />

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Open_Web_App-00C7B7?style=for-the-badge&logo=vercel&logoColor=white)](https://skin-disease-detection.vercel.app/)
[![API Backend](https://img.shields.io/badge/Backend_API-Render_Live-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://skin-disease-detection-6kxm.onrender.com/health)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br />

**[🚀 Try Live Demo (No Signup Needed)](#-try-it-live-in-30-seconds-zero-setup)** • [📸 Screenshots](#-screenshots--ui-walkthrough) • [📖 Full Docs](docs/README.md) • [⚡ Quickstart](#-local-installation-in-3-steps) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## ⚡ Try It Live in 30 Seconds (Zero Setup)

You can test the entire platform right in your browser without installing anything or creating an account:

<div align="center">

👉 **[Launch Live Web Application](https://skin-disease-detection.vercel.app/)** 👈

</div>

### How to test in 3 easy clicks:
1. Open the **[Live Demo Link](https://skin-disease-detection.vercel.app/)**.
2. Click the blue **"1-Click Demo Login"** button *(Instant access &mdash; no email or password needed)*.
3. Under the **Sample Lesions** section, click on any pre-loaded test image (e.g. *Melanoma* or *Benign Mole*).
4. Click **"Analyze Image"** &mdash; watch the 3 AI models classify the disease with confidence score and medical breakdown!

> **Manual Demo Credentials (if needed):**
> * **Email:** `demo@skindisease.ai`
> * **Password:** `DemoUser@123`

---

## 📸 Screenshots & UI Walkthrough

<div align="center">
  <img src="frontend/public/screenshots/skin_disease_platform_screenshot.png" alt="Skin Disease Platform Dashboard" width="100%" style="border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.12);" />
</div>

<br />

| 🔍 **AI Diagnostic Scanner & Analysis** | 📊 **Patient History & Audit Logs** |
| :---: | :---: |
| <img src="frontend/public/screenshots/skin_disease_platform_screenshot.png" width="100%" alt="Scanner View" /> | <img src="frontend/public/screenshots/skin_disease_platform_screenshot.png" width="100%" alt="History View" /> |

| 🌓 **Dark / Light Theme Engine** | 🧪 **1-Click HAM10000 Sample Picker** |
| :---: | :---: |
| <img src="frontend/public/screenshots/skin_disease_platform_screenshot.png" width="100%" alt="Theme Engine" /> | <img src="frontend/public/screenshots/skin_disease_platform_screenshot.png" width="100%" alt="Sample Images Picker" /> |

---

## 💡 What is this project? (Simple Explanation)

Skin cancers (like **Melanoma**) can be fatal if ignored, but are **almost 100% curable if detected early**. However, getting an appointment with a dermatologist can take weeks.

This web application acts as an **intelligent early-screening assistant**:
1. **You upload a skin image** (or take a photo of a mole/rash).
2. **3 powerful AI models analyze it together** (like a panel of 3 specialist doctors comparing opinions).
3. **The app gives a clear result**:
   * What condition it looks like (Melanoma, Carcinoma, Benign Mole, etc.).
   * How confident the AI is (e.g. `96% Confidence`).
   * What it means and whether urgent doctor consultation is recommended.

---

## ✨ Key Features at a Glance

* 🧠 **3-in-1 AI Ensemble**: Combines 3 neural networks (**EfficientNetB3**, **ResNet101**, **DenseNet121**) so predictions don't rely on a single model.
* 🎯 **Detects 7 Skin Conditions**: Accurately classifies deadly cancers (Melanoma, Basal Cell Carcinoma), pre-cancers (Actinic Keratosis), and benign harmless moles.
* 🛡️ **Zero-Downtime Demo Mode**: Never crashes or blocks evaluators, even if cloud servers are waking up from sleep.
* 🧪 **Built-in Sample Lesion Picker**: Test immediately with pre-curated dermatoscopy samples without needing your own photos.
* 🎨 **Clean Clinical Dashboard**: Modern React 18 UI with drag-and-drop uploads, instant previews, and Dark/Light mode toggle.
* 🔐 **Secure User Accounts**: Register your own account with encrypted passwords (Bcrypt) and secure login (JWT).
* 🗄️ **Saved Scan History**: Past scans are securely saved to MongoDB Atlas cloud database for easy tracking over time.

---

## 🔬 The 7 Skin Diseases Detected (HAM10000 Dataset)

Our AI is trained on **10,015 real dermatoscopic images** from the gold-standard International Skin Imaging Collaboration (ISIC) HAM10000 dataset:

| Disease Code | Disease Name | What it is | Risk Level |
| :--- | :--- | :--- | :--- |
| **`mel`** | **Melanoma** | Serious, aggressive form of skin cancer. | 🚨 **High Risk (Malignant)** |
| **`bcc`** | **Basal Cell Carcinoma** | Common type of non-melanoma skin cancer. | ⚠️ **Moderate Risk (Malignant)** |
| **`akiec`** | **Actinic Keratosis** | Rough, scaly patch that can lead to cancer. | ⚠️ **Pre-Cancerous** |
| **`bkl`** | **Benign Keratosis** | Harmless, non-cancerous skin growth (age spot/seborrheic). | 🟢 **Safe (Benign)** |
| **`nv`** | **Melanocytic Nevi** | Normal, common everyday mole. | 🟢 **Safe (Benign)** |
| **`df`** | **Dermatofibroma** | Harmless, firm bump beneath the skin. | 🟢 **Safe (Benign)** |
| **`vasc`** | **Vascular Lesion** | Harmless blood vessel growth (cherry angioma). | 🟢 **Safe (Benign)** |

---

## 🛠️ Tech Stack

```
Frontend:     React 18, React Router v6, Styled Components, Axios
Backend:      Python 3.12, Flask, Flask-JWT-Extended, Flask-Bcrypt, Gunicorn
Machine Learning: TensorFlow 2.x, Keras (EfficientNetB3, ResNet101, DenseNet121), OpenCV
Database:     MongoDB Atlas (Cloud) / Local MongoDB
DevOps:       Docker, Docker Compose, Nginx, Vercel, Render
```

---

## ⚡ Local Installation in 3 Steps

If you want to run the project locally on your machine:

### Prerequisites
* Python 3.10+ (Recommended: 3.12)
* Node.js 18+ & npm
* Git

---

### 1. One-Click Launchers (Windows)
* **Backend:** Double-click [`run_backend.bat`](run_backend.bat) (Runs Flask on `localhost:5000`)
* **Frontend:** Double-click [`run_frontend.bat`](run_frontend.bat) (Runs React on `localhost:3000`)

---

### 2. Manual Setup (All Operating Systems)

```bash
# 1. Clone the repository
git clone https://github.com/Purvesh-PJ/skin_disease_detection.git
cd skin_disease_detection

# 2. Setup Backend (Terminal 1)
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate  |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python main.py

# 3. Setup Frontend (Terminal 2)
cd ../frontend
npm install
npm start
```
> App opens automatically at `http://localhost:3000`.

---

### 3. Docker Compose (1 Command)
```bash
docker-compose up --build
```

---

## 📥 AI Model Weights

The 3 pre-trained neural network weights (`.h5` files) are stored on Google Drive:

🔗 **[Download Trained AI Models (~395 MB)](https://drive.google.com/file/d/1RdJgiNy94sM2OfU3N6RqFbDBD-BawNYY/view)**

Extract files into the `trained_models/` folder:
```
trained_models/
├── densenet121.h5
├── efficientnetb3.h5
└── resnet101.h5
```
*(Note: If `MODEL_ZIP_URL` is set in `.env`, the backend automatically downloads missing weights on its first run).*

---

## 📚 Complete Documentation Hub

For in-depth technical guides, architecture deep-dives, and API specs, visit our [`docs/`](docs/README.md) portal:

* 🚀 **[Getting Started Guide](docs/getting-started/setup-and-installation.md)** &bull; [Quickstart](docs/getting-started/quickstart.md)
* 🏛️ **[System Architecture](docs/architecture/system-architecture.md)** &bull; [Frontend SPA](docs/architecture/frontend-architecture.md) &bull; [Data Models](docs/architecture/data-models.md)
* 🧠 **[Machine Learning Pipeline](docs/machine-learning/ml-pipeline.md)** &bull; [Model Training](docs/machine-learning/model-training.md)
* 📡 **[REST API Reference](docs/api-reference/endpoints.md)** (Endpoints, Schemas & cURL examples)
* 🐳 **[DevOps & Cloud Deployment](docs/devops/deployment-guide.md)** &bull; [Docker](docs/devops/docker-and-containerization.md) &bull; [Environment Variables](docs/devops/environment-variables.md)
* 🔒 **[Security & Governance](docs/security/security-and-compliance.md)**
* 🔍 **[Troubleshooting Playbook](docs/troubleshooting/diagnostic-playbook.md)**

---

## 🔒 Medical & Clinical Disclaimer

> [!IMPORTANT]
> **Clinical Notice:** This software is developed for **educational, research, and assistive early-screening demonstration purposes only**. It is **not** a certified medical diagnostic device and is **not** a replacement for clinical diagnosis or biopsy by a licensed dermatologist or medical professional.

---

## 📜 Open-Source Governance

* ⚖️ **[MIT License](LICENSE)** &mdash; Free to use, modify, and distribute.
* 🤝 **[Contributing Guide](CONTRIBUTING.md)** &mdash; How to contribute to this repo.
* 📜 **[Code of Conduct](CODE_OF_CONDUCT.md)** &mdash; Community standards.
* 🔒 **[Security Policy](SECURITY.md)** &mdash; Vulnerability disclosure.
* 📋 **[Changelog](CHANGELOG.md)** &mdash; Version release history.

---

## 👤 Author & Credits

* **Author & Lead Developer**: **Purvesh Patel** ([@Purvesh-PJ](https://github.com/Purvesh-PJ))
* **Dataset**: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (ISIC Archive)

<div align="center">
  <sub>Built with ❤️ for Healthcare AI & Open Science</sub>
</div>
