# 🔐 Environment Variables & Secrets Reference

> **Centralized specification for all configuration variables across Frontend, Backend, Database, and Machine Learning layers.**

---

## 📋 Comprehensive Environment Matrix

| Variable Name | Layer | Required | Default Value | Description |
| :--- | :--- | :---: | :--- | :--- |
| `REACT_APP_API_URL` | Frontend | Yes | `http://localhost:5000` | Target backend REST API URL. |
| `FLASK_SECRET_KEY` | Backend | Yes | *(Random hex in prod)* | Cryptographic secret for signing session cookies. |
| `JWT_SECRET_KEY` | Backend | Yes | *(Random hex in prod)* | Secret key for signing and verifying JWT access tokens. |
| `MONGO_DB_URI` | Backend | Yes | `mongodb://localhost:27017/skin_disease_db` | MongoDB Atlas / Local connection string. |
| `MONGO_DB_NAME` | Backend | No | `skin_disease_db` | Target MongoDB database name. |
| `CORS_ORIGINS` | Backend | Yes | `*` (Local) / `https://app.vercel.app` (Prod) | Comma-separated allowed HTTP origins. |
| `MODEL_DIR` | Backend | No | `trained_models` | Directory path containing `.h5` model files. |
| `MODEL_ZIP_URL` | Backend | No | Direct Google Drive URL | Direct URL for automated cloud weight downloading. |
| `FLASK_HOST` | Backend | No | `0.0.0.0` | Host IP address for local Flask development. |
| `FLASK_PORT` | Backend | No | `5000` | Port for local Flask server. |
| `FLASK_DEBUG` | Backend | No | `True` (Local) / `False` (Prod) | Debug mode trigger. |

---

## 📁 Sample Environment Files

### Root `.env` Template
```env
# Production & Security Secrets
FLASK_SECRET_KEY=prod_flask_secret_key_change_me
JWT_SECRET_KEY=prod_jwt_secret_key_change_me

# MongoDB Atlas Connection
MONGO_DB_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/skin_disease_db?retryWrites=true&w=majority
MONGO_DB_NAME=skin_disease_db

# CORS Origin
CORS_ORIGINS=https://your-frontend.vercel.app

# Cloud Model Downloader
MODEL_ZIP_URL=https://drive.google.com/uc?export=download&id=1kzUcTDnaL3297BWEJfnrTtvQhejAeEHa
MODEL_DIR=trained_models
```

---

## 🔗 Cross References
* [Production Deployment (`deployment-guide.md`)](./deployment-guide.md)
* [Docker & Containerization (`docker-and-containerization.md`)](./docker-and-containerization.md)
* [Security & Compliance (`../security/security-and-compliance.md`)](../security/security-and-compliance.md)
