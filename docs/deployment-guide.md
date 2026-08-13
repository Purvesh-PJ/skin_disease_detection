# 🚀 Deployment Guide - Skin Disease Detection Platform

This guide provides step-by-step instructions to deploy the **Skin Disease Detection Platform** to production using free/low-cost cloud services (MongoDB Atlas + Render + Vercel) or Docker.

---

## 📌 Summary of Environment Variables

All environment variables are declared in the single unified template file at the root: [`.env.example`](file:///d:/repos/skin_disease_detection/.env.example).

| Variable Name | Required? | Location | Description |
|---|---|---|---|
| `FLASK_SECRET_KEY` | Yes | Backend | Random secure string for Flask session encryption |
| `JWT_SECRET_KEY` | Yes | Backend | Random secure string for JWT token signing |
| `MONGO_URI` | Yes | Backend | MongoDB Connection String (Local or Cloud Atlas) |
| `MONGO_DB_NAME` | Optional | Backend | Database Name (Default: `skin_disease_db`) |
| `CORS_ORIGINS` | Yes | Backend | Allowed frontend domain URL(s) (e.g., `https://your-frontend.vercel.app`) |
| `MODEL_DIR` | Optional | Backend | Directory containing `.h5` model files (Default: `trained_models`) |
| `REACT_APP_API_URL` | Yes | Frontend | Public Backend API base URL (e.g., `https://your-backend.onrender.com`) |

---

## 🗄️ Step 1: Set Up Cloud Database (MongoDB Atlas)

1. Sign up for a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Create a new **M0 Free Cluster**.
3. Under **Database Access**, create a database user (username & password).
4. Under **Network Access**, click **Add IP Address** and select `0.0.0.0/0` (Allow Access from Anywhere for Cloud Deployments).
5. Click **Connect** → **Drivers** to get your connection string. It will look like:
   ```text
   mongodb+srv://<username>:<password>@cluster0.mongodb.net/skin_disease_db?retryWrites=true&w=majority
   ```
6. Replace `<username>` and `<password>` with your database user credentials. This is your `MONGO_URI`.

---

## 🐍 Step 2: Deploy Backend API (Render.com)

1. Push your repository code to GitHub.
2. Sign in to [Render.com](https://render.com).
3. Click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Configure the service:
   - **Name**: `skin-disease-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
6. Scroll down to **Environment Variables** and add the following keys:
   - `FLASK_SECRET_KEY` = `<your_secret_key>`
   - `JWT_SECRET_KEY` = `<your_jwt_secret>`
   - `MONGO_URI` = `<your_mongodb_atlas_uri>`
   - `MONGO_DB_NAME` = `skin_disease_db`
   - `CORS_ORIGINS` = `*` (or your frontend Vercel URL once deployed)
   - `MODEL_DIR` = `trained_models`
7. Ensure your pre-trained `.h5` files are present in the `trained_models/` directory or downloaded onto server storage.
8. Click **Create Web Service**. Render will build and deploy your backend, giving you a URL like:
   `https://skin-disease-backend.onrender.com`

---

## ⚛️ Step 3: Deploy Frontend (Vercel)

1. Sign in to [Vercel](https://vercel.com).
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. Set the **Root Directory** to `frontend`.
5. Expand **Environment Variables** and add:
   - **Key**: `REACT_APP_API_URL`
   - **Value**: `https://skin-disease-backend.onrender.com` (Your Render Backend URL from Step 2)
6. Click **Deploy**. Vercel will build your React application and provide a live production URL (e.g., `https://skin-disease-detection.vercel.app`).
7. Update the `CORS_ORIGINS` variable in your Render backend settings to match your live Vercel URL:
   - `CORS_ORIGINS` = `https://skin-disease-detection.vercel.app`

---

## 🐳 Alternative: Single-Command Docker Deployment (VPS or Local)

If you are deploying to an AWS EC2, DigitalOcean Droplet, or running locally with Docker:

1. Ensure Docker and Docker Compose are installed.
2. Ensure trained models are placed in the `trained_models/` directory.
3. Run the following command in the project root:
   ```bash
   docker-compose up -d --build
   ```
4. Access services:
   - **Frontend**: `http://localhost:3000`
   - **Backend API**: `http://localhost:5000`
   - **MongoDB**: `localhost:27017`

---

## 📋 Hinglish Summary (Quick Reference)

- **`.env` files ko Git me push kyu nahi karte?**
  Kyuki `.env` me passwords aur secret keys hoti hain. GitHub par secret keys push hone se security breach ho sakta hai. Isliye `.env` `.gitignore` me rehta hai aur cloud dashboard me environment variables manually enter kiye jaate hain.

- **Sample `.env` file kahan se lein?**
  Aap root directory me maujood [`.env.example`](file:///d:/repos/skin_disease_detection/.env.example) file ko copy karke `.env` naam se save kar sakte hain:
  ```bash
  cp .env.example .env
  ```
