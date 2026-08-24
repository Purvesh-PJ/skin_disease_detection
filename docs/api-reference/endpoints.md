# 📡 REST API Reference & Endpoint Specifications

> **Complete developer specification for all REST API endpoints, request/response schemas, authentication headers, error envelopes, and executable cURL examples.**

---

## 🌐 API Overview & Base URL

* **Default Base URL (Local)**: `http://localhost:5000`
* **Production Base URL**: `https://<your-backend-domain>.onrender.com`
* **API Protocol**: REST over HTTPS / HTTP
* **Standard Payload Format**: `application/json`
* **File Upload Payload**: `multipart/form-data`

---

## 🔑 Authentication & Headers

Protected endpoints require a valid JSON Web Token (JWT) passed in the `Authorization` HTTP header:

```http
Authorization: Bearer <YOUR_JWT_ACCESS_TOKEN>
Content-Type: application/json
```

---

## 📋 Endpoint Summary Table

| Category | Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| **System** | `GET` | [`/health`](#1-system-health-check) | ❌ No | Server liveness & readiness check |
| **Auth** | `POST` | [`/auth/register`](#2-user-registration) | ❌ No | Create a new user account in MongoDB |
| **Auth** | `POST` | [`/auth/login`](#3-user-login) | ❌ No | Authenticate credentials and receive JWT |
| **Auth** | `POST` | [`/auth/demo-login`](#4-1-click-demo-login) | ❌ No | Instant 1-click recruiter demo authentication |
| **Auth** | `GET` | [`/auth/verify-token`](#5-verify-jwt-token) | ✅ Yes | Validate existing token session |
| **Auth** | `GET` | [`/auth/profile`](#6-get-user-profile) | ✅ Yes | Retrieve profile & settings data |
| **Auth** | `PUT` | [`/auth/profile`](#7-update-user-profile) | ✅ Yes | Update profile preferences & full name |
| **Inference** | `POST` | [`/predict`](#8-skin-lesion-prediction) | 🟡 Optional/Yes | Upload image and get ensemble prediction |
| **History** | `GET` | [`/history`](#9-fetch-scan-history) | ✅ Yes | Retrieve past scans for current user |
| **History** | `DELETE` | [`/history/{id}`](#10-delete-scan-record) | ✅ Yes | Delete a specific scan by ID |
| **History** | `DELETE` | [`/history/clear`](#11-clear-all-history) | ✅ Yes | Clear all scan history for current user |

---

## 🛠️ Endpoint Details & Examples

### 1. System Health Check
`GET /health` or `GET /`

* **Auth**: None
* **Response `200 OK`**:
```json
{
  "status": "healthy",
  "service": "Skin Disease Detection API",
  "version": "1.0.0"
}
```

---

### 2. User Registration
`POST /auth/register`

* **Auth**: None
* **Request Body**:
```json
{
  "username": "Dr. Aditi Verma",
  "email": "aditi.verma@hospital.org",
  "password": "SecurePassword123!"
}
```
* **Response `201 Created`**:
```json
{
  "message": "User registered successfully!",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "username": "Dr. Aditi Verma",
    "email": "aditi.verma@hospital.org",
    "roles": ["user"],
    "settings": { ... }
  }
}
```

---

### 3. User Login
`POST /auth/login`

* **Auth**: None
* **Request Body**:
```json
{
  "email": "aditi.verma@hospital.org",
  "password": "SecurePassword123!"
}
```
* **Response `200 OK`**:
```json
{
  "message": "Login successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": { ... }
}
```

---

### 4. 1-Click Demo Login
`POST /auth/demo-login`

* **Auth**: None
* **Response `200 OK`**:
```json
{
  "is_demo": true,
  "message": "Welcome Demo",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "username": "Demo",
    "email": "demo@skindisease.ai",
    "roles": ["demo", "user"]
  }
}
```

---

### 5. Skin Lesion Prediction
`POST /predict`

* **Auth**: Optional / Recommended Bearer Token
* **Content-Type**: `multipart/form-data`
* **Form Field**: `image` (binary file: JPG, PNG, JPEG)
* **cURL Example**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -F "image=@/path/to/skin_lesion_sample.jpg"
```
* **Response `200 OK`**:
```json
{
  "predicted_disease": "mel",
  "confidence": "96",
  "filename": "skin_lesion_sample.jpg",
  "message": "Image processed successfully",
  "history_id": "66c9898f7e8a9c3d12f34567",
  "saved_to_db": true,
  "disease_details": {
    "name": "Melanoma",
    "description": "A serious form of skin cancer that begins in cells known as melanocytes.",
    "risk": "High Risk",
    "recommendation": "Urgent dermatological biopsy and clinical evaluation recommended."
  }
}
```

---

### 6. Fetch Scan History
`GET /history`

* **Auth**: Bearer Token
* **Response `200 OK`**:
```json
{
  "status": "success",
  "count": 1,
  "history": [
    {
      "_id": "66c9898f7e8a9c3d12f34567",
      "user_email": "aditi.verma@hospital.org",
      "predicted_disease": "mel",
      "confidence": "96",
      "filename": "lesion_scan_01.jpg",
      "created_at": "2026-08-24T11:20:00.000000"
    }
  ]
}
```

---

## 🛑 Standard Error Envelope Format

```json
{
  "error": "Error description string",
  "status_code": 400
}
```

---

## 🔗 Cross References
* [Database & Schemas (`../architecture/data-models.md`)](../architecture/data-models.md)
* [Security & JWT Lifecycle (`../security/security-and-compliance.md`)](../security/security-and-compliance.md)
* [Troubleshooting (`../troubleshooting/diagnostic-playbook.md`)](../troubleshooting/diagnostic-playbook.md)
