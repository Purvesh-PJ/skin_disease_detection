# Architecture Overview

## System Type

This is a **full-stack ML-powered web application** for skin disease detection using deep learning ensemble models. The system combines three CNN architectures (EfficientNetB3, ResNet101, DenseNet121) to classify dermatoscopic images into 7 skin disease categories.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (React 18 Frontend)                          │
│  - Image Upload UI                                              │
│  - Authentication Forms                                         │
│  - Results Display                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/REST API
                         │ (JWT Auth)
┌────────────────────────▼────────────────────────────────────────┐
│                      BACKEND API LAYER                          │
│                    (Flask + Python 3.12)                        │
│  - Authentication Routes (/auth/*)                              │
│  - Prediction Routes (/predict)                                 │
│  - JWT Middleware                                               │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           │                              │
┌──────────▼──────────┐        ┌─────────▼──────────────────────┐
│   DATABASE LAYER    │        │    ML INFERENCE LAYER          │
│     (MongoDB)       │        │  (TensorFlow/Keras Models)     │
│                     │        │                                │
│  - User Collection  │        │  ┌──────────────────────────┐  │
│  - Auth Data        │        │  │  EfficientNetB3 Model    │  │
│  - Roles            │        │  └──────────────────────────┘  │
└─────────────────────┘        │  ┌──────────────────────────┐  │
                               │  │  ResNet101 Model         │  │
                               │  └──────────────────────────┘  │
                               │  ┌──────────────────────────┐  │
                               │  │  DenseNet121 Model       │  │
                               │  └──────────────────────────┘  │
                               │                                │
                               │  Ensemble Averaging Logic      │
                               └────────────────────────────────┘
```

---

## Major Layers & Components

### 1. Frontend Layer (React 18)
**Location:** `frontend/src/`

**Purpose:** Provides user interface for authentication and skin disease prediction

**Key Components:**
- **Pages:** Login, Signup, Dashboard, NotFound
- **Features:** ImageUploadCard (drag-drop), ResultsCard (display predictions)
- **Services:** API communication via Axios
- **Routing:** React Router DOM with protected routes
- **State:** Local component state + localStorage for auth tokens
- **Styling:** Styled Components with theme support

**Tech Stack:**
- React 18.3.1
- React Router DOM 6.27.0
- Axios 1.7.7
- Styled Components 6.1.13
- React Icons 5.5.0

---

### 2. Backend API Layer (Flask)
**Location:** `backend/`

**Purpose:** REST API server handling authentication, file uploads, and ML predictions

**Key Modules:**
- **main.py:** Application factory, configuration, CORS, JWT setup
- **routes/auth_routes.py:** Login, register, token verification
- **routes/prediction_routes.py:** Image upload, preprocessing, ensemble prediction
- **routes/home_routes.py:** Health check endpoints
- **middleware/auth_middleware.py:** JWT token validation
- **config/mongo_config.py:** MongoDB connection setup

**Tech Stack:**
- Python 3.12
- Flask (web framework)
- Flask-JWT-Extended (authentication)
- Flask-CORS (cross-origin requests)
- Flask-Bcrypt (password hashing)
- PyMongo (MongoDB driver)

---

### 3. ML Inference Layer
**Location:** `backend/app/ai_models/`, `trained_models/`

**Purpose:** Load pre-trained models and perform ensemble predictions

**Model Architecture:**
Each model follows this structure:
1. Pre-trained base (ImageNet weights)
2. Global Average Pooling
3. Batch Normalization
4. Dense layers with LeakyReLU activation
5. Dropout for regularization
6. Softmax output (7 classes)

**Models:**
- **EfficientNetB3:** Efficient scaling, compound coefficient
- **ResNet101:** Deep residual learning, skip connections
- **DenseNet121:** Dense connections, feature reuse

**Ensemble Strategy:**
- Each model outputs softmax probabilities (7 classes)
- Average probabilities across all 3 models
- Select class with highest averaged probability
- Return max confidence from individual models

**Classes (7 skin diseases):**
1. `akiec` - Actinic Keratoses
2. `bcc` - Basal Cell Carcinoma
3. `bkl` - Benign Keratosis
4. `df` - Dermatofibroma
5. `mel` - Melanoma
6. `nv` - Melanocytic Nevus
7. `vasc` - Vascular Lesion

---

### 4. Database Layer (MongoDB)
**Location:** MongoDB instance (local or cloud)

**Purpose:** Store user authentication data

**Collections:**
- **users:** User accounts with hashed passwords, roles

**Schema (inferred):**
```javascript
{
  _id: ObjectId,
  username: String,
  email: String (unique),
  password: String (bcrypt hashed),
  roles: Array<String> (default: ["user"])
}
```

---

### 5. Data Processing Layer
**Location:** `backend/app/utils/preprocess.py`

**Purpose:** Image preprocessing pipeline for model input

**Pipeline:**
1. Read image from disk (OpenCV)
2. Convert BGR → RGB
3. Resize to 224×224
4. Apply EfficientNet preprocessing (normalization)
5. Expand dimensions for batch input

---

## Architecture Patterns

### Application Factory Pattern
The Flask app uses a factory function (`create_app()`) for:
- Environment-based configuration
- Testing support
- Modular initialization

### Blueprint Pattern
Routes are organized as Flask Blueprints:
- `auth_blueprint` for authentication
- Separate route setup functions for predictions

### Service Layer Pattern
Business logic separated from routes:
- `authService` (frontend)
- `predictionService` (frontend)
- Model loading and inference logic (backend)

### Repository Pattern
Database operations abstracted in `user_model.py`:
- `create_user()`
- `find_user_by_email()`
- `is_email_taken()`

---

## Security Architecture

### Authentication Flow
1. User registers → password hashed with bcrypt
2. User logs in → JWT token generated
3. Token stored in localStorage (frontend)
4. Protected routes require valid JWT in Authorization header
5. Backend validates token on each protected request

### Security Measures
- Bcrypt password hashing
- JWT with expiration (1 hour access, 30 days refresh)
- CORS configuration
- File upload size limits (16MB)
- Secure filename handling
- Environment variable configuration

---

## Deployment Architecture (Inferred)

**Development:**
- Frontend: `npm start` (port 3000)
- Backend: `python main.py` (port 5000)
- MongoDB: Local instance (port 27017)

**Production (Recommended):**
- Frontend: Static build served via Nginx/CDN
- Backend: Gunicorn/uWSGI behind reverse proxy
- MongoDB: Cloud instance (MongoDB Atlas)
- Models: Loaded once at startup, kept in memory

---

## Data Flow Summary

### Registration Flow
```
User Form → Frontend Validation → POST /auth/register → 
Hash Password → Store in MongoDB → Success Response
```

### Login Flow
```
User Credentials → POST /auth/login → Verify Password → 
Generate JWT → Return Token → Store in localStorage
```

### Prediction Flow
```
Upload Image → FormData → POST /predict (with JWT) → 
Save to uploads/ → Preprocess Image → 
Load 3 Models → Run Inference → Average Predictions → 
Map to Disease Info → Return JSON Response → Display Results
```

---

## Scalability Considerations

**Current Limitations:**
- Models loaded in memory (high RAM usage)
- Synchronous prediction (blocks during inference)
- Single-threaded Flask server
- File uploads stored locally

**Potential Improvements:**
- Model serving with TensorFlow Serving or TorchServe
- Async prediction queue (Celery + Redis)
- Horizontal scaling with load balancer
- Cloud storage for uploads (S3, GCS)
- Model caching and optimization (TensorRT, ONNX)

---

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend Framework | React | 18.3.1 |
| Frontend Routing | React Router DOM | 6.27.0 |
| HTTP Client | Axios | 1.7.7 |
| Styling | Styled Components | 6.1.13 |
| Backend Framework | Flask | Latest |
| Authentication | Flask-JWT-Extended | Latest |
| Database | MongoDB | Latest |
| ML Framework | TensorFlow/Keras | Latest |
| Image Processing | OpenCV | Latest |
| Password Hashing | Bcrypt | Latest |
| Language (Backend) | Python | 3.12 |
| Language (Frontend) | JavaScript (ES6+) | - |

---

## Assumptions & Clarifications

**Confirmed from Code:**
- Full-stack architecture with separate frontend/backend
- JWT-based authentication
- Ensemble of 3 CNN models
- MongoDB for user storage
- REST API communication

**Inferred/Assumed:**
- Production deployment strategy
- Scalability patterns
- Model training happens offline (notebooks exist but not integrated)
- Models are pre-trained and loaded at startup

**Missing/Unclear:**
- Actual model training integration with backend
- User history/prediction logging
- Admin panel or role-based features
- API rate limiting
- Monitoring/logging infrastructure
