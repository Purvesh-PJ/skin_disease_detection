# 🏛️ System Architecture & Engineering Design

> **Deep-dive technical specification of the multi-tier architecture, system context, component dependencies, and end-to-end data flows.**

---

## 📐 High-Level Architectural Overview

The Skin Disease Detection platform is structured as a **decoupled, multi-tier microservice-friendly architecture**. It enforces a strict separation of concerns among the user presentation layer, the application gateway, the deep learning inference engine, and persistent storage tiers.

```mermaid
graph TB
    subgraph PresentationTier ["1. Presentation Tier (Client)"]
        SPA["React 18 Single Page Application"]
        StateEngine["Context API (AuthContext, ThemeContext)"]
        LocalCache["Browser Storage (localStorage / sessionStorage)"]
        SPA --> StateEngine
        StateEngine --> LocalCache
        LocalCache --> StateEngine
    end

    subgraph GatewayTier ["2. Application Gateway Tier (Flask Backend)"]
        AppFactory["Application Factory (main.py)"]
        AuthMiddleware["JWT Auth Guard & Security Interceptor"]
        RouterHub["Versioned API Router (/api/v1/*)"]
        ErrorDomain["Centralized Error Handlers (exceptions.py)"]

        AppFactory --> RouterHub
        RouterHub --> AuthMiddleware
        AuthMiddleware --> ErrorDomain
    end

    subgraph InferenceTier ["3. AI & Deep Learning Inference Tier"]
        CloudLoader["Model Weight Downloader (cloud_downloader.py)"]
        ImagePreprocessor["Image Normalizer & Preprocessor (preprocess.py)"]
        EnsembleEngine["Soft-Voting Ensemble Engine (prediction_service.py)"]
        
        M_ResNet["ResNet101 Model (.h5)"]
        M_Dense["DenseNet121 Model (.h5)"]
        M_EffNet["EfficientNetB3 Model (.h5)"]

        ImagePreprocessor --> EnsembleEngine
        CloudLoader -.-> EnsembleEngine
        EnsembleEngine --> M_ResNet
        EnsembleEngine --> M_Dense
        EnsembleEngine --> M_EffNet
    end

    subgraph PersistenceTier ["4. Persistent Storage Tier"]
        MongoUsers[("MongoDB: users Collection")]
        MongoHistory[("MongoDB: prediction_history Collection")]
    end

    SPA -->|HTTP REST / multipart-form-data| RouterHub
    RouterHub -->|Forward Upload| ImagePreprocessor
    AuthMiddleware -->|User Data & Credential Ops| MongoUsers
    RouterHub -->|Save Scan History| MongoHistory
```

---

## 🧱 Architectural Layers Breakdown

### 1. Presentation Tier (Frontend Web Client)
* **Technology**: React 18, React Router v6, Styled Components, Axios.
* **Responsibilities**:
  * Provides intuitive clinical dermoscopy upload interfaces (drag-and-drop, sample lesion picker, camera input).
  * Manages global client state using React Context ([`AuthContext`](file:///d:/repos/skin_disease_detection/frontend/src/context/AuthContext.js), [`ThemeContext`](file:///d:/repos/skin_disease_detection/frontend/src/context/ThemeContext.js)).
  * Implements **Optimistic UI updates** and **Offline Fail-Safe Caching** through `localStorage` to guarantee 100% demo availability.
  * Formats diagnostic confidence scores, disease classifications, and clinical guidance data for medical evaluators.

### 2. Gateway & API Tier (Flask Application Layer)
* **Technology**: Python 3.12, Flask, Flask-JWT-Extended, Flask-Bcrypt, Flask-CORS.
* **Responsibilities**:
  * **Application Factory Pattern** ([`main.py`](file:///d:/repos/skin_disease_detection/backend/main.py)): Dynamically builds the Flask app with configured middleware, CORS origins, file size ceilings (16 MB), and error boundaries.
  * **Modular Route Blueprinting** ([`backend/app/api/router.py`](file:///d:/repos/skin_disease_detection/backend/app/api/router.py)): Isolates route definitions across `/auth`, `/predict`, `/history`, and `/health`.
  * **Domain Exception Interceptors** ([`backend/app/core/exceptions.py`](file:///d:/repos/skin_disease_detection/backend/app/core/exceptions.py)): Catches model failures, invalid image dimensions, and authorization rejections, serializing them into predictable JSON error responses.

### 3. Machine Learning Inference Tier (Computer Vision Engine)
* **Technology**: TensorFlow 2.x, Keras, OpenCV (`cv2`), Albumentations, NumPy.
* **Responsibilities**:
  * **Lazy Model Loading** ([`prediction_service.py`](file:///d:/repos/skin_disease_detection/backend/app/services/prediction_service.py)): Prevents slow application startup by loading TensorFlow `.h5` model graphs into memory only upon the first prediction request.
  * **Automated Cloud Weights Recovery** ([`cloud_downloader.py`](file:///d:/repos/skin_disease_detection/backend/app/ml/cloud_downloader.py)): Detects missing weight files in `trained_models/` and downloads required neural network checkpoints from remote cloud storage at runtime.
  * **Soft-Voting Ensemble Inference**: Computes class probability vectors across all three neural architectures simultaneously and calculates consensus probabilities.

### 4. Persistence Tier (Data Layer)
* **Technology**: MongoDB Atlas / Local MongoDB, PyMongo driver.
* **Responsibilities**:
  * **User Management** ([`backend/app/models/user.py`](file:///d:/repos/skin_disease_detection/backend/app/models/user.py)): Stores user identity records, salted Blowfish password hashes, roles, and profile preferences.
  * **Diagnostic Audit Log** ([`backend/app/models/history.py`](file:///d:/repos/skin_disease_detection/backend/app/models/history.py)): Stores timestamped records of predicted diseases, confidence metrics, and associated image filenames indexed by user email.

---

## 🔄 End-to-End Sequence & Data Flows

### A. Skin Lesion Prediction Lifecycle
The following sequence diagram outlines what happens when an authenticated user uploads a lesion image:

```mermaid
sequenceDiagram
    autonumber
    actor User as Doctor / Evaluator
    participant UI as React Frontend
    participant Gateway as Flask Gateway (/predict)
    participant Preproc as Image Preprocessor
    participant ML as Ensemble Engine
    participant DB as MongoDB Atlas

    User->>UI: Selects skin image & clicks "Scan Now"
    UI->>UI: Validates file type & renders local preview
    UI->>Gateway: POST /api/v1/predict (multipart/form-data + JWT)
    Gateway->>Gateway: Validates JWT token & saves temp image
    Gateway->>Preproc: Passes image file path
    Preproc->>Preproc: Converts BGR to RGB, resizes (224x224), normalizes
    Preproc->>ML: Sends normalized batch tensor
    par Parallel Inference
        ML->>ML: EfficientNetB3 prediction vector
        ML->>ML: ResNet101 prediction vector
        ML->>ML: DenseNet121 prediction vector
    end
    ML->>ML: Averages probability vectors (Soft-Voting)
    ML->>ML: Extracts top disease label & confidence %
    ML-->>Gateway: Returns prediction payload
    opt User is Authenticated
        Gateway->>DB: save_prediction(user_email, prediction_result)
        DB-->>Gateway: Returns inserted _id
    end
    Gateway->>Gateway: Deletes temporary uploaded image
    Gateway-->>UI: 200 OK (JSON with disease, confidence, details)
    UI->>UI: Updates local cache (skin_disease_scan_history)
    UI-->>User: Renders diagnostic card & confidence gauge
```

---

### B. Authentication & Fail-Safe Demo Workflow
The platform features a **Dual-Mode Authentication Architecture** supporting both real database accounts and instant zero-latency demo accounts:

```mermaid
sequenceDiagram
    autonumber
    actor User as Recruiter / Evaluator
    participant UI as React Client
    participant AuthAPI as Backend /auth/demo-login
    participant DB as MongoDB Atlas
    participant Local as Browser LocalStorage

    alt 1-Click Demo Login (Online Mode)
        User->>UI: Clicks "1-Click Demo Login"
        UI->>AuthAPI: POST /api/v1/auth/demo-login
        AuthAPI->>DB: seed_demo_user() (Auto-seeds demo user if missing)
        DB-->>AuthAPI: Demo user profile
        AuthAPI-->>UI: Returns JWT access token & user profile
        UI->>Local: Stores token & user in localStorage
        UI-->>User: Redirects to Dashboard
    else Backend Sleep / Cold-Start / Offline Mode (Fail-Safe)
        User->>UI: Clicks "1-Click Demo Login"
        UI->>AuthAPI: POST /api/v1/auth/demo-login (Timeouts or fails)
        AuthAPI--xUI: Network / Cold-Start Delay
        UI->>UI: Activates client-side fallback session
        UI->>Local: Generates demo-session-timestamp token & default profile
        UI-->>User: Instantly redirects to Dashboard with zero downtime
    end
```

---

## 🔗 Related Documentation
* [Frontend Architecture (`frontend-architecture.md`)](./frontend-architecture.md)
* [Database & Data Models (`data-models.md`)](./data-models.md)
* [Machine Learning Pipeline (`../machine-learning/ml-pipeline.md`)](../machine-learning/ml-pipeline.md)
* [REST API Reference (`../api-reference/endpoints.md`)](../api-reference/endpoints.md)
