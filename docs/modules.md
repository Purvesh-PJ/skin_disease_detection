# Modules & Components Documentation

This document provides a detailed breakdown of all major modules, folders, and components in the project.

---

## Backend Modules

### 📁 `backend/main.py`
**Purpose:** Application entry point and Flask app factory

**Key Functions:**
- `create_app(testing=False)` - Application factory pattern
- `configure_app(app, testing)` - Load configuration from environment
- `setup_cors(app)` - Configure cross-origin resource sharing
- `setup_jwt(app)` - Initialize JWT manager with error handlers
- `register_blueprints(app)` - Register all route blueprints
- `register_error_handlers(app)` - Global error handling
- `register_request_handlers(app)` - Request/response logging

**Configuration:**
- JWT expiration: 1 hour (access), 30 days (refresh)
- Max upload size: 16MB
- Upload folder: `backend/uploads/`
- CORS origins: Configurable via environment

**Why it exists:** Centralizes app initialization, makes testing easier, follows Flask best practices

---

### 📁 `backend/app/routes/`

#### `auth_routes.py`
**Purpose:** Authentication endpoints

**Routes:**
- `POST /auth/register` - Create new user account
- `POST /auth/login` - Authenticate and get JWT token
- `GET /auth/verify-token` - Validate existing token

**Key Functions:**
- `login()` - Verify credentials, generate JWT
- `register()` - Validate email, hash password, create user
- `verify_token()` - Decode and validate JWT
- `token_required` - Custom decorator for protected routes
- `is_email_taken()` - Check email uniqueness

**Dependencies:**
- Flask-Bcrypt for password hashing
- PyJWT for token generation
- MongoDB for user storage

**Security Features:**
- Bcrypt password hashing with salt
- JWT with HS256 algorithm
- Token expiration handling
- Email uniqueness validation

---

#### `prediction_routes.py`
**Purpose:** ML prediction endpoint

**Routes:**
- `POST /predict` - Upload image and get disease prediction

**Key Functions:**
- `prediction()` - Main prediction handler
- `load_and_preprocess_image()` - Image preprocessing pipeline
- `setup_prediction_routes(app)` - Register routes with app

**Model Loading:**
```python
models = {
    "resnet": tf.keras.models.load_model("trained_models/resnet101.h5"),
    "densenet": tf.keras.models.load_model("trained_models/densenet121.h5"),
    "efficientnet": tf.keras.models.load_model("trained_models/efficientnetb3.h5")
}
```

**Preprocessing Pipeline:**
1. Read image with OpenCV
2. Convert BGR → RGB
3. Resize to 224×224
4. Apply EfficientNet preprocessing
5. Expand dimensions for batch

**Ensemble Logic:**
1. Run inference on all 3 models
2. Average softmax probabilities
3. Select class with highest average
4. Return max confidence from individual models

**Disease Mapping:**
- Maps class indices to disease codes
- Provides user-friendly names and descriptions
- 7 classes: akiec, bcc, bkl, df, mel, nv, vasc

**Why it exists:** Core ML functionality, handles entire prediction pipeline

---

#### `home_routes.py`
**Purpose:** Health check and basic endpoints

**Routes:**
- `GET /` - API status check
- Other utility endpoints

**Why it exists:** Monitoring, debugging, API availability checks

---

### 📁 `backend/app/ai_models/`

#### `densenet121_model.py`
**Purpose:** DenseNet121 model architecture and training

**Key Functions:**
- `create_densenet121_model()` - Build model architecture
- `train_densenet121_model()` - Training pipeline with callbacks
- `evaluate_densenet121_model()` - Evaluation metrics

**Architecture:**
- Base: DenseNet121 (ImageNet weights)
- Trainable: Last 70 layers unfrozen
- Custom head:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(512) + LeakyReLU + Dropout(0.5)
  - BatchNormalization
  - Dense(256) + LeakyReLU + Dropout(0.5)
  - Dense(7, softmax)

**Training Configuration:**
- Optimizer: Adam (lr=1e-4)
- Loss: Sparse categorical crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Regularization: L2 (1e-4), Dropout (0.5)

**Why it exists:** Dense connections enable feature reuse, good for medical imaging

---

#### `efficientnetB3_model.py`
**Purpose:** EfficientNetB3 model architecture and training

**Similar structure to DenseNet121:**
- Transfer learning from ImageNet
- Custom classification head
- Training and evaluation functions

**Why it exists:** Efficient scaling, compound coefficient optimization, state-of-the-art accuracy

---

#### `resnet101_model.py`
**Purpose:** ResNet101 model architecture and training

**Similar structure to DenseNet121:**
- Deep residual learning
- Skip connections
- Custom classification head

**Why it exists:** Very deep network (101 layers), skip connections prevent vanishing gradients

---

### 📁 `backend/app/config/`

#### `mongo_config.py`
**Purpose:** MongoDB connection configuration

**Key Components:**
```python
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/skin_disease_db")
client = MongoClient(MONGO_URI)
db = client.get_database("skin_disease_db")
users_collection = db.get_collection("users")
```

**Why it exists:** Centralized database configuration, easy to switch environments

---

### 📁 `backend/app/db_models/`

#### `user_model.py`
**Purpose:** User data access layer (Repository pattern)

**Key Functions:**
- `create_user(username, email, password)` - Insert new user
- `find_user_by_email(email)` - Query user by email
- `is_email_taken(email)` - Check email existence

**User Schema:**
```python
{
    "username": str,
    "email": str,
    "password": str,  # bcrypt hashed
    "roles": ["user"]  # default role
}
```

**Why it exists:** Abstracts database operations, makes code testable, follows repository pattern

---

### 📁 `backend/app/services/`

#### `training_ensemble_model_service.py`
**Purpose:** Ensemble model training orchestration

**Key Functions:**
- `train_and_evaluate_ensemble()` - Coordinate training of all models

**Note:** This appears to be for offline training, not used in production API

**Why it exists:** Orchestrates training of multiple models, ensemble evaluation

---

### 📁 `backend/app/utils/`

#### `preprocess.py`
**Purpose:** Image preprocessing utilities

**Key Functions:**
- Image loading
- Resizing
- Normalization
- Data augmentation (for training)

**Why it exists:** Reusable preprocessing logic, ensures consistency between training and inference

---

### 📁 `backend/app/middleware/`

#### `auth_middleware.py`
**Purpose:** JWT authentication middleware

**Key Functions:**
- Token extraction from headers
- Token validation
- User context injection

**Why it exists:** Centralized authentication logic, reusable across routes

---

### 📁 `backend/app/notebooks/`

#### `model_training.ipynb`
**Purpose:** Interactive model training notebook

**Contents:**
- Data loading and exploration
- Model training experiments
- Hyperparameter tuning
- Evaluation and visualization

**Why it exists:** Experimentation, visualization, documentation of training process

---

#### `dataset_inspection.ipynb`
**Purpose:** Dataset analysis and visualization

**Contents:**
- Class distribution analysis
- Image visualization
- Data quality checks
- Augmentation previews

**Why it exists:** Understanding dataset characteristics, identifying imbalances

---

### 📁 `backend/data/skin_disease_dataset/`

**Structure:**
```
base_dir/
├── train_dir/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── val_dir/
│   └── (same structure)
└── test_dir/
    └── (same structure)
```

**Purpose:** Organized dataset for training/validation/testing

**Why it exists:** Standard directory structure for Keras ImageDataGenerator

---

## Frontend Modules

### 📁 `frontend/src/`

#### `App.js`
**Purpose:** Root application component

**Key Features:**
- Authentication state management
- Token verification on mount
- Loading state during verification
- Route rendering based on auth status

**Flow:**
1. Check if token exists in localStorage
2. Verify token with backend
3. Set authentication state
4. Render appropriate routes

**Why it exists:** Entry point, manages global auth state

---

#### `index.js`
**Purpose:** React application bootstrap

**Key Features:**
- Render React app to DOM
- Wrap with BrowserRouter
- Apply global styles
- Theme provider setup

**Why it exists:** Standard React entry point

---

### 📁 `frontend/src/routes/`

#### `index.js`
**Purpose:** Application routing configuration

**Routes:**
- `/` - Redirect based on auth status
- `/login` - Login page (public)
- `/signup` - Registration page (public)
- `/dashboard` - Main app (protected)
- `*` - 404 Not Found

**Route Protection:**
- Public routes redirect to dashboard if authenticated
- Protected routes redirect to login if not authenticated

**Why it exists:** Centralized routing, navigation logic

---

#### `ProtectedRoute.js`
**Purpose:** Route protection wrapper

**Logic:**
```javascript
if (isAuthenticated) {
  return children;
} else {
  return <Navigate to="/login" />;
}
```

**Why it exists:** Reusable route protection, prevents unauthorized access

---

### 📁 `frontend/src/pages/`

#### `Login/`
**Purpose:** User login page

**Features:**
- Email and password inputs
- Form validation
- Error message display
- Loading state during login
- Link to signup page

**Flow:**
1. User enters credentials
2. Submit form
3. Call authService.login()
4. Store token on success
5. Redirect to dashboard

**Why it exists:** User authentication entry point

---

#### `Signup/`
**Purpose:** User registration page

**Features:**
- Username, email, password inputs
- Form validation
- Password confirmation
- Error message display
- Link to login page

**Flow:**
1. User fills registration form
2. Validate inputs
3. Call authService.register()
4. Show success message
5. Redirect to login

**Why it exists:** New user onboarding

---

#### `Dashboard/`
**Purpose:** Main application interface

**Layout:**
```
┌─────────────────────────────────────┐
│           Header                    │
├──────────────────┬──────────────────┤
│   Left Panel     │   Right Panel    │
│                  │                  │
│  Upload Image    │  Analysis Results│
│                  │                  │
│  [Image Preview] │  [Disease Info]  │
│  [Analyze Btn]   │  [Confidence]    │
│                  │  [Description]   │
└──────────────────┴──────────────────┘
```

**State Management:**
- selectedImage (preview URL)
- imageFile (File object)
- predictionResult (API response)
- loading (boolean)
- error (error object)

**Why it exists:** Main user interface, prediction workflow

---

#### `NotFound/`
**Purpose:** 404 error page

**Features:**
- Friendly error message
- Link back to home/dashboard

**Why it exists:** Handle invalid routes gracefully

---

### 📁 `frontend/src/components/`

#### `components/layout/Header.js`
**Purpose:** Application header/navbar

**Features:**
- App logo/title
- User info display
- Logout button
- Theme toggle (if implemented)

**Why it exists:** Consistent navigation, user actions

---

#### `components/features/prediction/ImageUploadCard.js`
**Purpose:** Image upload interface

**Features:**
- Drag-and-drop zone
- Click to upload
- Image preview
- File validation
- Analyze button
- Clear button
- Warning message

**Validation:**
- File type: image/*
- File size: max 10MB
- Visual feedback on valid/invalid

**Why it exists:** User-friendly image upload, core feature interface

---

#### `components/features/prediction/ResultsCard.js`
**Purpose:** Display prediction results

**Features:**
- Disease name
- Confidence percentage
- Disease description
- Visual indicators (color-coded)
- Loading state
- Error state
- Empty state

**Display Logic:**
- Loading: Show spinner
- Error: Show error message
- Success: Show results with formatting
- Empty: Show placeholder

**Why it exists:** Present ML predictions in user-friendly format

---

#### `components/common/ui/`
**Purpose:** Reusable UI components

**Components:**
- Button (variants: primary, secondary, sizes)
- Spinner (loading indicator)
- Input (form inputs)
- Card (container component)
- Text (typography components)

**Why it exists:** Consistent UI, reusability, maintainability

---

### 📁 `frontend/src/services/`

#### `services/api/axios.js`
**Purpose:** Axios instance configuration

**Configuration:**
- Base URL: Backend API endpoint
- Default headers
- Request interceptor (add JWT token)
- Response interceptor (handle errors)

**Interceptors:**
```javascript
// Request: Add token to all requests
config.headers.Authorization = `Bearer ${token}`;

// Response: Handle 401 errors
if (error.response.status === 401) {
  authService.logout();
}
```

**Why it exists:** Centralized HTTP client, automatic token injection

---

#### `services/api/auth.service.js`
**Purpose:** Authentication API calls

**Methods:**
- `login(email, password)` - Authenticate user
- `register(userData)` - Create account
- `verifyToken()` - Validate token
- `logout()` - Clear session
- `getToken()` - Retrieve stored token
- `getUser()` - Retrieve user info
- `isAuthenticated()` - Check auth status

**Why it exists:** Encapsulates auth logic, reusable across components

---

#### `services/api/prediction.service.js`
**Purpose:** Prediction API calls

**Methods:**
- `predict(imageData)` - Upload image and get prediction

**Configuration:**
- Content-Type: multipart/form-data
- Authorization: JWT token (via interceptor)

**Why it exists:** Encapsulates prediction logic, handles file uploads

---

### 📁 `frontend/src/hooks/`

#### `usePrediction.js`
**Purpose:** Custom hook for prediction logic

**Returns:**
- `predict(imageData)` - Function to make prediction
- `loading` - Loading state
- `error` - Error state
- `reset()` - Reset error state

**Why it exists:** Reusable prediction logic, state management

---

### 📁 `frontend/src/context/`

#### `ThemeContext.js`
**Purpose:** Theme management (dark/light mode)

**Features:**
- Theme state
- Toggle function
- Theme persistence

**Why it exists:** Global theme state, consistent styling

---

### 📁 `frontend/src/styles/`

**Purpose:** Global styles and theme configuration

**Files:**
- `typography.js` - Text components
- `theme.js` - Color palette, spacing, etc.
- Global CSS reset

**Why it exists:** Consistent styling, theme management

---

### 📁 `frontend/src/constants/`

**Purpose:** Application constants

**Constants:**
- `ROUTES` - Route paths
- `API_ENDPOINTS` - API URLs
- `STORAGE_KEYS` - localStorage keys
- Disease information mappings

**Why it exists:** Single source of truth, easy to update

---

### 📁 `frontend/src/config/`

**Purpose:** Configuration files

**Config:**
- API base URL
- Environment-specific settings

**Why it exists:** Environment-based configuration

---

## Module Dependencies

### Backend Dependencies Flow
```
main.py
  ├── routes/auth_routes.py
  │     ├── db_models/user_model.py
  │     │     └── config/mongo_config.py
  │     └── flask_bcrypt, flask_jwt_extended
  ├── routes/prediction_routes.py
  │     ├── ai_models/*.py
  │     ├── utils/preprocess.py
  │     └── tensorflow, opencv
  └── middleware/auth_middleware.py
```

### Frontend Dependencies Flow
```
App.js
  ├── routes/index.js
  │     ├── pages/*
  │     └── ProtectedRoute.js
  ├── services/api/*
  │     ├── axios.js
  │     ├── auth.service.js
  │     └── prediction.service.js
  └── components/*
        ├── layout/*
        ├── features/*
        └── common/ui/*
```

---

## Key Entry Points

### Backend
1. **Application Start:** `backend/main.py`
2. **Authentication:** `backend/app/routes/auth_routes.py`
3. **Prediction:** `backend/app/routes/prediction_routes.py`
4. **Database:** `backend/app/config/mongo_config.py`

### Frontend
1. **Application Start:** `frontend/src/index.js`
2. **Root Component:** `frontend/src/App.js`
3. **Routing:** `frontend/src/routes/index.js`
4. **API Client:** `frontend/src/services/api/axios.js`

---

## Configuration Points

### Backend Configuration
- **Environment:** `.env` file
- **Flask Config:** `main.py::configure_app()`
- **Database:** `config/mongo_config.py`
- **Models:** `routes/prediction_routes.py` (model paths)

### Frontend Configuration
- **API URL:** `config/` or environment variables
- **Routes:** `constants/ROUTES`
- **Theme:** `styles/theme.js`
- **Storage Keys:** `constants/STORAGE_KEYS`

---

## Utility Areas

### Backend Utilities
- **Preprocessing:** `app/utils/preprocess.py`
- **Model Training:** `app/ai_models/*.py`
- **Notebooks:** `app/notebooks/*.ipynb`

### Frontend Utilities
- **Hooks:** `hooks/usePrediction.js`
- **Services:** `services/api/*`
- **Constants:** `constants/*`
- **Styles:** `styles/*`
