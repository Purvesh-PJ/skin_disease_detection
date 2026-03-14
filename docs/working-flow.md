# Working Flow Documentation

This document explains the end-to-end flows in the Skin Disease Detection system, from user actions to final results.

---

## 1. User Registration Flow

### Step-by-Step Process

```
User fills registration form
    ↓
Frontend validates input (email format, password strength)
    ↓
POST /auth/register
    {
      "username": "john_doe",
      "email": "john@example.com",
      "password": "SecurePass123"
    }
    ↓
Backend receives request
    ↓
Check if email already exists in MongoDB
    ↓
If email exists → Return 409 Conflict
    ↓
If email available:
    - Hash password using bcrypt
    - Create user document with default role ["user"]
    - Insert into MongoDB users collection
    ↓
Return 201 Created
    {
      "message": "User registered successfully!"
    }
    ↓
Frontend displays success message
    ↓
Redirect to login page
```

### Technical Details

**Frontend:**
- Component: `frontend/src/pages/Signup/`
- Service: `authService.register(userData)`
- Validation: Email format, password requirements

**Backend:**
- Route: `POST /auth/register`
- Handler: `auth_routes.py::register()`
- Database: `user_model.py::create_user()`
- Password: Bcrypt hashing with salt

**Database:**
```javascript
// MongoDB Document Created
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "email": "john@example.com",
  "password": "$2b$12$...", // bcrypt hash
  "roles": ["user"]
}
```

---

## 2. User Login Flow

### Step-by-Step Process

```
User enters email and password
    ↓
POST /auth/login
    {
      "email": "john@example.com",
      "password": "SecurePass123"
    }
    ↓
Backend finds user by email in MongoDB
    ↓
If user not found → Return 401 Unauthorized
    ↓
If user found:
    - Compare password with bcrypt hash
    ↓
If password incorrect → Return 401 Unauthorized
    ↓
If password correct:
    - Generate JWT token with payload:
      {
        "email": "john@example.com",
        "username": "john_doe",
        "roles": ["user"]
      }
    - Sign with JWT_SECRET_KEY
    ↓
Return 200 OK
    {
      "token": "eyJhbGciOiJIUzI1NiIs...",
      "message": "Login successful"
    }
    ↓
Frontend stores token in localStorage
    ↓
Frontend stores user info in localStorage
    ↓
Redirect to Dashboard
```

### Technical Details

**Frontend:**
- Component: `frontend/src/pages/Login/`
- Service: `authService.login(email, password)`
- Storage: `localStorage.setItem('token', token)`

**Backend:**
- Route: `POST /auth/login`
- Handler: `auth_routes.py::login()`
- Token: JWT with HS256 algorithm
- Expiry: 1 hour (access token)

**Token Structure:**
```javascript
// JWT Payload
{
  "email": "john@example.com",
  "username": "john_doe",
  "roles": ["user"],
  "exp": 1234567890 // expiration timestamp
}
```

---

## 3. Token Verification Flow

### Step-by-Step Process

```
App loads / Page refresh
    ↓
Check if token exists in localStorage
    ↓
If no token → Redirect to login
    ↓
If token exists:
    GET /auth/verify-token
    Headers: {
      "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."
    }
    ↓
Backend extracts token from Authorization header
    ↓
Decode and verify JWT signature
    ↓
If token expired → Return 401 "Token expired"
    ↓
If token invalid → Return 401 "Invalid token"
    ↓
If token valid:
    - Extract user payload
    - Return 200 OK with user info
    ↓
Frontend updates user state
    ↓
Allow access to protected routes
```

### Technical Details

**Frontend:**
- Trigger: `App.js` useEffect on mount
- Service: `authService.verifyToken()`
- Error Handling: Logout and redirect on failure

**Backend:**
- Route: `GET /auth/verify-token`
- Decorator: `@token_required`
- Validation: JWT decode with secret key

---

## 4. Skin Disease Prediction Flow (Main Feature)

### Complete End-to-End Flow

```
User navigates to Dashboard
    ↓
User selects/drags image file
    ↓
Frontend displays image preview
    ↓
User clicks "Analyze Image" button
    ↓
Create FormData with image file
    ↓
POST /predict
    Headers: {
      "Authorization": "Bearer <token>",
      "Content-Type": "multipart/form-data"
    }
    Body: FormData with 'image' field
    ↓
Backend validates JWT token
    ↓
If token invalid → Return 401 Unauthorized
    ↓
If token valid:
    - Extract image from request.files
    - Validate file exists and has filename
    - Generate secure filename
    - Save to uploads/ directory
    ↓
Image Preprocessing:
    - Read image with OpenCV (cv2.imread)
    - Convert BGR → RGB color space
    - Resize to 224×224 pixels
    - Apply EfficientNet preprocessing (normalization)
    - Expand dimensions to (1, 224, 224, 3)
    ↓
Model Inference (Parallel):
    ┌─────────────────────────────────────┐
    │  Load EfficientNetB3 Model          │
    │  → Predict probabilities [7 classes]│
    │  → Output: [0.05, 0.02, 0.85, ...]  │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │  Load ResNet101 Model               │
    │  → Predict probabilities [7 classes]│
    │  → Output: [0.03, 0.01, 0.90, ...]  │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │  Load DenseNet121 Model             │
    │  → Predict probabilities [7 classes]│
    │  → Output: [0.04, 0.03, 0.88, ...]  │
    └─────────────────────────────────────┘
    ↓
Ensemble Averaging:
    - Average probabilities across 3 models
    - avg_prediction = mean([model1_pred, model2_pred, model3_pred])
    - Example: avg = [0.04, 0.02, 0.88, 0.01, 0.02, 0.02, 0.01]
    ↓
Prediction Selection:
    - Find index with highest probability
    - predicted_index = argmax(avg_prediction) = 2
    - Map index to class label: idx2class[2] = "bkl"
    ↓
Confidence Calculation:
    - Get max confidence from individual models for predicted class
    - max_confidence = max(model1[2], model2[2], model3[2])
    - Convert to percentage: "88%"
    ↓
Disease Information Lookup:
    - Map class label to user-friendly info
    - Get disease name and description
    ↓
Return JSON Response:
    {
      "message": "Image received and processed successfully",
      "filename": "skin_lesion_123.jpg",
      "predicted_disease": "bkl",
      "confidence": "88",
      "disease_details": {
        "name": "Benign Keratosis",
        "description": "Benign keratoses are non-cancerous..."
      }
    }
    ↓
Frontend receives response
    ↓
Display results in ResultsCard:
    - Disease name
    - Confidence percentage
    - Description
    - Visual indicators (color-coded by severity)
    ↓
User views prediction results
```

### Technical Details

**Frontend Components:**
- Upload: `ImageUploadCard.js`
- Results: `ResultsCard.js`
- Hook: `usePrediction.js`
- Service: `predictionService.predict(formData)`

**Backend Processing:**
- Route: `POST /predict`
- Handler: `prediction_routes.py::prediction()`
- Preprocessing: `load_and_preprocess_image()`
- Models: Loaded once at startup, kept in memory

**Model Details:**
- Input Shape: (1, 224, 224, 3)
- Output Shape: (1, 7) - softmax probabilities
- Preprocessing: EfficientNet's preprocess_input
- Inference Time: ~1-3 seconds (depends on hardware)

**Disease Mapping:**
```python
class_indices = {
    "akiec": 0,  # Actinic Keratoses
    "bcc": 1,    # Basal Cell Carcinoma
    "bkl": 2,    # Benign Keratosis
    "df": 3,     # Dermatofibroma
    "mel": 4,    # Melanoma
    "nv": 5,     # Melanocytic Nevus
    "vasc": 6    # Vascular Lesion
}
```

---

## 5. Protected Route Access Flow

### Step-by-Step Process

```
User tries to access /dashboard
    ↓
React Router checks route protection
    ↓
ProtectedRoute component checks authentication
    ↓
If not authenticated:
    - Redirect to /login
    ↓
If authenticated:
    - Render Dashboard component
    ↓
Dashboard makes API calls with JWT token
    ↓
Backend validates token on each request
    ↓
If token valid → Process request
If token invalid → Return 401 → Frontend logs out user
```

### Technical Details

**Frontend:**
- Component: `ProtectedRoute.js`
- Check: `authService.isAuthenticated()`
- Redirect: React Router `<Navigate>`

**Backend:**
- Decorator: `@token_required` or `@jwt_required`
- Validation: JWT signature and expiration
- Error Responses: 401 with error type

---

## 6. Logout Flow

### Step-by-Step Process

```
User clicks logout button
    ↓
authService.logout() called
    ↓
Remove token from localStorage
    ↓
Remove user info from localStorage
    ↓
Redirect to /login
    ↓
User session ended
```

### Technical Details

**Frontend:**
- Service: `authService.logout()`
- Storage: `localStorage.removeItem()`
- Redirect: `window.location.href = '/login'`

**Backend:**
- No backend call needed (stateless JWT)
- Token becomes invalid after expiration

---

## 7. Error Handling Flow

### Authentication Errors

```
API Request with invalid/expired token
    ↓
Backend returns 401 Unauthorized
    ↓
Axios interceptor catches error
    ↓
Frontend calls authService.logout()
    ↓
Redirect to login page
```

### Prediction Errors

```
Image upload fails / Model error
    ↓
Backend returns error response
    ↓
Frontend catches error in try-catch
    ↓
Display error message in UI
    ↓
User can retry with different image
```

### Network Errors

```
Request fails (network issue)
    ↓
Axios throws error
    ↓
Frontend displays generic error message
    ↓
User can retry action
```

---

## 8. Application Startup Flow

### Backend Startup

```
python main.py executed
    ↓
Load environment variables from .env
    ↓
Create Flask app with create_app()
    ↓
Configure app settings (JWT, CORS, uploads)
    ↓
Initialize JWT Manager
    ↓
Register blueprints and routes
    ↓
Register error handlers
    ↓
Load 3 ML models from trained_models/
    - resnet101.h5
    - densenet121.h5
    - efficientnetb3.h5
    ↓
Models kept in memory for fast inference
    ↓
Start Flask server on 0.0.0.0:5000
    ↓
Ready to accept requests
```

### Frontend Startup

```
npm start executed
    ↓
React development server starts
    ↓
Load App.js
    ↓
Check authentication status
    ↓
Verify token if exists
    ↓
Route to appropriate page (login or dashboard)
    ↓
Render UI components
    ↓
Ready for user interaction
```

---

## 9. Image Upload & Validation Flow

### Frontend Validation

```
User selects image file
    ↓
Check file type (must be image/*)
    ↓
Check file size (max 10MB - frontend limit)
    ↓
If valid:
    - Create object URL for preview
    - Store file in state
    ↓
If invalid:
    - Show error message
    - Reject file
```

### Backend Validation

```
Receive file upload
    ↓
Check 'image' field exists in request.files
    ↓
Check filename is not empty
    ↓
Check file size (max 16MB - backend limit)
    ↓
Sanitize filename with secure_filename()
    ↓
Save to uploads/ directory
    ↓
Validate image can be read by OpenCV
    ↓
If any validation fails → Return 400 Bad Request
```

---

## 10. Model Training Flow (Offline Process)

**Note:** This flow is separate from the main application and runs offline.

```
Download HAM10000 dataset from Kaggle
    ↓
Extract to backend/data/skin_disease_dataset/
    ↓
Organize into train_dir/, val_dir/, test_dir/
    ↓
Open Jupyter notebook (model_training.ipynb)
    ↓
Load and preprocess dataset:
    - Resize images to 224×224
    - Apply data augmentation
    - Create data generators
    ↓
Train each model separately:
    ┌─────────────────────────────────┐
    │  Train EfficientNetB3           │
    │  - Transfer learning            │
    │  - Fine-tune last layers        │
    │  - Save to trained_models/      │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │  Train ResNet101                │
    │  - Transfer learning            │
    │  - Fine-tune last layers        │
    │  - Save to trained_models/      │
    └─────────────────────────────────┘
    ┌─────────────────────────────────┐
    │  Train DenseNet121              │
    │  - Transfer learning            │
    │  - Fine-tune last layers        │
    │  - Save to trained_models/      │
    └─────────────────────────────────┘
    ↓
Evaluate models on test set
    ↓
Save trained models as .h5 files
    ↓
Models ready for deployment
```

---

## Flow Timing Estimates

| Flow | Estimated Time |
|------|---------------|
| Registration | 1-2 seconds |
| Login | 1-2 seconds |
| Token Verification | < 1 second |
| Image Upload | 1-3 seconds (depends on size) |
| Model Inference | 1-3 seconds (depends on hardware) |
| Complete Prediction | 2-6 seconds total |
| Logout | < 1 second |

---

## Key Takeaways

1. **Stateless Authentication:** JWT tokens enable stateless auth, no session storage needed
2. **Ensemble Prediction:** Averaging 3 models improves accuracy and robustness
3. **Synchronous Processing:** Current implementation is synchronous, blocking during inference
4. **Client-Side State:** Authentication state managed in localStorage
5. **Error Recovery:** Automatic logout on token expiration, retry capability for predictions
6. **Preprocessing Critical:** Image preprocessing must match training pipeline exactly
7. **Model Loading:** Models loaded once at startup for performance
