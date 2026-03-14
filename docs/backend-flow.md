# Backend Flow Documentation

This document explains the Flask backend architecture, request lifecycle, business logic, and data processing patterns.

---

## Backend Architecture Overview

The backend is a **Flask REST API** using the application factory pattern, with JWT authentication, MongoDB storage, and TensorFlow model inference.

```
HTTP Request
    ↓
Flask App (WSGI)
    ↓
CORS Middleware
    ↓
Request Logging
    ↓
Route Handler (Blueprint)
    ↓
JWT Validation (if protected)
    ↓
Business Logic / Service Layer
    ↓
Database / ML Models
    ↓
Response Formatting
    ↓
HTTP Response
```

---

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| Flask | Web framework |
| Flask-JWT-Extended | JWT authentication |
| Flask-CORS | Cross-origin requests |
| Flask-Bcrypt | Password hashing |
| PyMongo | MongoDB driver |
| TensorFlow/Keras | ML model inference |
| OpenCV | Image processing |
| python-dotenv | Environment configuration |

---

## Project Structure

```
backend/
├── main.py                    # Application entry point
├── app/
│   ├── routes/               # API endpoints
│   │   ├── auth_routes.py   # Authentication
│   │   ├── prediction_routes.py  # ML predictions
│   │   └── home_routes.py   # Health checks
│   ├── ai_models/            # Model architectures
│   │   ├── densenet121_model.py
│   │   ├── efficientnetB3_model.py
│   │   └── resnet101_model.py
│   ├── config/               # Configuration
│   │   └── mongo_config.py  # Database setup
│   ├── db_models/            # Data access layer
│   │   └── user_model.py    # User repository
│   ├── middleware/           # Custom middleware
│   │   └── auth_middleware.py
│   ├── services/             # Business logic
│   │   └── training_ensemble_model_service.py
│   └── utils/                # Utilities
│       └── preprocess.py    # Image preprocessing
├── uploads/                  # Uploaded images
├── trained_models/           # Pre-trained .h5 files
├── data/                     # Training dataset
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables
```

---

## Application Factory Pattern

### main.py - Application Creation

```python
def create_app(testing=False):
    """
    Application factory function
    
    Benefits:
    - Easy testing with different configurations
    - Multiple app instances possible
    - Clean initialization flow
    """
    app = Flask(__name__)
    
    # Configuration
    configure_app(app, testing)
    
    # Setup components
    setup_cors(app)
    jwt = setup_jwt(app)
    
    # Register routes
    register_blueprints(app)
    
    # Error handling
    register_error_handlers(app)
    register_request_handlers(app)
    
    return app
```

### Configuration Management

```python
def configure_app(app, testing=False):
    # Load from environment variables
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_secret')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'default_jwt_secret')
    
    # JWT expiration
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    
    # Upload configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
```

---

## Request Lifecycle

### 1. Request Reception

```
Client sends HTTP request
    ↓
Flask receives request
    ↓
@app.before_request hook
    - Log request info (method, path, IP)
```

### 2. CORS Handling

```python
def setup_cors(app):
    origins = os.getenv('CORS_ORIGINS', '*')
    CORS(app, resources={r"/*": {"origins": origins}})
```

**CORS Headers Added:**
- `Access-Control-Allow-Origin`
- `Access-Control-Allow-Methods`
- `Access-Control-Allow-Headers`

### 3. Route Matching

```python
# Blueprint registration
app.register_blueprint(auth_blueprint, url_prefix='/auth')
setup_home_routes(app)
setup_prediction_routes(app)
```

**Route Resolution:**
1. Check if path matches registered route
2. If no match → 404 error handler
3. If match → execute route handler

### 4. JWT Validation (Protected Routes)

```python
from flask_jwt_extended import jwt_required, get_jwt_identity

@app.route('/predict', methods=['POST'])
@jwt_required()
def prediction():
    current_user = get_jwt_identity()
    # Route logic
```

**JWT Validation Flow:**
```
Extract Authorization header
    ↓
Parse "Bearer <token>"
    ↓
Decode JWT with secret key
    ↓
Check signature validity
    ↓
Check expiration
    ↓
If valid: Continue to route handler
If invalid: Return 401 error
```

### 5. Route Handler Execution

```python
def prediction():
    # 1. Extract request data
    image = request.files.get('image')
    
    # 2. Validate input
    if not image:
        return jsonify({'error': 'No image provided'}), 400
    
    # 3. Process request
    result = process_prediction(image)
    
    # 4. Return response
    return jsonify(result), 200
```

### 6. Response Formatting

```python
@app.after_request
def log_response_info(response):
    logger.debug(f"Response: {response.status}")
    return response
```

### 7. Error Handling

```python
@app.errorhandler(Exception)
def handle_generic_exception(error):
    response = {
        'status': 'error',
        'message': 'An unexpected error occurred',
        'error': 'internal_server_error'
    }
    logger.exception("Unhandled exception occurred")
    return jsonify(response), 500
```

---

## Authentication System

### User Registration Flow

**Route:** `POST /auth/register`

```python
def register():
    # 1. Extract data from request
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # 2. Validate required fields
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # 3. Check if email already exists
    if is_email_taken(email):
        return jsonify({'error': 'Email already registered'}), 409
    
    # 4. Hash password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # 5. Create user in database
    user_data = {
        'username': username,
        'email': email,
        'password': hashed_password,
        'roles': ['user']
    }
    create_user(user_data)
    
    # 6. Return success response
    return jsonify({'message': 'User registered successfully'}), 201
```

**Security Measures:**
- Bcrypt hashing with automatic salt generation
- Email uniqueness validation
- Password never stored in plain text
- Default role assignment

### User Login Flow

**Route:** `POST /auth/login`

```python
def login():
    # 1. Extract credentials
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # 2. Find user by email
    user = find_user_by_email(email)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # 3. Verify password
    if not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # 4. Generate JWT token
    access_token = create_access_token(
        identity=email,
        additional_claims={
            'username': user['username'],
            'roles': user['roles']
        }
    )
    
    # 5. Return token
    return jsonify({
        'token': access_token,
        'user': {
            'username': user['username'],
            'email': user['email'],
            'roles': user['roles']
        }
    }), 200
```

**JWT Token Structure:**
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "identity": "user@example.com",
    "username": "john_doe",
    "roles": ["user"],
    "exp": 1234567890,
    "iat": 1234564290
  },
  "signature": "..."
}
```

### Token Verification Flow

**Route:** `GET /auth/verify-token`

```python
@jwt_required()
def verify_token():
    # JWT already validated by decorator
    current_user = get_jwt_identity()
    
    return jsonify({
        'valid': True,
        'user': current_user
    }), 200
```

---

## Prediction System

### Model Loading (Startup)

```python
# Global model storage
models = {}

def load_models():
    """Load all models once at startup"""
    models['resnet'] = tf.keras.models.load_model('trained_models/resnet101.h5')
    models['densenet'] = tf.keras.models.load_model('trained_models/densenet121.h5')
    models['efficientnet'] = tf.keras.models.load_model('trained_models/efficientnetb3.h5')
    
    logger.info("All models loaded successfully")

# Load models when module is imported
load_models()
```

**Why load at startup:**
- Models are large (100-200MB each)
- Loading takes 5-10 seconds
- Keep in memory for fast inference
- Trade-off: High RAM usage (~1-2GB)

### Prediction Request Flow

**Route:** `POST /predict`

```python
@jwt_required()
def prediction():
    # 1. Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 2. Save uploaded file
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    
    # 3. Preprocess image
    processed_image = load_and_preprocess_image(filepath)
    
    # 4. Run ensemble prediction
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(processed_image)
        predictions[model_name] = pred[0]  # Get first (only) prediction
    
    # 5. Average predictions
    avg_prediction = np.mean([
        predictions['resnet'],
        predictions['densenet'],
        predictions['efficientnet']
    ], axis=0)
    
    # 6. Get predicted class
    predicted_index = np.argmax(avg_prediction)
    predicted_class = idx2class[predicted_index]
    
    # 7. Calculate confidence
    max_confidence = max([
        predictions['resnet'][predicted_index],
        predictions['densenet'][predicted_index],
        predictions['efficientnet'][predicted_index]
    ])
    confidence_percentage = int(max_confidence * 100)
    
    # 8. Get disease information
    disease_info = disease_details[predicted_class]
    
    # 9. Return response
    return jsonify({
        'message': 'Image processed successfully',
        'filename': filename,
        'predicted_disease': predicted_class,
        'confidence': str(confidence_percentage),
        'disease_details': disease_info
    }), 200
```

### Image Preprocessing Pipeline

```python
def load_and_preprocess_image(image_path):
    """
    Preprocess image for model input
    
    Steps:
    1. Read image with OpenCV
    2. Convert BGR to RGB
    3. Resize to 224x224
    4. Apply EfficientNet preprocessing
    5. Expand dimensions for batch
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Normalize (EfficientNet preprocessing)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
```

**Preprocessing Details:**
- Input: Raw image file (any size, any format)
- Output: NumPy array (1, 224, 224, 3)
- Normalization: EfficientNet's preprocess_input
  - Scales pixel values to [-1, 1] range
  - Matches training preprocessing

### Disease Mapping

```python
# Class index to disease code
idx2class = {
    0: 'akiec',  # Actinic Keratoses
    1: 'bcc',    # Basal Cell Carcinoma
    2: 'bkl',    # Benign Keratosis
    3: 'df',     # Dermatofibroma
    4: 'mel',    # Melanoma
    5: 'nv',     # Melanocytic Nevus
    6: 'vasc'    # Vascular Lesion
}

# Disease details
disease_details = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Precancerous skin lesions caused by sun damage...'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer...'
    },
    # ... other diseases
}
```

---

## Database Layer

### MongoDB Configuration

```python
# config/mongo_config.py
from pymongo import MongoClient
import os

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/skin_disease_db')
client = MongoClient(MONGO_URI)
db = client.get_database('skin_disease_db')
users_collection = db.get_collection('users')
```

### User Repository Pattern

```python
# db_models/user_model.py

def create_user(user_data):
    """Insert new user into database"""
    result = users_collection.insert_one(user_data)
    return result.inserted_id

def find_user_by_email(email):
    """Query user by email"""
    return users_collection.find_one({'email': email})

def is_email_taken(email):
    """Check if email exists"""
    return users_collection.find_one({'email': email}) is not None

def update_user(email, update_data):
    """Update user document"""
    return users_collection.update_one(
        {'email': email},
        {'$set': update_data}
    )

def delete_user(email):
    """Delete user by email"""
    return users_collection.delete_one({'email': email})
```

**Benefits of Repository Pattern:**
- Abstracts database operations
- Easy to test (can mock repository)
- Centralized data access logic
- Easy to switch databases

### User Schema

```javascript
{
  "_id": ObjectId("..."),
  "username": String,
  "email": String (unique),
  "password": String (bcrypt hash),
  "roles": Array<String> (default: ["user"]),
  "created_at": Date (optional),
  "updated_at": Date (optional)
}
```

**Indexes (Recommended):**
```python
# Create unique index on email
users_collection.create_index('email', unique=True)
```

---

## Error Handling System

### JWT Error Handlers

```python
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        'status': 'error',
        'message': 'The token has expired',
        'error': 'token_expired'
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        'status': 'error',
        'message': 'Signature verification failed',
        'error': 'invalid_token'
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({
        'status': 'error',
        'message': 'Request does not contain an access token',
        'error': 'authorization_required'
    }), 401
```

### HTTP Exception Handler

```python
@app.errorhandler(HTTPException)
def handle_http_exception(error):
    response = {
        'status': 'error',
        'message': error.description,
        'error': error.name
    }
    logger.error(f"HTTP Error: {error.code} - {error.name}")
    return jsonify(response), error.code
```

### Generic Exception Handler

```python
@app.errorhandler(Exception)
def handle_generic_exception(error):
    response = {
        'status': 'error',
        'message': 'An unexpected error occurred',
        'error': 'internal_server_error'
    }
    logger.exception("Unhandled exception occurred")
    return jsonify(response), 500
```

---

## Logging System

### Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to file
        logging.StreamHandler()           # Log to console
    ]
)
logger = logging.getLogger(__name__)
```

### Request/Response Logging

```python
@app.before_request
def log_request_info():
    logger.debug(f"Request: {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response_info(response):
    logger.debug(f"Response: {response.status}")
    return response
```

**Log Levels:**
- `DEBUG`: Detailed information for debugging
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

---

## Security Best Practices

### 1. Password Security
```python
# Use bcrypt with automatic salt
hashed = bcrypt.generate_password_hash(password).decode('utf-8')
is_valid = bcrypt.check_password_hash(hashed, password)
```

### 2. JWT Security
```python
# Use strong secret keys
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

# Set appropriate expiration
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
```

### 3. File Upload Security
```python
from werkzeug.utils import secure_filename

# Sanitize filename
filename = secure_filename(image_file.filename)

# Limit file size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### 4. CORS Configuration
```python
# Restrict origins in production
CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}})
```

### 5. Environment Variables
```python
# Never hardcode secrets
SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
MONGO_URI = os.getenv('MONGO_URI')
```

---

## Performance Considerations

### 1. Model Loading
**Current:** Models loaded once at startup
**Impact:** 1-2GB RAM usage, fast inference

**Alternative:** Load on-demand
```python
def get_model(model_name):
    if model_name not in models:
        models[model_name] = tf.keras.models.load_model(f'trained_models/{model_name}.h5')
    return models[model_name]
```

### 2. Synchronous Processing
**Current:** Blocking prediction (1-3 seconds)
**Impact:** Server blocked during inference

**Alternative:** Async with Celery
```python
@celery.task
def predict_async(image_path):
    # Run prediction in background
    result = run_prediction(image_path)
    return result
```

### 3. Database Queries
**Current:** Simple queries, no optimization
**Recommended:** Add indexes
```python
users_collection.create_index('email', unique=True)
```

---

## Deployment Configuration

### Development

```bash
# .env file
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
MONGO_URI=mongodb://localhost:27017/skin_disease_db
CORS_ORIGINS=*
```

```bash
python main.py
```

### Production (Recommended)

```bash
# Use production WSGI server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

**Gunicorn Configuration:**
- `-w 4`: 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
- `--timeout 120`: Increase timeout for ML inference

**Nginx Reverse Proxy:**
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Testing Strategy (Recommended)

### Unit Tests

```python
import unittest
from main import create_app

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(testing=True)
        self.client = self.app.test_client()
    
    def test_register(self):
        response = self.client.post('/auth/register', json={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPass123'
        })
        self.assertEqual(response.status_code, 201)
    
    def test_login(self):
        response = self.client.post('/auth/login', json={
            'email': 'test@example.com',
            'password': 'TestPass123'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('token', response.json)
```

---

## Key Takeaways

1. **Application Factory:** Clean initialization, easy testing
2. **Blueprint Pattern:** Modular route organization
3. **JWT Authentication:** Stateless, scalable auth
4. **Repository Pattern:** Abstracted data access
5. **Error Handling:** Comprehensive error management
6. **Logging:** Request/response tracking
7. **Security:** Bcrypt, JWT, secure file handling
8. **Model Loading:** Startup loading for performance
9. **Ensemble Prediction:** Average of 3 models for accuracy

---

## Common Issues & Solutions

### Issue: Models not loading
**Solution:** Check file paths, ensure .h5 files exist
```python
if not os.path.exists('trained_models/resnet101.h5'):
    raise FileNotFoundError("Model file not found")
```

### Issue: MongoDB connection failed
**Solution:** Verify MONGO_URI, check MongoDB is running
```python
try:
    client.server_info()  # Test connection
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
```

### Issue: JWT token expired
**Solution:** Frontend should handle 401 and redirect to login
```python
if error.response.status === 401:
    authService.logout()
```

### Issue: Image upload fails
**Solution:** Check file size, format, and upload folder permissions
```python
if not os.access(app.config['UPLOAD_FOLDER'], os.W_OK):
    raise PermissionError("Upload folder not writable")
```
