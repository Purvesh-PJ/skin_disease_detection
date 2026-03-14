# API Overview Documentation

Complete reference for all REST API endpoints in the Skin Disease Detection system.

---

## Base URL

```
Development: http://localhost:5000
Production: https://api.yourdomain.com
```

---

## Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

---

## Response Format

### Success Response
```json
{
  "message": "Success message",
  "data": { ... }
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Error description",
  "error": "error_type"
}
```

---

## HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Authentication required or failed |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 500 | Internal Server Error - Server error |

---

## Endpoints

### 1. Health Check

#### GET /

Check if API is running.

**Authentication:** Not required

**Request:**
```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "message": "Skin Disease Detection API is running",
  "status": "healthy"
}
```

---

### 2. User Registration

#### POST /auth/register

Create a new user account.

**Authentication:** Not required

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123"
}
```

**Validation Rules:**
- `username`: Required, 3-20 characters
- `email`: Required, valid email format, must be unique
- `password`: Required, minimum 8 characters

**Success Response (201):**
```json
{
  "message": "User registered successfully!"
}
```

**Error Responses:**

**400 - Missing Fields:**
```json
{
  "status": "error",
  "message": "Missing required fields",
  "error": "bad_request"
}
```

**409 - Email Already Exists:**
```json
{
  "status": "error",
  "message": "Email already registered",
  "error": "conflict"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePass123"
  }'
```

---

### 3. User Login

#### POST /auth/login

Authenticate user and receive JWT token.

**Authentication:** Not required

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "SecurePass123"
}
```

**Success Response (200):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "username": "john_doe",
    "email": "john@example.com",
    "roles": ["user"]
  },
  "message": "Login successful"
}
```

**Error Responses:**

**401 - Invalid Credentials:**
```json
{
  "status": "error",
  "message": "Invalid credentials",
  "error": "unauthorized"
}
```

**400 - Missing Fields:**
```json
{
  "status": "error",
  "message": "Email and password are required",
  "error": "bad_request"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "SecurePass123"
  }'
```

**Token Usage:**
```bash
# Store token from response
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Use in subsequent requests
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/predict
```

---

### 4. Verify Token

#### GET /auth/verify-token

Validate JWT token and get user information.

**Authentication:** Required

**Request Headers:**
```
Authorization: Bearer <token>
```

**Success Response (200):**
```json
{
  "valid": true,
  "user": {
    "email": "john@example.com",
    "username": "john_doe",
    "roles": ["user"]
  }
}
```

**Error Responses:**

**401 - Token Expired:**
```json
{
  "status": "error",
  "message": "The token has expired",
  "error": "token_expired"
}
```

**401 - Invalid Token:**
```json
{
  "status": "error",
  "message": "Signature verification failed",
  "error": "invalid_token"
}
```

**401 - Missing Token:**
```json
{
  "status": "error",
  "message": "Request does not contain an access token",
  "error": "authorization_required"
}
```

**Example:**
```bash
curl -X GET http://localhost:5000/auth/verify-token \
  -H "Authorization: Bearer $TOKEN"
```

---

### 5. Predict Skin Disease

#### POST /predict

Upload skin image and get disease prediction.

**Authentication:** Required

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | Image file (JPEG, PNG) |

**File Requirements:**
- Format: JPEG, PNG, or other image formats
- Max size: 16MB
- Recommended: Dermatoscopic images, clear skin lesion photos

**Success Response (200):**
```json
{
  "message": "Image received and processed successfully",
  "filename": "skin_lesion_123.jpg",
  "predicted_disease": "bkl",
  "confidence": "88",
  "disease_details": {
    "name": "Benign Keratosis",
    "description": "Benign keratoses are non-cancerous skin growths that include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. They are common, especially in older adults, and typically do not require treatment unless they cause discomfort or cosmetic concerns."
  }
}
```

**Response Fields:**
- `filename`: Saved filename on server
- `predicted_disease`: Disease code (akiec, bcc, bkl, df, mel, nv, vasc)
- `confidence`: Confidence percentage (0-100)
- `disease_details.name`: User-friendly disease name
- `disease_details.description`: Disease description

**Disease Codes:**
| Code | Disease Name |
|------|-------------|
| akiec | Actinic Keratoses and Intraepithelial Carcinoma |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevus |
| vasc | Vascular Lesion |

**Error Responses:**

**400 - No Image Provided:**
```json
{
  "status": "error",
  "message": "No image provided",
  "error": "bad_request"
}
```

**400 - No File Selected:**
```json
{
  "status": "error",
  "message": "No file selected",
  "error": "bad_request"
}
```

**401 - Unauthorized:**
```json
{
  "status": "error",
  "message": "Request does not contain an access token",
  "error": "authorization_required"
}
```

**413 - File Too Large:**
```json
{
  "status": "error",
  "message": "File size exceeds maximum limit",
  "error": "payload_too_large"
}
```

**500 - Prediction Error:**
```json
{
  "status": "error",
  "message": "An error occurred during prediction",
  "error": "internal_server_error"
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@/path/to/skin_image.jpg"
```

**Example (JavaScript/Axios):**
```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await axios.post('http://localhost:5000/predict', formData, {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'multipart/form-data'
  }
});

console.log(response.data);
```

**Example (Python/Requests):**
```python
import requests

url = 'http://localhost:5000/predict'
headers = {'Authorization': f'Bearer {token}'}
files = {'image': open('skin_image.jpg', 'rb')}

response = requests.post(url, headers=headers, files=files)
print(response.json())
```

---

## Complete API Flow Example

### 1. Register New User

```bash
curl -X POST http://localhost:5000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jane_doe",
    "email": "jane@example.com",
    "password": "MySecurePass456"
  }'
```

**Response:**
```json
{
  "message": "User registered successfully!"
}
```

### 2. Login

```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane@example.com",
    "password": "MySecurePass456"
  }'
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGl0eSI6ImphbmVAZXhhbXBsZS5jb20iLCJ1c2VybmFtZSI6ImphbmVfZG9lIiwicm9sZXMiOlsidXNlciJdLCJleHAiOjE3MzQ1Njc4OTB9.abc123...",
  "user": {
    "username": "jane_doe",
    "email": "jane@example.com",
    "roles": ["user"]
  }
}
```

### 3. Upload Image for Prediction

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST http://localhost:5000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@skin_lesion.jpg"
```

**Response:**
```json
{
  "message": "Image received and processed successfully",
  "filename": "skin_lesion.jpg",
  "predicted_disease": "mel",
  "confidence": "92",
  "disease_details": {
    "name": "Melanoma",
    "description": "Melanoma is the most serious type of skin cancer. It develops in melanocytes, the cells that produce melanin. Early detection and treatment are crucial for successful outcomes."
  }
}
```

---

## Rate Limiting (Recommended for Production)

**Current Status:** Not implemented

**Recommended Implementation:**
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.headers.get('Authorization'),
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@jwt_required()
def prediction():
    # ...
```

**Suggested Limits:**
- `/auth/register`: 5 per hour per IP
- `/auth/login`: 10 per hour per IP
- `/predict`: 20 per hour per user

---

## CORS Configuration

**Development:**
```python
CORS(app, resources={r"/*": {"origins": "*"}})
```

**Production:**
```python
CORS(app, resources={
    r"/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

---

## Error Handling Best Practices

### Client-Side Error Handling

```javascript
try {
  const response = await axios.post('/predict', formData);
  // Handle success
} catch (error) {
  if (error.response) {
    // Server responded with error
    switch (error.response.status) {
      case 400:
        console.error('Invalid request:', error.response.data.message);
        break;
      case 401:
        console.error('Unauthorized - redirecting to login');
        authService.logout();
        break;
      case 500:
        console.error('Server error:', error.response.data.message);
        break;
    }
  } else if (error.request) {
    // Request made but no response
    console.error('Network error - no response from server');
  } else {
    // Something else happened
    console.error('Error:', error.message);
  }
}
```

---

## Testing the API

### Using Postman

1. **Create Collection:** "Skin Disease Detection API"

2. **Add Environment Variables:**
   - `base_url`: http://localhost:5000
   - `token`: (will be set after login)

3. **Test Register:**
   - Method: POST
   - URL: `{{base_url}}/auth/register`
   - Body: JSON with username, email, password

4. **Test Login:**
   - Method: POST
   - URL: `{{base_url}}/auth/login`
   - Body: JSON with email, password
   - Tests: Save token to environment

5. **Test Prediction:**
   - Method: POST
   - URL: `{{base_url}}/predict`
   - Headers: `Authorization: Bearer {{token}}`
   - Body: form-data with image file

### Using Python Script

```python
import requests

BASE_URL = 'http://localhost:5000'

# Register
register_data = {
    'username': 'testuser',
    'email': 'test@example.com',
    'password': 'TestPass123'
}
response = requests.post(f'{BASE_URL}/auth/register', json=register_data)
print('Register:', response.json())

# Login
login_data = {
    'email': 'test@example.com',
    'password': 'TestPass123'
}
response = requests.post(f'{BASE_URL}/auth/login', json=login_data)
token = response.json()['token']
print('Login:', response.json())

# Predict
headers = {'Authorization': f'Bearer {token}'}
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(f'{BASE_URL}/predict', headers=headers, files=files)
print('Prediction:', response.json())
```

---

## API Versioning (Future Consideration)

**Current:** No versioning

**Recommended for v2:**
```
/api/v1/auth/register
/api/v1/auth/login
/api/v1/predict
```

**Implementation:**
```python
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

@api_v1.route('/predict', methods=['POST'])
def prediction():
    # ...

app.register_blueprint(api_v1)
```

---

## Security Considerations

1. **HTTPS Only in Production**
   - Use SSL/TLS certificates
   - Redirect HTTP to HTTPS

2. **Token Expiration**
   - Access tokens expire in 1 hour
   - Implement refresh token mechanism

3. **Input Validation**
   - Validate all user inputs
   - Sanitize filenames
   - Check file types and sizes

4. **Rate Limiting**
   - Prevent abuse
   - Protect against DDoS

5. **CORS Configuration**
   - Restrict origins in production
   - Only allow necessary methods

---

## Monitoring & Logging

**Recommended Tools:**
- **Logging:** Python logging module (already implemented)
- **Monitoring:** Prometheus + Grafana
- **Error Tracking:** Sentry
- **API Analytics:** Google Analytics, Mixpanel

**Key Metrics to Track:**
- Request count per endpoint
- Response times
- Error rates
- Token expiration rates
- Prediction accuracy (if ground truth available)

---

## Summary

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/` | GET | No | Health check |
| `/auth/register` | POST | No | Create account |
| `/auth/login` | POST | No | Get JWT token |
| `/auth/verify-token` | GET | Yes | Validate token |
| `/predict` | POST | Yes | Get disease prediction |

**Key Points:**
- JWT authentication for protected endpoints
- Token expires in 1 hour
- Max file upload: 16MB
- Ensemble of 3 models for predictions
- 7 disease classes supported
