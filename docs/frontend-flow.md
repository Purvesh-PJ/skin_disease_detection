# Frontend Flow Documentation

This document explains the React frontend architecture, component structure, state management, and data flow patterns.

---

## Frontend Architecture Overview

The frontend is a **React 18 Single Page Application (SPA)** using functional components with hooks, styled-components for styling, and React Router for navigation.

```
User Interface (React Components)
    ↓
React Router (Navigation)
    ↓
Custom Hooks (Business Logic)
    ↓
Services Layer (API Communication)
    ↓
Axios Instance (HTTP Client)
    ↓
Backend API
```

---

## Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.3.1 | UI framework |
| React Router DOM | 6.27.0 | Client-side routing |
| Axios | 1.7.7 | HTTP client |
| Styled Components | 6.1.13 | CSS-in-JS styling |
| React Icons | 5.5.0 | Icon library |

---

## Project Structure

```
frontend/src/
├── assets/              # Static assets (images, icons)
├── components/          # Reusable UI components
│   ├── common/         # Generic UI components (Button, Input, Card)
│   ├── features/       # Feature-specific components
│   │   └── prediction/ # Prediction-related components
│   └── layout/         # Layout components (Header, Footer)
├── config/             # Configuration files
├── constants/          # Application constants
├── context/            # React Context providers
├── hooks/              # Custom React hooks
├── pages/              # Page components
│   ├── Dashboard/
│   ├── Login/
│   ├── Signup/
│   └── NotFound/
├── routes/             # Routing configuration
├── services/           # API service layer
│   └── api/           # API communication
├── styles/             # Global styles and theme
├── App.js              # Root component
└── index.js            # Application entry point
```

---

## Component Hierarchy

```
App.js
├── BrowserRouter
    ├── Routes
        ├── / (Redirect based on auth)
        ├── /login → Login Page
        ├── /signup → Signup Page
        ├── /dashboard → ProtectedRoute
        │   └── Dashboard
        │       ├── Header
        │       ├── ImageUploadCard
        │       └── ResultsCard
        └── * → NotFound Page
```

---

## State Management Strategy

### 1. Local Component State
Used for UI-specific state that doesn't need to be shared.

**Example: ImageUploadCard**
```javascript
const [selectedImage, setSelectedImage] = useState(null);
const [imageFile, setImageFile] = useState(null);
const [isDragging, setIsDragging] = useState(false);
```

### 2. localStorage
Used for persisting authentication data across sessions.

**Stored Data:**
- `token` - JWT access token
- `user` - User information (username, email, roles)

**Access Pattern:**
```javascript
// Store
localStorage.setItem('token', token);
localStorage.setItem('user', JSON.stringify(user));

// Retrieve
const token = localStorage.getItem('token');
const user = JSON.parse(localStorage.getItem('user'));

// Remove
localStorage.removeItem('token');
localStorage.removeItem('user');
```

### 3. Context API (ThemeContext)
Used for global theme state (if dark mode is implemented).

```javascript
const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

---

## Routing System

### Route Configuration

**File:** `frontend/src/routes/index.js`

```javascript
<Routes>
  <Route path="/" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} />
  <Route path="/login" element={<Login />} />
  <Route path="/signup" element={<Signup />} />
  <Route 
    path="/dashboard" 
    element={
      <ProtectedRoute>
        <Dashboard />
      </ProtectedRoute>
    } 
  />
  <Route path="*" element={<NotFound />} />
</Routes>
```

### Protected Route Pattern

**File:** `frontend/src/routes/ProtectedRoute.js`

```javascript
const ProtectedRoute = ({ children }) => {
  const isAuthenticated = authService.isAuthenticated();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
};
```

**How it works:**
1. Check if user has valid token in localStorage
2. If authenticated → render children (Dashboard)
3. If not authenticated → redirect to login

---

## Custom Hooks

### useAuth Hook

**File:** `frontend/src/hooks/useAuth.js`

**Purpose:** Manage authentication state and operations

```javascript
const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    verifyToken();
  }, []);
  
  const verifyToken = async () => {
    // Verify token with backend
  };
  
  const login = async (email, password) => {
    // Login logic
  };
  
  const logout = () => {
    // Logout logic
  };
  
  return { user, loading, login, logout };
};
```

### usePrediction Hook

**File:** `frontend/src/hooks/usePrediction.js`

**Purpose:** Handle prediction logic and state

```javascript
const usePrediction = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const predict = async (imageFile) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      
      const response = await predictionService.predict(formData);
      setResult(response.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const reset = () => {
    setResult(null);
    setError(null);
  };
  
  return { result, loading, error, predict, reset };
};
```

---

## Services Layer

### Axios Configuration

**File:** `frontend/src/services/api/axios.js`

```javascript
import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add JWT token
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - Handle errors
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;
```

### Auth Service

**File:** `frontend/src/services/api/auth.service.js`

```javascript
import axiosInstance from './axios';

const authService = {
  login: async (email, password) => {
    const response = await axiosInstance.post('/auth/login', {
      email,
      password,
    });
    
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  },
  
  register: async (userData) => {
    const response = await axiosInstance.post('/auth/register', userData);
    return response.data;
  },
  
  verifyToken: async () => {
    const response = await axiosInstance.get('/auth/verify-token');
    return response.data;
  },
  
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login';
  },
  
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },
  
  getToken: () => {
    return localStorage.getItem('token');
  },
  
  getUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },
};

export default authService;
```

### Prediction Service

**File:** `frontend/src/services/api/prediction.service.js`

```javascript
import axiosInstance from './axios';

const predictionService = {
  predict: async (formData) => {
    const response = await axiosInstance.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default predictionService;
```

---

## Key Components Deep Dive

### 1. App.js (Root Component)

**Responsibilities:**
- Initialize application
- Verify authentication on mount
- Manage global loading state
- Render routing

**Flow:**
```javascript
useEffect(() => {
  const verifyAuth = async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        await authService.verifyToken();
        setIsAuthenticated(true);
      } catch (error) {
        authService.logout();
      }
    }
    setLoading(false);
  };
  
  verifyAuth();
}, []);
```

### 2. Login Page

**File:** `frontend/src/pages/Login/`

**State:**
```javascript
const [formData, setFormData] = useState({
  email: '',
  password: '',
});
const [error, setError] = useState('');
const [loading, setLoading] = useState(false);
```

**Form Handling:**
```javascript
const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError('');
  
  try {
    await authService.login(formData.email, formData.password);
    navigate('/dashboard');
  } catch (err) {
    setError(err.response?.data?.message || 'Login failed');
  } finally {
    setLoading(false);
  }
};
```

**Validation:**
- Email format validation
- Password minimum length
- Required field checks

### 3. Signup Page

**File:** `frontend/src/pages/Signup/`

**State:**
```javascript
const [formData, setFormData] = useState({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
});
```

**Validation:**
- Username: 3-20 characters
- Email: Valid email format
- Password: Minimum 8 characters, must include uppercase, lowercase, number
- Confirm Password: Must match password

**Flow:**
```javascript
const handleSubmit = async (e) => {
  e.preventDefault();
  
  // Validate passwords match
  if (formData.password !== formData.confirmPassword) {
    setError('Passwords do not match');
    return;
  }
  
  try {
    await authService.register({
      username: formData.username,
      email: formData.email,
      password: formData.password,
    });
    
    // Show success message
    setSuccess('Registration successful! Redirecting to login...');
    
    // Redirect after 2 seconds
    setTimeout(() => navigate('/login'), 2000);
  } catch (err) {
    setError(err.response?.data?.message || 'Registration failed');
  }
};
```

### 4. Dashboard Page

**File:** `frontend/src/pages/Dashboard/`

**Layout:**
```
┌─────────────────────────────────────────┐
│              Header                     │
│  [Logo] [User Info] [Logout]           │
├──────────────────┬──────────────────────┤
│                  │                      │
│  ImageUploadCard │    ResultsCard       │
│                  │                      │
│  [Drag & Drop]   │  [Disease Name]      │
│  [Image Preview] │  [Confidence]        │
│  [Analyze Btn]   │  [Description]       │
│                  │                      │
└──────────────────┴──────────────────────┘
```

**State:**
```javascript
const [selectedImage, setSelectedImage] = useState(null);
const [imageFile, setImageFile] = useState(null);
const [predictionResult, setPredictionResult] = useState(null);
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);
```

**Prediction Flow:**
```javascript
const handleAnalyze = async () => {
  if (!imageFile) return;
  
  setLoading(true);
  setError(null);
  
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await predictionService.predict(formData);
    setPredictionResult(response);
  } catch (err) {
    setError(err.response?.data?.message || 'Prediction failed');
  } finally {
    setLoading(false);
  }
};
```

### 5. ImageUploadCard Component

**File:** `frontend/src/components/features/prediction/ImageUploadCard.js`

**Features:**
- Drag and drop support
- Click to upload
- Image preview
- File validation
- Clear functionality

**Drag & Drop Implementation:**
```javascript
const handleDragOver = (e) => {
  e.preventDefault();
  setIsDragging(true);
};

const handleDragLeave = (e) => {
  e.preventDefault();
  setIsDragging(false);
};

const handleDrop = (e) => {
  e.preventDefault();
  setIsDragging(false);
  
  const file = e.dataTransfer.files[0];
  handleFileSelect(file);
};

const handleFileSelect = (file) => {
  // Validate file type
  if (!file.type.startsWith('image/')) {
    setError('Please select an image file');
    return;
  }
  
  // Validate file size (10MB)
  if (file.size > 10 * 1024 * 1024) {
    setError('File size must be less than 10MB');
    return;
  }
  
  // Create preview URL
  const previewUrl = URL.createObjectURL(file);
  setSelectedImage(previewUrl);
  setImageFile(file);
  setError(null);
};
```

### 6. ResultsCard Component

**File:** `frontend/src/components/features/prediction/ResultsCard.js`

**Display States:**
1. **Empty State:** No prediction yet
2. **Loading State:** Showing spinner during prediction
3. **Success State:** Display results
4. **Error State:** Show error message

**Result Display:**
```javascript
{predictionResult && (
  <ResultContainer>
    <DiseaseLabel>{predictionResult.disease_details.name}</DiseaseLabel>
    <ConfidenceBar>
      <ConfidenceFill width={predictionResult.confidence} />
    </ConfidenceBar>
    <ConfidenceText>{predictionResult.confidence}% Confidence</ConfidenceText>
    <Description>{predictionResult.disease_details.description}</Description>
  </ResultContainer>
)}
```

---

## Styling System

### Styled Components Pattern

**Theme Configuration:**
```javascript
// frontend/src/styles/theme.js
export const theme = {
  colors: {
    primary: '#4A90E2',
    secondary: '#50C878',
    danger: '#E74C3C',
    warning: '#F39C12',
    text: '#333333',
    textLight: '#666666',
    background: '#FFFFFF',
    backgroundLight: '#F5F5F5',
    border: '#E0E0E0',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    full: '50%',
  },
  shadows: {
    sm: '0 2px 4px rgba(0,0,0,0.1)',
    md: '0 4px 8px rgba(0,0,0,0.15)',
    lg: '0 8px 16px rgba(0,0,0,0.2)',
  },
};
```

**Component Styling Example:**
```javascript
import styled from 'styled-components';

const Button = styled.button`
  background-color: ${props => props.theme.colors.primary};
  color: white;
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  border: none;
  cursor: pointer;
  font-size: 16px;
  transition: all 0.3s ease;
  
  &:hover {
    opacity: 0.9;
    transform: translateY(-2px);
    box-shadow: ${props => props.theme.shadows.md};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;
```

---

## Form Handling Patterns

### Controlled Components

```javascript
const [formData, setFormData] = useState({
  email: '',
  password: '',
});

const handleChange = (e) => {
  const { name, value } = e.target;
  setFormData(prev => ({
    ...prev,
    [name]: value,
  }));
};

<input
  type="email"
  name="email"
  value={formData.email}
  onChange={handleChange}
/>
```

### Form Validation

```javascript
const validateForm = () => {
  const errors = {};
  
  if (!formData.email) {
    errors.email = 'Email is required';
  } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
    errors.email = 'Email is invalid';
  }
  
  if (!formData.password) {
    errors.password = 'Password is required';
  } else if (formData.password.length < 8) {
    errors.password = 'Password must be at least 8 characters';
  }
  
  return errors;
};
```

---

## Error Handling

### API Error Handling

```javascript
try {
  const response = await authService.login(email, password);
  // Success handling
} catch (error) {
  if (error.response) {
    // Server responded with error
    setError(error.response.data.message);
  } else if (error.request) {
    // Request made but no response
    setError('Network error. Please check your connection.');
  } else {
    // Something else happened
    setError('An unexpected error occurred');
  }
}
```

### Global Error Boundary (Recommended Addition)

```javascript
class ErrorBoundary extends React.Component {
  state = { hasError: false };
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback />;
    }
    return this.props.children;
  }
}
```

---

## Performance Optimizations

### 1. Code Splitting (Recommended)

```javascript
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Login = lazy(() => import('./pages/Login'));

<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/login" element={<Login />} />
  </Routes>
</Suspense>
```

### 2. Memoization

```javascript
import { useMemo, useCallback } from 'react';

// Memoize expensive calculations
const processedData = useMemo(() => {
  return expensiveOperation(data);
}, [data]);

// Memoize callback functions
const handleClick = useCallback(() => {
  doSomething(id);
}, [id]);
```

### 3. Image Optimization

```javascript
// Cleanup object URLs to prevent memory leaks
useEffect(() => {
  return () => {
    if (selectedImage) {
      URL.revokeObjectURL(selectedImage);
    }
  };
}, [selectedImage]);
```

---

## Build & Deployment

### Development Build

```bash
npm start
# Runs on http://localhost:3000
# Hot reload enabled
```

### Production Build

```bash
npm run build
# Creates optimized build in build/ folder
# Minified, bundled, ready for deployment
```

### Environment Variables

Create `.env` file:
```
REACT_APP_API_URL=http://localhost:5000
```

Access in code:
```javascript
const API_URL = process.env.REACT_APP_API_URL;
```

---

## Testing Strategy (Recommended)

### Unit Tests

```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import Login from './Login';

test('renders login form', () => {
  render(<Login />);
  expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
});

test('shows error on invalid credentials', async () => {
  render(<Login />);
  fireEvent.click(screen.getByRole('button', { name: /login/i }));
  expect(await screen.findByText(/invalid credentials/i)).toBeInTheDocument();
});
```

---

## Key Takeaways

1. **Component-Based Architecture:** Modular, reusable components
2. **Hooks-Based State:** Modern React patterns with functional components
3. **Service Layer:** Separation of concerns, API logic isolated
4. **Protected Routes:** Authentication-based navigation
5. **Styled Components:** CSS-in-JS for scoped styling
6. **Error Handling:** Comprehensive error handling at multiple levels
7. **Token Management:** JWT stored in localStorage, auto-injected in requests
8. **Responsive Design:** Mobile-friendly UI (assumed from styled-components)

---

## Common Patterns Used

- **Container/Presentational Pattern:** Pages (containers) use feature components (presentational)
- **Custom Hooks Pattern:** Reusable logic extracted into hooks
- **Service Pattern:** API calls abstracted into service layer
- **HOC Pattern:** ProtectedRoute wraps components for auth
- **Compound Components:** Complex components broken into smaller parts
