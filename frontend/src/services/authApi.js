import axios from 'axios';
const API_URL = 'http://127.0.0.1:5000/auth'; // Update with your backend URL if different

export const verifyToken = async () => {
  const token = getToken();
  if (!token) {
    throw new Error('No token found');
  }

  try {
    const response = await axios.get(`${API_URL}/verify-token`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    console.log(response);
    localStorage.setItem('user', JSON.stringify(response.data.user));
    return response.data;
  } 
  catch(error) {
    logout(); // Clear token on verification failure
    throw new Error(error.response?.data?.error || 'Token verification failed');
  }
};

export const login = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/login`, { email, password }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    console.log("Full Response:", response);
    console.log("Response Data:", response.data.user);

    if (response.status === 200 && response.data?.token) {
      // Store token in localStorage
      localStorage.setItem('token', response.data.token);
      console.log("Token and user stored in localStorage");
      return response.data;
    }
    throw new Error('Token is missing');
  } 
  catch(error) {
    console.error("Login failed:", error);
    throw new Error(error.response?.data?.error || 'Login failed');
  }
};

// Signup service
export const register = async (userData) => {
  try {
    const response = await axios.post(`${API_URL}/register`, userData, {
      headers: {
        'Content-Type': 'application/json', // Explicitly set the content type
      },
    });
    console.log(response);
    return response.data;
  } 
  catch(error) {
    throw new Error(error.response?.data?.error || 'Signup failed');
  }
};

// Logout service
export const logout = () => {
  localStorage.removeItem('token'); // Clear token
  localStorage.removeItem('user'); // Clear user
  window.location.href = '/login'; // Redirect to login
};

// Check if user is authenticated
export const isAuthenticated = () => {
  return !!localStorage.getItem('token');
};

// Get token from local storage
export const getToken = () => {
  return localStorage.getItem('token');
};
