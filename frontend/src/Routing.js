import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { getToken, verifyToken } from './services/authServices';
import styled from 'styled-components';
import Login from './pages/Login_Page';
import Signup from './pages/Signup_Page';
// import Dashboard from './pages/Dashboard';
import ProtectedRoute from './components/ProtectedRoute';
import Dashboard from './pages/Dashboard';


const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f5f5f5;

  &::after {
    content: '';
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;


const AppRoutes = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(null); // null = loading, true = authenticated, false = unauthenticated

  useEffect(() => {
    const checkToken = async () => {
      const token = getToken();
      if (token) {
        try {
          await verifyToken(); // Verify if the token is valid
          setIsAuthenticated(true); // Token is valid, user is authenticated
        } catch {
          setIsAuthenticated(false); // Token is invalid, clear it
          localStorage.removeItem('token');
        }
      } else {
        setIsAuthenticated(false); // No token present
      }
    };
    checkToken();
  }, []);

  if (isAuthenticated === null) return <LoadingSpinner />; // Display loading while checking token

  return (
      <Routes>
        {/* Redirect based on authentication */}
        <Route
          path="/"
          element={isAuthenticated ? <Navigate to="/dashboard" /> : <Navigate to="/login" />}
        />
        <Route path="/login" element={isAuthenticated ? <Navigate to="/dashboard" /> : <Login />} />
        <Route path="/signup" element={isAuthenticated ? <Navigate to="/dashboard" /> : <Signup />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<div>404 - Page Not Found</div>} />
      </Routes>
  );
};

export default AppRoutes;
