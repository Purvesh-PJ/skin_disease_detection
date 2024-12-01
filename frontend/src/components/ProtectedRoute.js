import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { verifyToken } from '../services/authServices';

const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        await verifyToken();
        setIsAuthenticated(true);
      } catch {
        setIsAuthenticated(false);
      }
    };
    checkAuth();
  }, []);

  if (isAuthenticated === null) return <p>Loading...</p>;
  return isAuthenticated ? children : <Navigate to="/login" />;
};

export default ProtectedRoute;
