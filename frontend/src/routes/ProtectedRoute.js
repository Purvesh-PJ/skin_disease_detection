import React from 'react';
import { Navigate } from 'react-router-dom';
import { authService } from '../services';
import { ROUTES } from '../constants';

const ProtectedRoute = ({ children }) => {
  const token = authService.getToken();

  if (!token) {
    return <Navigate to={ROUTES.LOGIN} replace />;
  }

  return children;
};

export default ProtectedRoute;

