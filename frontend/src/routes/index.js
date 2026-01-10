import { Routes, Route, Navigate } from 'react-router-dom';
import { ROUTES } from '../constants';
import ProtectedRoute from './ProtectedRoute';

// Pages
import { Login, Signup, Dashboard, NotFound } from '../pages';

const AppRoutes = ({ isAuthenticated }) => {
  return (
    <Routes>
      <Route 
        path={ROUTES.HOME} 
        element={isAuthenticated ? <Navigate to={ROUTES.DASHBOARD} /> : <Navigate to={ROUTES.LOGIN} />} 
      />
      <Route 
        path={ROUTES.LOGIN} 
        element={isAuthenticated ? <Navigate to={ROUTES.DASHBOARD} /> : <Login />} 
      />
      <Route 
        path={ROUTES.SIGNUP} 
        element={isAuthenticated ? <Navigate to={ROUTES.DASHBOARD} /> : <Signup />} 
      />
      <Route 
        path={ROUTES.DASHBOARD} 
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } 
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default AppRoutes;
