import { Routes, Route, Navigate } from 'react-router-dom';
import { ROUTES } from '../constants';

// Pages
import { Landing, Login, Signup, Dashboard, NotFound } from '../pages';

const AppRoutes = ({ isAuthenticated }) => {
  return (
    <Routes>
      <Route 
        path={ROUTES.HOME} 
        element={<Landing isAuthenticated={isAuthenticated} />} 
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
        element={<Dashboard />} 
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default AppRoutes;
