import { Routes, Route } from 'react-router-dom';
import { ROUTES } from '../constants';

// Pages
import { Landing, Dashboard, NotFound } from '../pages';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path={ROUTES.HOME} element={<Landing />} />
      <Route path={ROUTES.DASHBOARD} element={<Dashboard />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default AppRoutes;
