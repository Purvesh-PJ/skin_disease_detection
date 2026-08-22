import { useEffect, useState } from 'react';
import AppRoutes from './routes';
import { authService } from './services';


const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!authService.getToken());

  useEffect(() => {
    const token = authService.getToken();
    if (token) {
      authService.verifyToken().catch(() => {
        authService.logout();
        setIsAuthenticated(false);
      });
    }
  }, []);

  return <AppRoutes isAuthenticated={isAuthenticated} />;
};


export default App;
