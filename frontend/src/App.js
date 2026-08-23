import { useEffect, useState } from 'react';
import AppRoutes from './routes';
import { authService } from './services';


const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!authService.getToken());

  useEffect(() => {
    const token = authService.getToken();
    if (token && !token.startsWith('demo-session-')) {
      authService.verifyToken().catch((err) => {
        // Only force logout if server explicitly rejected with 401
        if (err.response?.status === 401) {
          authService.logout();
          setIsAuthenticated(false);
        }
      });
    }
  }, []);

  return <AppRoutes isAuthenticated={isAuthenticated} />;
};



export default App;
