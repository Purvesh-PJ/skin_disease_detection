import AppRoutes from './routes';
import { useAuth } from './context/AuthContext';

const App = () => {
  const { isAuthenticated } = useAuth();
  return <AppRoutes isAuthenticated={isAuthenticated} />;
};



export default App;
