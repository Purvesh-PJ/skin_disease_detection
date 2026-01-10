import { useEffect, useState } from 'react';
import styled from 'styled-components';
import AppRoutes from './routes';
import { authService } from './services';
import { Spinner } from './components/common/ui';

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const checkToken = async () => {
      const token = authService.getToken();
      if (token) {
        try {
          await authService.verifyToken();
          setIsAuthenticated(true);
        } catch {
          authService.logout();
          setIsAuthenticated(false);
        }
      } else {
        setIsAuthenticated(false);
      }
    };
    checkToken();
  }, []);

  if (isAuthenticated === null) {
    return (
      <LoadingContainer>
        <Spinner size="lg" />
      </LoadingContainer>
    );
  }

  return <AppRoutes isAuthenticated={isAuthenticated} />;
};

export default App;
