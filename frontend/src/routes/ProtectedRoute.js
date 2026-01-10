import { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { authService } from '../services';
import { ROUTES } from '../constants';
import { Spinner } from '../components/common/ui';
import styled from 'styled-components';

const LoadingContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        await authService.verifyToken();
        setIsAuthenticated(true);
      } catch {
        setIsAuthenticated(false);
      }
    };
    checkAuth();
  }, []);

  if (isAuthenticated === null) {
    return (
      <LoadingContainer>
        <Spinner size="lg" />
      </LoadingContainer>
    );
  }

  return isAuthenticated ? children : <Navigate to={ROUTES.LOGIN} />;
};

export default ProtectedRoute;
