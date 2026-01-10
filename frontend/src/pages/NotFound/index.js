import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { Button } from '../../components/common/ui';
import { H1, Text } from '../../styles/typography';
import { ROUTES } from '../../constants';

const Container = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  text-align: center;
`;

const ErrorCode = styled(H1)`
  font-size: 6rem;
  color: ${({ theme }) => theme.colors.primary[500]};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const NotFound = () => {
  return (
    <Container>
      <ErrorCode>404</ErrorCode>
      <Text variant="secondary" style={{ marginBottom: '24px' }}>
        Page not found
      </Text>
      <Link to={ROUTES.HOME}>
        <Button>Go Home</Button>
      </Link>
    </Container>
  );
};

export default NotFound;
