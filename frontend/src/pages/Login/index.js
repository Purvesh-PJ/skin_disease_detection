import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Input, Alert, Spinner } from '../../components/common/ui';
import { H2, SmallText } from '../../styles/typography';
import loginImage from '../../assets/images/7108455 1.png';
import {
  Container,
  AuthContainer,
  LeftColumn,
  RightColumn,
  Illustration,
  Form,
  StyledInput,
  LinkText,
  Divider,
} from './styles';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login, loading, error, clearError } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      return;
    }

    try {
      await login(email, password);
      navigate(ROUTES.DASHBOARD);
    } catch {
      // Error is handled by useAuth hook
    }
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) clearError();
  };

  return (
    <Container>
      <AuthContainer>
        <LeftColumn>
          <Illustration>
            <img src={loginImage} alt="Skin Disease Illustration" />
          </Illustration>
        </LeftColumn>
        <RightColumn>
          <Form onSubmit={handleSubmit}>
            <H2>Login</H2>
            <StyledInput
              type="email"
              placeholder="Email"
              value={email}
              onChange={handleInputChange(setEmail)}
              error={!!error}
              required
            />
            <StyledInput
              type="password"
              placeholder="Password"
              value={password}
              onChange={handleInputChange(setPassword)}
              error={!!error}
              required
            />
            {error && <Alert variant="error">{error}</Alert>}
            <Button type="submit" disabled={loading} style={{ width: '60%' }}>
              {loading ? <Spinner size="sm" color="white" /> : 'Login'}
            </Button>
            <LinkText>
              <Link to="/forgot-password">Forgot password?</Link>
            </LinkText>
            <Divider />
            <LinkText>
              Don't have an account? <Link to={ROUTES.SIGNUP}>Sign up</Link>
            </LinkText>
          </Form>
        </RightColumn>
      </AuthContainer>
    </Container>
  );
};

export default Login;
