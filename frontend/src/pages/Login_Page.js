import React, { useState } from 'react';
import styled from 'styled-components';
import { login } from '../services/authApi';
import { Link, useNavigate } from 'react-router-dom';
import loginImage from '../resources/images/7108455 1.png';
import { Button, Input, Alert, Spinner } from '../components/ui';
import { H2, Text, SmallText } from '../styles/typography';

const Container = styled.div`
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
`;

const LoginContainer = styled.div`
  display: flex;
  width: 100%;
  max-width: 1150px;
  min-height: 70vh;
  border-radius: ${({ theme }) => theme.borderRadius['3xl']};
  box-shadow: ${({ theme }) => theme.shadows.subtle};
  background-color: ${({ theme }) => theme.colors.background.primary};
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    flex-direction: column;
    min-height: auto;
  }
`;

const Column = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[6]};
`;

const LeftColumn = styled(Column)`
  background-color: ${({ theme }) => theme.colors.background.primary};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const Illustration = styled.div`
  width: 80%;
  text-align: center;

  img {
    width: 100%;
    height: auto;
  }
`;

const RightColumn = styled(Column)`
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-width: 400px;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const StyledInput = styled(Input)`
  width: 100%;
`;

const LinkText = styled(SmallText)`
  text-align: center;
  color: ${({ theme }) => theme.colors.text.secondary};

  a {
    color: ${({ theme }) => theme.colors.primary[600]};
    text-decoration: none;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const Divider = styled.hr`
  width: 100%;
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  margin: ${({ theme }) => theme.spacing[2]} 0;
`;

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (!email || !password) {
      setError('Email and password are required');
      setLoading(false);
      return;
    }

    try {
      const response = await login(email, password);
      if (response?.token) {
        navigate('/dashboard');
      } else {
        setError('Invalid credentials');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) setError('');
  };

  return (
    <Container>
      <LoginContainer>
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
              Don't have an account? <Link to="/signup">Sign up</Link>
            </LinkText>
          </Form>
        </RightColumn>
      </LoginContainer>
    </Container>
  );
};

export default Login;
