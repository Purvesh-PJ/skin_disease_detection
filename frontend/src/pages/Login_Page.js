import React, { useState } from 'react';
import styled from 'styled-components';
import { login } from '../services/authApi';
import { Link, useNavigate } from 'react-router-dom';
import loginImage from '../resources/images/7108455 1.png';

const Container = styled.div`
  min-height: 99.5vh;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: auto;
`;

const LoginContainer = styled.div`
  display: flex;
  height: 70vh;
  width: 1150px;
  border-radius: 40px;
  box-shadow: rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px;
  background-color: white;
`;

const Column = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const LeftColumn = styled(Column)`
  border-radius: 40px;
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
  background-color: #ffffff;
  border-radius: 40px;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-width: 400px;
  padding: 20px;
  background: white;
  border-radius: 8px;
`;

const Input = styled.input`
  margin-bottom: 15px;
  padding: 10px;
  width: 100%;
  border-radius: 5px;
  font-size: 14px;
  border: 2px solid ${(props) => (props.error ? 'red' : 'gray')};
`;

const Button = styled.button`
  display : flex;
  justify-content : center;
  padding: 10px;
  width: 60%;
  background-color: ${(props) => (props.disabled ? '#f1f5f9' : '#2563eb')};
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 16px;
  cursor: ${(props) => (props.disabled ? 'not-allowed' : 'pointer')};

  &:hover {
    background-color: ${(props) => (props.disabled ? '#f1f5f9' : '#3b82f6')};
  }
`;

const Spinner = styled.div`
  border: 2px solid #ccc;
  border-top: 2px solid #2563eb;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

const ErrorMessage = styled.p`
  color: #ef4444;
  font-size: 12px;
  padding-left : 6px;
  padding-right : 6px; 
  border-radius : 5px;
  background-color : #fef2f2;
  border : 1px solid #f87171;
`;

const Heading = styled.h1`
  color: #1e293b;
  font-size: 1.8rem;
  text-align: center;
  margin-bottom: 20px;
`;

const LinkText = styled.p`
  text-align: center;
  margin-top: 15px;
  font-size: 14px;
  color: gray;

  a {
    color: #2563eb;
    text-decoration: none;
  }
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
      console.log('Login response:', response);

      if (response?.token) {
        navigate('/dashboard');
      } 
      else {
        setError('Invalid credentials');
      }
    } 
    catch (err) {
      setError(err.response?.data?.message || 'Something went wrong. Please try again.');
    } 
    finally {
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
            <Heading>Login</Heading>
            <Input
              type="email"
              placeholder="Email"
              value={email}
              onChange={handleInputChange(setEmail)}
              error={!!error}
              required
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={handleInputChange(setPassword)}
              error={!!error}
              required
            />
            {error && <ErrorMessage>{error}</ErrorMessage>}
            <Button type="submit" disabled={loading}>
              {loading ? <Spinner /> : 'Login'}
            </Button>
            <LinkText>
              <Link to="/forgot-password">Forgot password?</Link>
            </LinkText>
            <hr />
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
