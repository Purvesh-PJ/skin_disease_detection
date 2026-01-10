import React, { useState } from 'react';
import styled from 'styled-components';
import { register } from '../services/authApi';
import { Link, useNavigate } from 'react-router-dom';
import loginImage from '../resources/images/7108455 1.png';
import { Button, Input, Alert } from '../components/ui';
import { H2, SmallText } from '../styles/typography';

const Container = styled.div`
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
`;

const SignupContainer = styled.div`
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

const Signup = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    try {
      const userData = { username, email, password };
      await register(userData);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message || 'Registration failed. Please try again.');
    }
  };

  return (
    <Container>
      <SignupContainer>
        <LeftColumn>
          <Illustration>
            <img src={loginImage} alt="Signup Illustration" />
          </Illustration>
        </LeftColumn>
        <Column>
          <Form onSubmit={handleSubmit}>
            <H2>Sign Up</H2>
            <StyledInput
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <StyledInput
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <StyledInput
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <StyledInput
              type="password"
              placeholder="Confirm Password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
            {error && <Alert variant="error">{error}</Alert>}
            <Button type="submit" style={{ width: '60%' }}>Sign Up</Button>
            <Divider />
            <LinkText>
              Already have an account? <Link to="/login">Log in</Link>
            </LinkText>
          </Form>
        </Column>
      </SignupContainer>
    </Container>
  );
};

export default Signup;
