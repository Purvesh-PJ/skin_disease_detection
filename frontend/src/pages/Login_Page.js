import React, { useState } from 'react';
import styled from 'styled-components';
import { login } from '../services/authServices';
import { Link } from 'react-router-dom';

const LoginContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    background-color: #f4f4f4;
`;

const Form = styled.form`
    display: flex;
    flex-direction: column;
    width: 300px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

const Input = styled.input`
    margin-bottom: 15px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-size: 16px;
`;

const Button = styled.button`
    padding: 10px;
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;

    &:hover {
        background-color: #3b82f6;
    }
`;

const ErrorMessage = styled.p`
    color: red;
    font-size: 14px;
`;

const Heading = styled.h1`
    color : #1e293b;
    font-size : 1rem;
    text-align : center;
`;

const Paragraph = styled.p`
    color : #4b5563;
    font-size : 0.80rem;
    text-align : center;
`;

const SignupLink = styled(Link)`
    text-decoration : none;
    margin-left : 4px;
    color : #2563eb;
`;

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(email, password);
      window.location.href = '/dashboard'; // Redirect after successful login
    } catch (err) {
      setError('Invalid email or password');
    }
  };

  return (
    <LoginContainer>
      <Form onSubmit={handleSubmit}>
        <Heading >Login</Heading>
        <Input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Button type="submit">Login</Button>
        <hr/>
        <Paragraph>
            Don't have an account? 
            <SignupLink to="/signup">Sign up here</SignupLink>
        </Paragraph>
      </Form>
    </LoginContainer>
  );
};

export default Login;
