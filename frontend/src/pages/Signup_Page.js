import React, { useState } from 'react';
import styled from 'styled-components';
import { register } from '../services/authServices';
import { Link } from 'react-router-dom';

const SignupContainer = styled.div`
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

const SuccessMessage = styled.p`
  color: green;
  font-size: 14px;
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

const LoginLink = styled(Link)`
    text-decoration : none;
    margin-left : 4px;
    color : #2563eb;
`;

const Signup = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await register({ username, email, password });
      setSuccess('Registration successful! Please log in.');
      setError('');
    } catch (err) {
      setError('Registration failed');
      setSuccess('');
    }
  };

  return (
    <SignupContainer>
      <Form onSubmit={handleSubmit}>
        <Heading >Signup</Heading>
        <Input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
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
        {success && <SuccessMessage>{success}</SuccessMessage>}
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Button type="submit">Signup</Button>
        <hr/>
        <Paragraph>
            Already have an account? 
            <LoginLink  to="/login">Login here</LoginLink>
      </Paragraph>
      </Form>
    </SignupContainer>
  );
};

export default Signup;
