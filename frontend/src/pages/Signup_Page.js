import React, { useState } from 'react';
import styled from 'styled-components';
import { register } from '../services/authServices'; // Assume a register service exists
import { Link } from 'react-router-dom';
import loginImage from '../resources/images/7108455 1.png'; // Use the same image or update the path

const Container = styled.div`
  min-height: 99.5vh;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: auto;
`;

const SignupContainer = styled.div`
  display: flex;
  height: 70vh;
  width: 1150px;
  box-sizing: border-box;
  border-radius: 40px;
  background-color: white;
  box-shadow: rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px;
`;

const Column = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const LeftColumn = styled(Column)`
  background-color: white;
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
  justify-content: space-between;
  align-items: center;
  width: 100%;
  max-width: 400px;
  height: 70%;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-sizing: border-box;
`;

const Input = styled.input`
  margin-bottom: 15px;
  padding: 10px;
  width: 100%;
  border-radius: 5px;
  font-size: 14px;
  box-sizing: border-box;
  border: 2px solid gray;
`;

const Button = styled.button`
  padding: 10px;
  width: 60%;
  background-color: #2563eb;
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 16px;
  cursor: pointer;
  box-sizing: border-box;

  &:hover {
    background-color: #3b82f6;
  }
`;

const ErrorMessage = styled.p`
  color: red;
  font-size: 14px;
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

const Signup = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      await register(email, password); // Call the register service
      window.location.href = '/dashboard'; // Redirect after successful signup
    } catch (err) {
      setError('Registration failed. Please try again.');
    }
  };

  return (
    <Container>
      <SignupContainer>
        {/* Left Column: Illustration */}
        <LeftColumn>
          <Illustration>
            <img
              src={loginImage} // Add your image in the public folder
              alt="Signup Illustration"
            />
          </Illustration>
        </LeftColumn>

        {/* Right Column: Signup Form */}
        <RightColumn>
          <Form onSubmit={handleSubmit}>
            <Heading>Sign Up</Heading>
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
            <Input
              type="password"
              placeholder="Confirm Password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
            {error && <ErrorMessage>{error}</ErrorMessage>}
            <Button type="submit">Sign Up</Button>
            <hr />
            <LinkText>
              Already have an account? <Link to="/login">Log in</Link>
            </LinkText>
          </Form>
        </RightColumn>
      </SignupContainer>
    </Container>
  );
};

export default Signup;
