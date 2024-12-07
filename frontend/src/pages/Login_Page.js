import React, { useState } from 'react';
import styled from 'styled-components';
import { login } from '../services/authServices';
import { Link } from 'react-router-dom';
import loginImage from '../resources/images/7108455 1.png';

const Container = styled.div`
  min-height: 99.5vh;
  // border: 2px solid blue;
  display : flex;
  justify-content : center;
  align-items : center;
  margin : auto;
`;

const LoginContainer = styled.div`
  display: flex;
  height: 70vh;
  width : 1150px;
  box-sizing: border-box;
  border-radius : 40px;
  box-shadow: rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px;
  // border : 1px solid gray;
  background-color : white ;
`;

const Column = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const LeftColumn = styled(Column)`
  background-color : white;
  border-radius : 40px;
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
  border-radius : 40px;
  // border : 1px solid gray;
`;

const Form = styled.form`
  display : flex;
  flex-direction : column;
  justify-content : space-between;
  align-items : center;
  width: 100%;
  max-width: 400px;
  height : 70%;
  padding: 20px;
  background: white;
  border-radius: 8px;
  box-sizing : border-box;
  // border : 1px solid gray;
`;

const Input = styled.input`
  margin-bottom: 15px;
  padding: 10px;
  width: 100%;
  border-radius: 5px;
  font-size: 14px;
  box-sizing : border-box;
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
  box-sizing : border-box;
  // margin-left : auto;
  // margin-right : auto;

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

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(email, password);
      // window.location.href = '/dashboard'; // Redirect after successful login
    } 
    catch (err) {
      setError('Invalid email or password');
    }
  };

  return (
    <Container>
      <LoginContainer>
        {/* Left Column: Illustration */}
        <LeftColumn>
          <Illustration>
            <img
              src={loginImage} // Add your image in the public folder
              alt="Skin Disease Illustration"
            />
          </Illustration>
        </LeftColumn>

        {/* Right Column: Login Form */}
        <RightColumn>
          <Form onSubmit={handleSubmit}>
            <Heading>Login</Heading>
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
