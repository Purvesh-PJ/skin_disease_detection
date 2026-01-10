import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Input, Alert, Spinner } from '../../components/common/ui';
import { H2, SmallText } from '../../styles/typography';
import signupImage from '../../assets/images/7108455 1.png';
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
} from '../Login/styles';

const Signup = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState('');
  const { register, loading, error } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError('');

    if (password !== confirmPassword) {
      setLocalError('Passwords do not match');
      return;
    }

    try {
      await register({ username, email, password });
      navigate(ROUTES.LOGIN);
    } catch {
      // Error is handled by useAuth hook
    }
  };

  const displayError = localError || error;

  return (
    <Container>
      <AuthContainer>
        <LeftColumn>
          <Illustration>
            <img src={signupImage} alt="Signup Illustration" />
          </Illustration>
        </LeftColumn>
        <RightColumn>
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
            {displayError && <Alert variant="error">{displayError}</Alert>}
            <Button type="submit" disabled={loading} style={{ width: '60%' }}>
              {loading ? <Spinner size="sm" color="white" /> : 'Sign Up'}
            </Button>
            <Divider />
            <LinkText>
              Already have an account? <Link to={ROUTES.LOGIN}>Log in</Link>
            </LinkText>
          </Form>
        </RightColumn>
      </AuthContainer>
    </Container>
  );
};

export default Signup;
