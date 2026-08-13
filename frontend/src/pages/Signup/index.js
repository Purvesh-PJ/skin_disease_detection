import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Input, Alert, Spinner } from '../../components/common/ui';
import { AuthLayout } from '../../components/layout';
import { H2, Text } from '../../styles/typography';
import {
  FormHeader,
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
      // Error handled by useAuth
    }
  };

  const displayError = localError || error;

  return (
    <AuthLayout>
      <FormHeader>
        <H2>Create account</H2>
        <Text variant="secondary" size="sm">
          Get started with AI-driven skin lesion classification
        </Text>
      </FormHeader>

      <Form onSubmit={handleSubmit}>
        <StyledInput
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <StyledInput
          type="email"
          placeholder="Email address"
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

        <Button type="submit" disabled={loading} fullWidth size="lg">
          {loading ? <Spinner size="sm" color="white" /> : 'Create Account'}
        </Button>

        <Divider />

        <LinkText>
          Already have an account? <Link to={ROUTES.LOGIN}>Log in</Link>
        </LinkText>
      </Form>
    </AuthLayout>
  );
};

export default Signup;
