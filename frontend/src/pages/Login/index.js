import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Alert, Spinner } from '../../components/common/ui';
import { AuthLayout } from '../../components/layout';
import { H2, Text } from '../../styles/typography';
import {
  FormHeader,
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
      // Error handled by useAuth
    }
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) clearError();
  };

  return (
    <AuthLayout>
      <FormHeader>
        <H2>Welcome back</H2>
        <Text variant="secondary" size="sm">
          Sign in to your account to analyze skin lesion diagnostics
        </Text>
      </FormHeader>

      <Form onSubmit={handleSubmit}>
        <StyledInput
          type="email"
          placeholder="Email address"
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

        <Button type="submit" disabled={loading} fullWidth size="lg">
          {loading ? <Spinner size="sm" color="white" /> : 'Sign In'}
        </Button>

        <LinkText>
          <Link to="/forgot-password">Forgot password?</Link>
        </LinkText>

        <Divider />

        <LinkText>
          Don't have an account? <Link to={ROUTES.SIGNUP}>Sign up</Link>
        </LinkText>
      </Form>
    </AuthLayout>
  );
};

export default Login;
