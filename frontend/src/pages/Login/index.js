import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiZap } from 'react-icons/fi';
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
  DemoNotice,
} from './styles';

const DEMO_EMAIL = 'demo@skindisease.ai';
const DEMO_PASSWORD = 'DemoUser@123';

const Login = () => {
  const [email, setEmail] = useState(DEMO_EMAIL);
  const [password, setPassword] = useState(DEMO_PASSWORD);
  const [filledNotice, setFilledNotice] = useState(false);
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

  const handleAutoFill = () => {
    setEmail(DEMO_EMAIL);
    setPassword(DEMO_PASSWORD);
    if (error) clearError();
    setFilledNotice(true);
    setTimeout(() => setFilledNotice(false), 2000);
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) clearError();
  };

  return (
    <AuthLayout>
      <FormHeader>
        <H2>Sign In</H2>
        <Text variant="secondary" size="sm">
          Access diagnostic screening & scan records
        </Text>
      </FormHeader>

      <DemoNotice>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <FiZap color="#16a34a" size={14} />
          <span>
            <span className="bold">Demo Account:</span> Pre-filled
          </span>
        </div>
        <button type="button" className="fill-btn" onClick={handleAutoFill}>
          {filledNotice ? '✓ Loaded' : 'Auto-Fill'}
        </button>
      </DemoNotice>

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
          {loading ? (
            <Spinner size="sm" color="white" />
          ) : email === DEMO_EMAIL ? (
            'Sign In as Demo'
          ) : (
            'Sign In'
          )}
        </Button>

        <Divider />

        <LinkText>
          Don't have an account? <Link to={ROUTES.SIGNUP}>Sign up</Link>
        </LinkText>
      </Form>
    </AuthLayout>
  );
};

export default Login;



