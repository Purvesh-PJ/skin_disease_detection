import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiZap, FiArrowRight } from 'react-icons/fi';
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
  DemoCard,
  DemoHeader,
  DemoDesc,
  DemoButton,
  OrDivider,
} from './styles';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [demoLoading, setDemoLoading] = useState(false);
  const { login, demoLogin, loading, error, clearError } = useAuth();
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

  const handleDemoSignIn = async () => {
    setDemoLoading(true);
    if (error) clearError();
    try {
      await demoLogin();
      navigate(ROUTES.DASHBOARD);
    } catch (err) {
      console.error("Demo login error:", err);
    } finally {
      setDemoLoading(false);
    }
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) clearError();
  };

  return (
    <AuthLayout>
      <FormHeader>
        <H2>Welcome Back</H2>
        <Text variant="secondary" size="sm">
          Sign in to analyze skin lesion diagnostics & persist records to MongoDB
        </Text>
      </FormHeader>

      {/* 1-Click Recruiter / Demo Access Box */}
      <DemoCard>
        <DemoHeader>
          <h4>
            <FiZap color="#16a34a" size={16} />
            Recruiter / Evaluator Fast Access
          </h4>
          <span className="badge">Instant</span>
        </DemoHeader>
        <DemoDesc>
          No account creation needed. Click below to instantly log in as a verified evaluator with live MongoDB storage.
        </DemoDesc>
        <DemoButton
          type="button"
          onClick={handleDemoSignIn}
          disabled={demoLoading || loading}
        >
          {demoLoading ? (
            <Spinner size="sm" color="white" />
          ) : (
            <>
              <span>⚡ 1-Click Demo Sign In</span>
              <FiArrowRight size={16} />
            </>
          )}
        </DemoButton>
      </DemoCard>

      <OrDivider>
        <span>Or Sign In With Email</span>
      </OrDivider>

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

        <Button type="submit" disabled={loading || demoLoading} fullWidth size="lg">
          {loading ? <Spinner size="sm" color="white" /> : 'Sign In'}
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

