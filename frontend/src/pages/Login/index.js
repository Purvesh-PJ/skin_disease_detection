import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiZap, FiKey, FiCheck } from 'react-icons/fi';
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
  CredentialRow,
  AutoFillButton,
  OrDivider,
} from './styles';

const DEMO_EMAIL = 'demo@skindisease.ai';
const DEMO_PASSWORD = 'DemoUser@123';

const Login = () => {
  // Pre-fill demo credentials by default for zero-friction evaluator experience
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
        <H2>Sign In to Predictor</H2>
        <Text variant="secondary" size="sm">
          Access dermoscopic AI diagnostics & MongoDB scan history
        </Text>
      </FormHeader>

      {/* Pre-filled Test Account Card */}
      <DemoCard>
        <DemoHeader>
          <h4>
            <FiZap color="#16a34a" size={16} />
            Evaluator / Recruiter Test Account
          </h4>
          <span className="badge">Pre-configured</span>
        </DemoHeader>

        <CredentialRow>
          <div className="item">
            <span className="label">Demo Email:</span>
            <code>{DEMO_EMAIL}</code>
          </div>
          <div className="item">
            <span className="label">Password:</span>
            <code>{DEMO_PASSWORD}</code>
          </div>
        </CredentialRow>

        <AutoFillButton type="button" onClick={handleAutoFill}>
          {filledNotice ? (
            <>
              <FiCheck size={14} color="#16a34a" />
              <span>Credentials Loaded in Form!</span>
            </>
          ) : (
            <>
              <FiKey size={14} />
              <span>Auto-Fill Demo Credentials</span>
            </>
          )}
        </AutoFillButton>
      </DemoCard>

      <OrDivider>
        <span>Account Credentials</span>
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

        <Button type="submit" disabled={loading} fullWidth size="lg">
          {loading ? (
            <Spinner size="sm" color="white" />
          ) : email === DEMO_EMAIL ? (
            'Sign In as Demo Evaluator'
          ) : (
            'Sign In'
          )}
        </Button>

        <Divider />

        <LinkText>
          Want to create a personal account? <Link to={ROUTES.SIGNUP}>Sign up</Link>
        </LinkText>
      </Form>
    </AuthLayout>
  );
};

export default Login;


