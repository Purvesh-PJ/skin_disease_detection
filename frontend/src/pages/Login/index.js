import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FiZap } from 'react-icons/fi';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Input, Alert, Spinner } from '../../components/ui';
import { AuthLayout } from '../../components/layout';

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
      // Handled in useAuth
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
      {/* Header */}
      <div className="flex flex-col gap-1 text-center mb-2">
        <h2 className="text-2xl font-bold tracking-tight text-neutral-900 dark:text-neutral-100 m-0">
          Sign In
        </h2>
        <p className="text-sm text-neutral-500 dark:text-neutral-400 m-0">
          Access diagnostic screening & scan records
        </p>
      </div>

      {/* Demo Account Box */}
      <div className="flex items-center justify-between px-3 py-2 bg-emerald-50/80 dark:bg-emerald-950/20 border border-emerald-200/80 dark:border-emerald-800/40 rounded-lg text-xs text-neutral-700 dark:text-neutral-300 mb-1">
        <div className="flex items-center gap-1.5">
          <FiZap className="text-emerald-600 dark:text-emerald-400 shrink-0" size={14} />
          <span>
            <strong className="font-bold text-emerald-700 dark:text-emerald-400">Demo Account:</strong> Pre-filled
          </span>
        </div>
        <button
          type="button"
          onClick={handleAutoFill}
          className="font-bold text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 underline cursor-pointer bg-transparent border-none p-0 text-xs"
        >
          {filledNotice ? '✓ Loaded' : 'Auto-Fill'}
        </button>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="flex flex-col w-full gap-3">
        <Input
          type="email"
          placeholder="Email address"
          value={email}
          onChange={handleInputChange(setEmail)}
          error={!!error}
          required
        />
        <Input
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

        <hr className="w-full border-t border-neutral-200 dark:border-neutral-800 my-1" />

        <p className="text-center text-xs text-neutral-500 dark:text-neutral-400 m-0">
          Don't have an account?{' '}
          <Link
            to={ROUTES.SIGNUP}
            className="text-emerald-600 dark:text-emerald-400 font-semibold hover:underline"
          >
            Sign up
          </Link>
        </p>
      </form>
    </AuthLayout>
  );
};

export default Login;
