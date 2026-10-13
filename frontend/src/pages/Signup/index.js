import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks';
import { ROUTES } from '../../constants';
import { Button, Input, Alert, Spinner } from '../../components/ui';
import { AuthLayout } from '../../components/layout';

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
      // Handled in useAuth
    }
  };

  const displayError = localError || error;

  return (
    <AuthLayout>
      {/* Header */}
      <div className="flex flex-col gap-1 text-center mb-2">
        <h2 className="text-2xl font-bold tracking-tight text-neutral-900 dark:text-neutral-100 m-0">
          Create account
        </h2>
        <p className="text-sm text-neutral-500 dark:text-neutral-400 m-0">
          Get started with AI-driven skin lesion classification
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="flex flex-col w-full gap-3">
        <Input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <Input
          type="email"
          placeholder="Email address"
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

        {displayError && <Alert variant="error">{displayError}</Alert>}

        <Button type="submit" disabled={loading} fullWidth size="lg">
          {loading ? <Spinner size="sm" color="white" /> : 'Create Account'}
        </Button>

        <hr className="w-full border-t border-neutral-200 dark:border-neutral-800 my-1" />

        <p className="text-center text-xs text-neutral-500 dark:text-neutral-400 m-0">
          Already have an account?{' '}
          <Link
            to={ROUTES.LOGIN}
            className="text-emerald-600 dark:text-emerald-400 font-semibold hover:underline"
          >
            Log in
          </Link>
        </p>
      </form>
    </AuthLayout>
  );
};

export default Signup;
