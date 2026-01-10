import { useState, useCallback } from 'react';
import { authService } from '../services';

const useAuth = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const login = useCallback(async (email, password) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await authService.login(email, password);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || 'Login failed';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (userData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await authService.register(userData);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || 'Registration failed';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    authService.logout();
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    login,
    register,
    logout,
    loading,
    error,
    clearError,
    isAuthenticated: authService.isAuthenticated(),
    user: authService.getUser(),
  };
};

export default useAuth;
