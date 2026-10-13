import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authService } from '../services';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => authService.getUser());
  const [token, setToken] = useState(() => authService.getToken());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const isAuthenticated = Boolean(token);

  useEffect(() => {
    // Synchronize session on mount if token exists
    if (token && !token.startsWith('demo-session-')) {
      authService.verifyToken()
        .then((res) => {
          if (res?.user) setUser(res.user);
        })
        .catch((err) => {
          if (err.response?.status === 401) {
            authService.logout();
            setToken(null);
            setUser(null);
          }
        });
    }
  }, [token]);

  const login = useCallback(async (email, password) => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.login(email, password);
      setToken(response.token);
      setUser(response.user);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Login failed';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const demoLogin = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.demoLogin();
      setToken(response.token);
      setUser(response.user);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Demo login failed';
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
      setToken(response.token);
      setUser(response.user);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Registration failed';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const updateProfile = useCallback(async (settingsData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await authService.updateProfile(settingsData);
      const updatedUser = response?.user || authService.getUser();
      setUser(updatedUser);
      return response;
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Failed to update settings';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    authService.logout();
    setToken(null);
    setUser(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    error,
    login,
    demoLogin,
    register,
    updateProfile,
    logout,
    clearError,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
