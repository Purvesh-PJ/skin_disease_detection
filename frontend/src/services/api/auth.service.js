import axiosInstance from './axios';
import { API_ENDPOINTS, STORAGE_KEYS, ROUTES } from '../../constants';

const DEMO_USER_PROFILE = {
  username: 'Demo User',
  email: 'demo@skindisease.ai',
  roles: ['demo', 'user'],
  settings: {
    full_name: 'Demo',
    role_title: 'Demo',
    specialization: 'Dermatology AI',
    theme: 'dark',
    email_notifications: true
  }
};

export const authService = {
  async login(email, password) {
    const isDemoAccount = email.trim().toLowerCase() === 'demo@skindisease.ai';
    
    try {
      const response = await axiosInstance.post(API_ENDPOINTS.AUTH.LOGIN, { email, password });
      
      if (response.data?.token) {
        localStorage.setItem(STORAGE_KEYS.TOKEN, response.data.token);
        if (response.data.user) {
          localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
        }
      }
      return response.data;
    } catch (err) {
      // If demo user encounters backend cold-start or temporary network failure, activate instant demo session
      if (isDemoAccount) {
        console.warn('Backend connection pending or slow, activating instant Demo session');
        const fallbackToken = 'demo-session-' + Date.now();
        localStorage.setItem(STORAGE_KEYS.TOKEN, fallbackToken);
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(DEMO_USER_PROFILE));
        return {
          token: fallbackToken,
          user: DEMO_USER_PROFILE,
          message: 'Welcome Demo'
        };
      }
      throw err;
    }
  },

  async demoLogin() {
    try {
      const response = await axiosInstance.post(API_ENDPOINTS.AUTH.DEMO_LOGIN);
      if (response.data?.token) {
        localStorage.setItem(STORAGE_KEYS.TOKEN, response.data.token);
        if (response.data.user) {
          localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
        }
      }
      return response.data;
    } catch (err) {
      console.warn('Backend connection pending or slow, activating instant Demo session');
      const fallbackToken = 'demo-session-' + Date.now();
      localStorage.setItem(STORAGE_KEYS.TOKEN, fallbackToken);
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(DEMO_USER_PROFILE));
      return {
        token: fallbackToken,
        user: DEMO_USER_PROFILE,
        message: 'Welcome Demo'
      };
    }
  },


  async register(userData) {
    const response = await axiosInstance.post(API_ENDPOINTS.AUTH.REGISTER, userData);
    if (response.data?.token) {
      localStorage.setItem(STORAGE_KEYS.TOKEN, response.data.token);
      if (response.data.user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
      }
    }
    return response.data;
  },

  async verifyToken() {
    const token = this.getToken();
    if (!token) throw new Error('No token found');

    if (token.startsWith('demo-session-')) {
      const cached = this.getUser() || DEMO_USER_PROFILE;
      return { user: cached, message: 'Demo token valid' };
    }

    try {
      const response = await axiosInstance.get(API_ENDPOINTS.AUTH.VERIFY);
      if (response.data?.user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
      }
      return response.data;
    } catch (err) {
      if (token.startsWith('demo-session-')) {
        return { user: DEMO_USER_PROFILE };
      }
      throw err;
    }
  },

  async getProfile() {
    const token = this.getToken();
    if (token && token.startsWith('demo-session-')) {
      return this.getUser() || DEMO_USER_PROFILE;
    }

    try {
      const response = await axiosInstance.get(API_ENDPOINTS.AUTH.PROFILE);
      if (response.data?.user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
      }
      return response.data?.user;
    } catch (err) {
      return this.getUser() || DEMO_USER_PROFILE;
    }
  },

  async updateProfile(settingsData) {
    const token = this.getToken();
    if (token && token.startsWith('demo-session-')) {
      const current = this.getUser() || DEMO_USER_PROFILE;
      const updated = {
        ...current,
        username: settingsData.full_name || current.username,
        settings: {
          ...current.settings,
          ...settingsData,
        }
      };
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(updated));
      return { user: updated, message: 'Settings saved' };
    }

    const response = await axiosInstance.put(API_ENDPOINTS.AUTH.PROFILE, settingsData);
    if (response.data?.user) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
    }
    return response.data;
  },

  logout() {
    localStorage.removeItem(STORAGE_KEYS.TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER);
    window.location.href = ROUTES.LOGIN;
  },

  getToken() {
    return localStorage.getItem(STORAGE_KEYS.TOKEN);
  },

  getUser() {
    const user = localStorage.getItem(STORAGE_KEYS.USER);
    return user ? JSON.parse(user) : null;
  },

  isAuthenticated() {
    return !!this.getToken();
  },
};

export default authService;


