import axiosInstance from './axios';
import { API_ENDPOINTS, STORAGE_KEYS, ROUTES } from '../../constants';

export const authService = {
  async login(email, password) {
    const response = await axiosInstance.post(API_ENDPOINTS.AUTH.LOGIN, { email, password });
    
    if (response.data?.token) {
      localStorage.setItem(STORAGE_KEYS.TOKEN, response.data.token);
      if (response.data.user) {
        localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(response.data.user));
      }
    }
    return response.data;
  },

  async register(userData) {
    const response = await axiosInstance.post(API_ENDPOINTS.AUTH.REGISTER, userData);
    return response.data;
  },

  async verifyToken() {
    const token = this.getToken();
    if (!token) throw new Error('No token found');

    const response = await axiosInstance.get(API_ENDPOINTS.AUTH.VERIFY);
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
