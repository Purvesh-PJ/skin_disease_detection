// Application constants
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  SIGNUP: '/signup',
  DASHBOARD: '/dashboard',
};

export const STORAGE_KEYS = {
  TOKEN: 'token',
  USER: 'user',
  THEME: 'theme',
};

export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    DEMO_LOGIN: '/auth/demo-login',
    REGISTER: '/auth/register',
    VERIFY: '/auth/verify-token',
    PROFILE: '/auth/profile',
  },
  PREDICT: '/predict',
  HISTORY: {
    GET_ALL: '/history',
    DELETE: '/history',
    CLEAR: '/history/clear',
  },
};

export * from './sampleImages';


