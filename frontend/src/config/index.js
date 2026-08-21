// Application configuration following 12-Factor / React environment standards
const isProduction = process.env.NODE_ENV === 'production';

export const config = {
  api: {
    baseUrl: process.env.REACT_APP_API_URL || (isProduction
      ? 'https://skin-disease-detection-6kxm.onrender.com'
      : 'http://localhost:5000'),
    timeout: 60000,
  },
  app: {
    name: 'Skin Disease Predictor',
    version: '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  },
};

export default config;
