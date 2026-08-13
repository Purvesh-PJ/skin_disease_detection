// Application configuration
export const config = {
  api: {
    baseUrl: process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000',
    timeout: 60000,
  },
  app: {
    name: 'Skin Disease Predictor',
    version: '1.0.0',
  },
};

export default config;
