// Application configuration
export const config = {
  api: {
    baseUrl: process.env.REACT_APP_API_URL || 'https://skin-disease-detection-6kxm.onrender.com',
    timeout: 60000,
  },

  app: {
    name: 'Skin Disease Predictor',
    version: '1.0.0',
  },
};

export default config;
