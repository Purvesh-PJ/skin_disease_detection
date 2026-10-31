import axios from 'axios';
import { config } from '../../config';

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: config.api.baseUrl,
  timeout: 30000, // 30 seconds timeout to allow for first cold start or complex image inference
  headers: {
    'Content-Type': 'application/json',
  },
});

export default axiosInstance;
