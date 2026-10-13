import axiosInstance from './axios';
import { API_ENDPOINTS } from '../../constants';

export const predictionService = {
  async predict(imageData) {
    const response = await axiosInstance.post(API_ENDPOINTS.PREDICT, imageData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 120000, // 120 seconds generous timeout for heavy multi-model ensemble downloads
    });
    return response;
  },
};

export default predictionService;
