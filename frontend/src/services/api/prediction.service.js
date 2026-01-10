import axiosInstance from './axios';
import { API_ENDPOINTS } from '../../constants';

export const predictionService = {
  async predict(imageData) {
    const response = await axiosInstance.post(API_ENDPOINTS.PREDICT, imageData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  },
};

export default predictionService;
