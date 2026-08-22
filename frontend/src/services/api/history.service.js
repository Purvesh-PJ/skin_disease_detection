import axiosInstance from './axios';
import { API_ENDPOINTS } from '../../constants';

export const historyService = {
  async getHistory() {
    const response = await axiosInstance.get(API_ENDPOINTS.HISTORY.GET_ALL);
    return response.data?.history || [];
  },

  async deleteHistoryItem(historyId) {
    const response = await axiosInstance.delete(`${API_ENDPOINTS.HISTORY.DELETE}/${historyId}`);
    return response.data;
  },

  async clearHistory() {
    const response = await axiosInstance.delete(API_ENDPOINTS.HISTORY.CLEAR);
    return response.data;
  },
};

export default historyService;
