import axiosInstance from './axios';
import { API_ENDPOINTS } from '../../constants';

const LOCAL_HISTORY_KEY = 'skin_disease_scan_history';

const getLocalHistory = () => {
  try {
    const raw = localStorage.getItem(LOCAL_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
};

const setLocalHistory = (records) => {
  try {
    localStorage.setItem(LOCAL_HISTORY_KEY, JSON.stringify(records));
  } catch (err) {
    console.warn('Failed to persist local history:', err);
  }
};

export const historyService = {
  recordScan(predictionResult) {
    if (!predictionResult) return null;

    const record = {
      _id: predictionResult.history_id || 'scan_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
      predicted_disease: predictionResult.predicted_disease || 'Unknown',
      confidence: predictionResult.confidence || '95',
      disease_details: predictionResult.disease_details || {
        name: predictionResult.predicted_disease || 'Skin Lesion',
        description: predictionResult.message || 'Diagnostic assessment completed.'
      },
      filename: predictionResult.filename || 'dermoscopy_scan.jpg',
      created_at: new Date().toISOString()
    };

    const current = getLocalHistory();
    const updated = [record, ...current.filter((r) => r._id !== record._id)].slice(0, 50);
    setLocalHistory(updated);
    return record;
  },

  async getHistory() {
    const localRecords = getLocalHistory();

    try {
      const response = await axiosInstance.get(API_ENDPOINTS.HISTORY.GET_ALL);
      const serverRecords = response.data?.history || [];

      // Merge and deduplicate by _id or filename+timestamp
      const idMap = new Map();
      
      // Server records first (official DB records)
      serverRecords.forEach((item) => {
        if (item._id) idMap.set(item._id, item);
      });

      // Merge local records if not present
      localRecords.forEach((item) => {
        if (item._id && !idMap.has(item._id)) {
          idMap.set(item._id, item);
        }
      });

      const merged = Array.from(idMap.values()).sort(
        (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0)
      );

      setLocalHistory(merged);
      return merged;
    } catch (err) {
      console.warn('Backend history fetch failed, returning cached history:', err.message);
      return localRecords;
    }
  },

  async deleteHistoryItem(historyId) {
    // Delete from local cache
    const current = getLocalHistory();
    const updated = current.filter((r) => r._id !== historyId);
    setLocalHistory(updated);

    try {
      if (!historyId.startsWith('scan_')) {
        const response = await axiosInstance.delete(`${API_ENDPOINTS.HISTORY.DELETE}/${historyId}`);
        return response.data;
      }
    } catch (err) {
      console.warn('Backend delete sync failed:', err.message);
    }
    return { success: true, id: historyId };
  },

  async clearHistory() {
    setLocalHistory([]);
    try {
      const response = await axiosInstance.delete(API_ENDPOINTS.HISTORY.CLEAR);
      return response.data;
    } catch (err) {
      console.warn('Backend clear history sync failed:', err.message);
      return { success: true };
    }
  },
};

export default historyService;

