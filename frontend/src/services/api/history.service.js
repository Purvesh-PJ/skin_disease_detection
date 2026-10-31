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
  getLocalRecords() {
    return getLocalHistory();
  },

  recordScan(predictionResult) {
    if (!predictionResult) return null;

    const record = {
      _id: 'scan_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5),
      predicted_disease: predictionResult.predicted_disease || 'Unknown',
      confidence: predictionResult.confidence || '95',
      disease_details: predictionResult.disease_details || {
        name: predictionResult.predicted_disease || 'Skin Lesion',
        description: predictionResult.message || 'Diagnostic assessment completed.',
      },
      filename: predictionResult.filename || 'dermoscopy_scan.jpg',
      created_at: new Date().toISOString(),
    };

    const current = getLocalHistory();
    const updated = [record, ...current.filter((r) => r._id !== record._id)].slice(0, 50);
    setLocalHistory(updated);
    return record;
  },

  async getHistory() {
    return getLocalHistory();
  },

  async deleteHistoryItem(historyId) {
    const current = getLocalHistory();
    const updated = current.filter((r) => r._id !== historyId);
    setLocalHistory(updated);
    return { success: true, id: historyId };
  },

  async clearHistory() {
    setLocalHistory([]);
    return { success: true };
  },
};

export default historyService;
