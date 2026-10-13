import { useState, useCallback } from 'react';
import { historyService } from '../services';

const urlToFile = async (url, filename) => {
  try {
    const response = await fetch(url);
    if (response.ok) {
      const blob = await response.blob();
      if (blob.type && blob.type.startsWith('image/')) {
        return new File([blob], filename, { type: blob.type });
      }
    }
  } catch (e) {
    console.warn('Canvas fallback triggered for sample image');
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth || 400;
      canvas.height = img.naturalHeight || 300;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(new File([blob], filename, { type: 'image/jpeg' }));
        } else {
          reject(new Error('Canvas conversion error'));
        }
      }, 'image/jpeg', 0.9);
    };
    img.onerror = () => reject(new Error('Sample image failed to load'));
    img.src = url;
  });
};

export const useDashboard = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [activeSample, setActiveSample] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [userUpdateKey, setUserUpdateKey] = useState(0);

  const localRecords = historyService.getLocalRecords();
  const hasScans = Boolean(predictionResult || (localRecords && localRecords.length > 0));

  const handlePredictionSuccess = useCallback((result) => {
    setPredictionResult(result);
    historyService.recordScan(result);
    setActiveTab('analysis');
    setHistoryRefreshKey((prev) => prev + 1);
  }, []);

  const handleUserUpdated = useCallback(() => {
    setUserUpdateKey((prev) => prev + 1);
  }, []);

  const handleSelectSample = useCallback(async (sample) => {
    if (loading) return;
    setActiveSample(sample);
    setSelectedImage(sample.imagePath);
    setError(null);
    setPredictionResult(null);

    try {
      const file = await urlToFile(sample.imagePath, sample.fileName);
      setImageFile(file);
    } catch (err) {
      console.error('Failed to convert sample image:', err);
    }
  }, [loading]);

  return {
    state: {
      selectedImage,
      imageFile,
      activeSample,
      predictionResult,
      loading,
      error,
      activeTab,
      historyRefreshKey,
      userUpdateKey,
      hasScans,
    },
    actions: {
      setSelectedImage,
      setImageFile,
      setActiveSample,
      setLoading,
      setError,
      setActiveTab,
      handlePredictionSuccess,
      handleUserUpdated,
      handleSelectSample,
    },
  };
};

export default useDashboard;
