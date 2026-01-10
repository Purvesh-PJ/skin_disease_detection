import { useState, useCallback } from 'react';
import { predictionService } from '../services';

const usePrediction = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async (imageData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await predictionService.predict(imageData);
      return response;
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setError(null);
  }, []);

  return {
    predict,
    loading,
    error,
    reset,
  };
};

export default usePrediction;
