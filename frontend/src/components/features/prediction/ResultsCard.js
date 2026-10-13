import React from 'react';
import { FiAlertCircle, FiCheckCircle, FiShield, FiFileText } from 'react-icons/fi';
import { Spinner, Badge } from '../../ui';

const ResultsCard = ({ predictionResult, loading, error }) => {
  const getDiseaseName = (result) => {
    return result.disease_details?.name || result.predicted_disease;
  };

  const getDescription = (result) => {
    return result.disease_details?.description || null;
  };

  const confidenceValue = parseInt(predictionResult?.confidence || '0', 10);

  if (loading) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-center p-6">
        <Spinner size="lg" />
        <h4 className="mt-5 font-semibold text-neutral-700 dark:text-neutral-300 text-base m-0">
          Analyzing Skin Lesion...
        </h4>
        <p className="mt-1 text-xs text-neutral-400 dark:text-neutral-500 m-0">
          Running ensemble deep neural network inference
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center text-center p-6">
        <div className="w-16 h-16 rounded-full flex items-center justify-center bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 mb-4 shadow-sm">
          <FiAlertCircle size={32} />
        </div>
        <h4 className="font-semibold text-neutral-800 dark:text-neutral-200 text-lg m-0">
          Analysis Failed
        </h4>
        <p className="mt-1.5 text-xs text-neutral-500 dark:text-neutral-400 max-w-xs m-0 leading-relaxed">
          {error.response?.data?.error ||
            error.response?.data?.message ||
            error.message ||
            'Unable to process image. Please verify backend connectivity and try again.'}
        </p>
      </div>
    );
  }

  if (predictionResult) {
    return (
      <div className="bg-neutral-50 dark:bg-[#181818] border border-neutral-200/80 dark:border-neutral-800/80 rounded-2xl p-6 flex-1 overflow-y-auto flex flex-col gap-4 animate-in fade-in duration-300">
        {/* Result Header */}
        <div className="flex items-center gap-3 pb-4 border-b border-neutral-200 dark:border-neutral-800">
          <div className="w-11 h-11 rounded-full flex items-center justify-center bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-600 dark:text-emerald-400 shrink-0">
            <FiCheckCircle size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-neutral-900 dark:text-neutral-100 m-0 tracking-tight">
              {getDiseaseName(predictionResult)}
            </h3>
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              AI Diagnosis Output
            </span>
          </div>
          <Badge variant="medical" className="ml-auto text-sm font-bold px-3 py-1">
            {confidenceValue}% Match
          </Badge>
        </div>

        {/* Progress Bar Section */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-neutral-600 dark:text-neutral-400">
              Ensemble Confidence Score
            </span>
            <span className="font-bold text-emerald-600 dark:text-emerald-400">
              {confidenceValue}%
            </span>
          </div>
          <div className="w-full h-2 bg-neutral-200 dark:bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full transition-all duration-700 ease-out"
              style={{ width: `${confidenceValue}%` }}
            />
          </div>
        </div>

        {/* Condition Summary */}
        {getDescription(predictionResult) && (
          <div className="p-4 bg-white dark:bg-[#121212] rounded-xl border border-neutral-200/80 dark:border-neutral-800/80">
            <div className="flex items-center gap-2 text-xs font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
              <FiFileText size={16} className="text-emerald-600 dark:text-emerald-400" />
              <span>Condition Summary & Guidance</span>
            </div>
            <p className="text-sm text-neutral-600 dark:text-neutral-300 leading-relaxed m-0">
              {getDescription(predictionResult)}
            </p>
          </div>
        )}

        {/* Metadata Grid */}
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(predictionResult)
            .filter(([key]) => !['predicted_disease', 'confidence', 'disease_details', 'message'].includes(key))
            .map(([key, value]) => (
              <div
                key={key}
                className="p-3 bg-white dark:bg-[#121212] rounded-xl border border-neutral-200/80 dark:border-neutral-800/80"
              >
                <div className="text-[0.7rem] uppercase font-bold text-neutral-400 dark:text-neutral-500 mb-1 tracking-wider">
                  {key.split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </div>
                <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100 truncate">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </div>
              </div>
            ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col items-center justify-center text-center p-6">
      <div className="w-16 h-16 rounded-full flex items-center justify-center bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200/60 dark:border-emerald-800/30 text-emerald-600 dark:text-emerald-400 mb-4 shadow-sm">
        <FiShield size={32} />
      </div>
      <h4 className="font-semibold text-neutral-800 dark:text-neutral-200 text-lg m-0">
        No Diagnostic Data
      </h4>
      <p className="mt-1.5 text-xs text-neutral-500 dark:text-neutral-400 max-w-xs m-0 leading-relaxed">
        Upload a skin lesion image on the left panel to execute multi-model ensemble analysis.
      </p>
    </div>
  );
};

export default ResultsCard;
