import React, { useState, useEffect, useCallback } from 'react';
import {
  FiClock,
  FiTrash2,
  FiRefreshCw,
  FiDatabase,
  FiShield,
  FiLayers,
} from 'react-icons/fi';
import { historyService, authService } from '../../../services';
import { Spinner } from '../../ui';

const getCategoryClass = (diseaseCode) => {
  const code = (diseaseCode || '').toLowerCase();
  if (['mel', 'melanoma', 'bcc', 'basal cell carcinoma'].includes(code)) return 'malignant';
  if (['akiec', 'actinic keratoses', 'bowen'].includes(code)) return 'precancerous';
  return 'benign';
};

const formatDate = (isoString) => {
  if (!isoString) return 'Recent';
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return 'Recent';
  }
};

export const HistoryList = ({ refreshTrigger }) => {
  const [history, setHistory] = useState(() => historyService.getLocalRecords());
  const [loading, setLoading] = useState(false);

  const fetchHistory = useCallback(async (isManual = false) => {
    if (!authService.isAuthenticated()) return;

    if (isManual && history.length === 0) {
      setLoading(true);
    }

    try {
      const records = await historyService.getHistory();
      if (Array.isArray(records)) {
        setHistory(records);
      }
    } catch (err) {
      console.warn("Scan history sync notice:", err);
    } finally {
      setLoading(false);
    }
  }, [history.length]);

  useEffect(() => {
    setHistory(historyService.getLocalRecords());
    fetchHistory(false);
  }, [fetchHistory, refreshTrigger]);

  const handleDelete = (id, e) => {
    e.stopPropagation();
    setHistory((prev) => prev.filter((item) => item._id !== id));
    historyService.deleteHistoryItem(id).catch((err) => {
      console.warn("Background delete sync failed:", err);
    });
  };

  const handleClearAll = (e) => {
    if (e) e.stopPropagation();
    if (window.confirm("Are you sure you want to clear all MongoDB scan records for your account?")) {
      setHistory([]);
      historyService.clearHistory().catch((err) => {
        console.warn("Background clear sync failed:", err);
      });
    }
  };

  if (!authService.isAuthenticated()) {
    return (
      <div className="p-6 text-center border border-dashed border-neutral-200 dark:border-neutral-800 rounded-xl bg-neutral-50 dark:bg-[#181818] flex flex-col items-center gap-2">
        <FiShield size={24} className="text-emerald-600 dark:text-emerald-400" />
        <p className="m-0 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
          Guest Session (History Inactive)
        </p>
        <span className="text-xs text-neutral-500 dark:text-neutral-400">
          Sign in or use <strong>⚡ 1-Click Demo Login</strong> to record and save your scans.
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* History Header */}
      <div className="flex items-center justify-between pb-2 border-b border-neutral-200/80 dark:border-neutral-800/80">
        <div className="flex items-center gap-2">
          <FiDatabase className="text-emerald-600 dark:text-emerald-400" size={16} />
          <h4 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 m-0">
            Diagnostic Scan History
          </h4>
          <span className="text-xs font-semibold bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 px-2 py-0.5 rounded-full border border-neutral-200 dark:border-neutral-700">
            {history.length} Scans
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => fetchHistory(true)}
            disabled={loading}
            title="Refresh records"
            className="flex items-center gap-1 px-2 py-1 text-xs font-semibold border border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-800 bg-transparent cursor-pointer transition-colors"
          >
            <FiRefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          {history.length > 0 && (
            <button
              onClick={handleClearAll}
              title="Clear all history"
              className="flex items-center gap-1 px-2 py-1 text-xs font-semibold border border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400 hover:text-red-600 hover:border-red-300 dark:hover:border-red-900/50 hover:bg-red-50 dark:hover:bg-red-950/20 rounded-md bg-transparent cursor-pointer transition-colors"
            >
              <FiTrash2 size={12} />
              Clear
            </button>
          )}
        </div>
      </div>

      {loading && history.length === 0 ? (
        <div className="flex justify-center p-6">
          <Spinner size="md" />
        </div>
      ) : history.length === 0 ? (
        <div className="p-6 text-center border border-dashed border-neutral-200 dark:border-neutral-800 rounded-xl bg-neutral-50 dark:bg-[#181818] flex flex-col items-center gap-2">
          <FiLayers size={24} className="text-emerald-600 dark:text-emerald-400" />
          <p className="m-0 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
            No scans recorded yet
          </p>
          <span className="text-xs text-neutral-500 dark:text-neutral-400 max-w-sm">
            Upload an image or pick a demo benchmark from the bottom rail to record a diagnostic scan!
          </span>
        </div>
      ) : (
        <div className="flex flex-col gap-2 max-h-[480px] overflow-y-auto pr-1">
          {history.map((item) => {
            const category = getCategoryClass(item.predicted_disease);
            return (
              <div
                key={item._id}
                className="bg-neutral-50 dark:bg-[#181818] border border-neutral-200/80 dark:border-neutral-800/80 rounded-xl p-3 flex items-center justify-between gap-3 hover:border-emerald-500/80 hover:bg-white dark:hover:bg-[#1c1c1c] transition-all duration-150"
              >
                <div className="flex flex-col gap-1 flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-sm text-neutral-900 dark:text-neutral-100 capitalize">
                      {item.disease_details?.name || item.predicted_disease}
                    </span>
                    <span
                      className={`text-[0.65rem] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider border ${
                        category === 'malignant'
                          ? 'bg-red-500/15 text-red-500 border-red-500/30'
                          : category === 'precancerous'
                          ? 'bg-amber-500/15 text-amber-500 border-amber-500/30'
                          : 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30'
                      }`}
                    >
                      {category}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-neutral-400 dark:text-neutral-500">
                    <span className="flex items-center gap-1">
                      <FiClock size={11} />
                      {formatDate(item.created_at)}
                    </span>
                    {item.filename && (
                      <span className="truncate max-w-[140px]" title={item.filename}>
                        • {item.filename}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex flex-col items-end gap-0.5">
                  <span className="font-bold text-sm text-emerald-600 dark:text-emerald-400">
                    {item.confidence}%
                  </span>
                  <span className="text-[0.65rem] uppercase text-neutral-400 tracking-wider">
                    Confidence
                  </span>
                </div>

                <button
                  onClick={(e) => handleDelete(item._id, e)}
                  title="Delete scan record"
                  className="p-1.5 rounded-lg text-neutral-400 hover:text-red-500 hover:bg-red-500/10 transition-colors bg-transparent border-none cursor-pointer"
                >
                  <FiTrash2 size={15} />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default HistoryList;
