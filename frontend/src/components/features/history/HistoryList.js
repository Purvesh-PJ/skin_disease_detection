import React, { useState, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import {
  FiClock,
  FiTrash2,
  FiRefreshCw,
  FiDatabase,
  FiShield,
  FiLayers
} from 'react-icons/fi';
import { historyService, authService } from '../../../services';
import { Spinner } from '../../common/ui';


const HistoryContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  width: 100%;
`;

const HistoryHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: ${({ theme }) => theme.spacing[2]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;

    h4 {
      font-size: 0.95rem;
      font-weight: 700;
      margin: 0;
      color: ${({ theme }) => theme.colors.text.primary};
    }

    span.badge {
      font-size: 0.72rem;
      font-weight: 600;
      background: ${({ theme }) => theme.colors.background.tertiary};
      color: ${({ theme }) => theme.colors.text.secondary};
      padding: 2px 8px;
      border-radius: 9999px;
      border: 1px solid ${({ theme }) => theme.colors.border.default};
    }
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }
`;

const HeaderActionButton = styled.button`
  background: transparent;
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  color: ${({ theme }) => theme.colors.text.secondary};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: 4px 8px;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    color: ${({ theme }) => theme.colors.text.primary};
    background: ${({ theme }) => theme.colors.background.tertiary};
    border-color: ${({ theme }) => theme.colors.border.strong};
  }

  &.danger:hover:not(:disabled) {
    color: #ef4444;
    border-color: #f87171;
    background: rgba(239, 68, 68, 0.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const RecordsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 480px;
  overflow-y: auto;
  padding-right: 4px;

  &::-webkit-scrollbar {
    width: 6px;
  }
  &::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.default};
    border-radius: 4px;
  }
`;

const RecordCard = styled.div`
  background: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[500]};
    background: ${({ theme }) => theme.colors.background.tertiary};
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  }
`;

const RecordInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;

  .top-line {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;

    span.disease-name {
      font-weight: 700;
      font-size: 0.92rem;
      color: ${({ theme }) => theme.colors.text.primary};
      text-transform: capitalize;
    }

    span.severity-badge {
      font-size: 0.7rem;
      font-weight: 700;
      padding: 1px 7px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.04em;

      &.malignant {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
      }
      &.precancerous {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
      }
      &.benign {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
      }
    }
  }

  .meta-line {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.78rem;
    color: ${({ theme }) => theme.colors.text.tertiary};

    span.date {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    span.file {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 140px;
    }
  }
`;

const ConfidencePill = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;

  span.score {
    font-weight: 700;
    font-size: 0.95rem;
    color: ${({ theme }) => theme.colors.primary[600] || '#16a34a'};
  }

  span.label {
    font-size: 0.68rem;
    color: ${({ theme }) => theme.colors.text.tertiary};
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
`;

const DeleteButton = styled.button`
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.colors.text.tertiary};
  cursor: pointer;
  padding: 6px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
  }
`;

const EmptyState = styled.div`
  padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[4]}`};
  text-align: center;
  border: 1px dashed ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  background: ${({ theme }) => theme.colors.background.tertiary};
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;

  p {
    margin: 0;
    font-size: 0.88rem;
    color: ${({ theme }) => theme.colors.text.secondary};
  }

  span.subtext {
    font-size: 0.78rem;
    color: ${({ theme }) => theme.colors.text.tertiary};
  }
`;

const MongoBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};
  padding: 6px 10px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  margin-top: 4px;
`;

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
      minute: '2-digit'
    });
  } catch {
    return 'Recent';
  }
};

export const HistoryList = ({ refreshTrigger }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchHistory = useCallback(async () => {
    if (!authService.isAuthenticated()) return;
    setLoading(true);
    setError(null);
    try {
      const records = await historyService.getHistory();
      setHistory(records);
    } catch (err) {
      console.warn("Could not load history from MongoDB:", err);
      // Non-blocking error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory, refreshTrigger]);

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    try {
      await historyService.deleteHistoryItem(id);
      setHistory((prev) => prev.filter((item) => item._id !== id));
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const handleClearAll = async () => {
    if (window.confirm("Are you sure you want to clear all MongoDB scan records for your account?")) {
      try {
        await historyService.clearHistory();
        setHistory([]);
      } catch (err) {
        console.error("Clear all failed:", err);
      }
    }
  };

  if (!authService.isAuthenticated()) {
    return (
      <EmptyState>
        <FiShield size={24} color="#16a34a" />
        <p>Guest Session (History Storage Inactive)</p>
        <span className="subtext">
          Sign in or use <strong>⚡ 1-Click Demo Login</strong> to auto-record your scans in MongoDB Atlas.
        </span>
      </EmptyState>
    );
  }

  return (
    <HistoryContainer>
      <HistoryHeader>
        <div className="header-left">
          <FiDatabase color="#16a34a" size={16} />
          <h4>MongoDB Scan Records</h4>
          <span className="badge">{history.length} Saved</span>
        </div>

        <div className="header-actions">
          <HeaderActionButton onClick={fetchHistory} disabled={loading} title="Refresh records">
            <FiRefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            Refresh
          </HeaderActionButton>
          {history.length > 0 && (
            <HeaderActionButton className="danger" onClick={handleClearAll} title="Clear all history">
              <FiTrash2 size={12} />
              Clear
            </HeaderActionButton>
          )}
        </div>
      </HistoryHeader>

      {loading && history.length === 0 ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '24px' }}>
          <Spinner size="md" />
        </div>
      ) : history.length === 0 ? (
        <EmptyState>
          <FiLayers size={24} color="#16a34a" />
          <p>No scans stored in MongoDB yet</p>
          <span className="subtext">Upload an image or pick a demo benchmark above to record a diagnostic scan!</span>
        </EmptyState>
      ) : (
        <RecordsList>
          {history.map((item) => {
            const category = getCategoryClass(item.predicted_disease);
            return (
              <RecordCard key={item._id}>
                <RecordInfo>
                  <div className="top-line">
                    <span className="disease-name">
                      {item.disease_details?.name || item.predicted_disease}
                    </span>
                    <span className={`severity-badge ${category}`}>
                      {category}
                    </span>
                  </div>
                  <div className="meta-line">
                    <span className="date">
                      <FiClock size={11} />
                      {formatDate(item.created_at)}
                    </span>
                    {item.filename && (
                      <span className="file" title={item.filename}>
                        • {item.filename}
                      </span>
                    )}
                  </div>
                </RecordInfo>

                <ConfidencePill>
                  <span className="score">{item.confidence}%</span>
                  <span className="label">Confidence</span>
                </ConfidencePill>

                <DeleteButton onClick={(e) => handleDelete(item._id, e)} title="Delete scan record">
                  <FiTrash2 size={15} />
                </DeleteButton>
              </RecordCard>
            );
          })}
        </RecordsList>
      )}

      <MongoBadge>
        <FiDatabase size={13} color="#16a34a" />
        <span>Connected to live MongoDB collection: <code>prediction_history</code></span>
      </MongoBadge>
    </HistoryContainer>
  );
};

export default HistoryList;
