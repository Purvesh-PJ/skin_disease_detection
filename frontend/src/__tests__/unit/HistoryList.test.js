import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import HistoryList from '../../components/features/history/HistoryList';
import { historyService } from '../../services/api/history.service';

describe('HistoryList Component Unit Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('renders empty placeholder when history is empty', () => {
    render(<HistoryList refreshTrigger={0} />);

    expect(screen.getByText(/No scans recorded yet/i)).toBeInTheDocument();
    expect(screen.getByText(/0 Scans/i)).toBeInTheDocument();
  });

  test('renders scan entries with clinical details and confidence', () => {
    historyService.recordScan({
      predicted_disease: 'mel',
      confidence: '97',
      disease_details: { name: 'Melanoma' },
      filename: 'lesion_a.jpg',
    });

    historyService.recordScan({
      predicted_disease: 'akiec',
      confidence: '89',
      disease_details: { name: 'Actinic Keratosis' },
      filename: 'lesion_b.jpg',
    });

    render(<HistoryList refreshTrigger={0} />);

    expect(screen.getByText(/Melanoma/i)).toBeInTheDocument();
    expect(screen.getByText(/97%/i)).toBeInTheDocument();
    expect(screen.getByText(/Actinic Keratosis/i)).toBeInTheDocument();
    expect(screen.getByText(/89%/i)).toBeInTheDocument();
    expect(screen.getByText(/2 Scans/i)).toBeInTheDocument();
  });

  test('allows deleting an individual scan record', () => {
    historyService.recordScan({
      predicted_disease: 'bcc',
      confidence: '91',
      disease_details: { name: 'Basal Cell Carcinoma' },
      filename: 'lesion_c.jpg',
    });

    render(<HistoryList refreshTrigger={0} />);

    expect(screen.getByText(/Basal Cell Carcinoma/i)).toBeInTheDocument();

    const deleteBtn = screen.getByTitle(/Delete scan record/i);
    fireEvent.click(deleteBtn);

    expect(screen.queryByText(/Basal Cell Carcinoma/i)).not.toBeInTheDocument();
    expect(screen.getByText(/No scans recorded yet/i)).toBeInTheDocument();
  });

  test('allows clearing all scan records upon confirmation', () => {
    window.confirm = jest.fn(() => true);

    historyService.recordScan({
      predicted_disease: 'nv',
      confidence: '93',
      disease_details: { name: 'Melanocytic Nevus' },
    });

    render(<HistoryList refreshTrigger={0} />);

    const clearBtn = screen.getByTitle(/Clear all history/i);
    fireEvent.click(clearBtn);

    expect(window.confirm).toHaveBeenCalled();
    expect(screen.getByText(/No scans recorded yet/i)).toBeInTheDocument();
  });
});
