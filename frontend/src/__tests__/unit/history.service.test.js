import { historyService } from '../../services/api/history.service';

describe('historyService Unit Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('getLocalRecords returns empty array initially', () => {
    const records = historyService.getLocalRecords();
    expect(records).toEqual([]);
  });

  test('recordScan saves a valid scan record into localStorage', () => {
    const mockPrediction = {
      predicted_disease: 'mel',
      confidence: '95',
      disease_details: {
        name: 'Melanoma',
        description: 'Malignant melanoma assessment',
      },
      filename: 'test_scan.jpg',
    };

    const record = historyService.recordScan(mockPrediction);

    expect(record).toBeDefined();
    expect(record.predicted_disease).toBe('mel');
    expect(record.confidence).toBe('95');
    expect(record._id).toMatch(/^scan_/);

    const savedRecords = historyService.getLocalRecords();
    expect(savedRecords).toHaveLength(1);
    expect(savedRecords[0].predicted_disease).toBe('mel');
  });

  test('recordScan ignores null/undefined input', () => {
    const result = historyService.recordScan(null);
    expect(result).toBeNull();
    expect(historyService.getLocalRecords()).toHaveLength(0);
  });

  test('recordScan limits history to maximum 50 records', () => {
    for (let i = 0; i < 60; i++) {
      historyService.recordScan({
        predicted_disease: `disease_${i}`,
        confidence: '90',
        filename: `scan_${i}.jpg`,
      });
    }

    const saved = historyService.getLocalRecords();
    expect(saved).toHaveLength(50);
  });

  test('deleteHistoryItem removes specific item by ID', async () => {
    const item1 = historyService.recordScan({ predicted_disease: 'bcc', confidence: '92' });
    const item2 = historyService.recordScan({ predicted_disease: 'nv', confidence: '88' });

    expect(historyService.getLocalRecords()).toHaveLength(2);

    await historyService.deleteHistoryItem(item1._id);

    const remaining = historyService.getLocalRecords();
    expect(remaining).toHaveLength(1);
    expect(remaining[0]._id).toBe(item2._id);
  });

  test('clearHistory empties all scan history records', async () => {
    historyService.recordScan({ predicted_disease: 'mel', confidence: '96' });
    historyService.recordScan({ predicted_disease: 'bkl', confidence: '85' });

    expect(historyService.getLocalRecords()).toHaveLength(2);

    await historyService.clearHistory();

    expect(historyService.getLocalRecords()).toHaveLength(0);
  });
});

