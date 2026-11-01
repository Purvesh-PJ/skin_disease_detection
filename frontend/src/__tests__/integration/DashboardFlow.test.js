import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Dashboard } from '../../pages';
import { ThemeProvider } from '../../context/ThemeContext';
import { historyService } from '../../services/api/history.service';

describe('Dashboard Flow Integration Tests', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('renders dropzone, upload controls, and demo samples rail', () => {
    render(
      <MemoryRouter>
        <ThemeProvider>
          <Dashboard />
        </ThemeProvider>
      </MemoryRouter>
    );

    // Dropzone upload prompt
    expect(screen.getByText(/Click to upload/i)).toBeInTheDocument();
    expect(screen.getByText(/or drag lesion image/i)).toBeInTheDocument();

    // Sample images benchmark rail
    expect(screen.getByText(/Demo Samples:/i)).toBeInTheDocument();
  });

  test('history button disabled initially when no scans exist', () => {
    render(
      <MemoryRouter>
        <ThemeProvider>
          <Dashboard />
        </ThemeProvider>
      </MemoryRouter>
    );

    const historyBtn = screen.getByRole('button', { name: /see history/i });
    expect(historyBtn).toBeDisabled();
  });

  test('history tab becomes active and displays saved scan', () => {
    historyService.recordScan({
      predicted_disease: 'mel',
      confidence: '95',
      disease_details: {
        name: 'Melanoma',
        description: 'Malignant condition detected.',
      },
      filename: 'sample_mel.jpg',
    });

    render(
      <MemoryRouter>
        <ThemeProvider>
          <Dashboard />
        </ThemeProvider>
      </MemoryRouter>
    );

    const historyBtn = screen.getByRole('button', { name: /see history/i });
    expect(historyBtn).not.toBeDisabled();

    fireEvent.click(historyBtn);

    expect(screen.getByText(/Melanoma/i)).toBeInTheDocument();
    expect(screen.getByText(/95%/i)).toBeInTheDocument();
    expect(screen.getByText(/1 Scans/i)).toBeInTheDocument();
  });
});
