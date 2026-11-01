import React from 'react';
import { render, screen } from '@testing-library/react';
import ResultsCard from '../../components/features/prediction/ResultsCard';

describe('ResultsCard Unit Tests', () => {
  test('renders empty state when no prediction result is present', () => {
    render(<ResultsCard predictionResult={null} loading={false} error={null} />);

    expect(screen.getByText(/No Diagnostic Data/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Upload a skin lesion image on the left panel/i)
    ).toBeInTheDocument();
  });

  test('renders loading state when inference is in progress', () => {
    render(<ResultsCard predictionResult={null} loading={true} error={null} />);

    expect(screen.getByText(/Analyzing Skin Lesion.../i)).toBeInTheDocument();
    expect(
      screen.getByText(/Running ensemble deep neural network inference/i)
    ).toBeInTheDocument();
  });

  test('renders error state with custom error message', () => {
    const mockError = { message: 'Cloud container timeout' };
    render(<ResultsCard predictionResult={null} loading={false} error={mockError} />);

    expect(screen.getByText(/Analysis Failed/i)).toBeInTheDocument();
    expect(screen.getByText(/Cloud container timeout/i)).toBeInTheDocument();
  });

  test('renders analysis results with diagnosis, confidence, and models', () => {
    const mockResult = {
      predicted_disease: 'mel',
      confidence: '95',
      disease_details: {
        name: 'Melanoma',
        description: 'A serious malignancy arising from melanocytes.',
      },
      filename: 'lesion_scan_01.jpg',
    };

    render(<ResultsCard predictionResult={mockResult} loading={false} error={null} />);

    // Diagnosis heading
    expect(screen.getByText(/Melanoma/i)).toBeInTheDocument();
    expect(screen.getByText(/95% Match/i)).toBeInTheDocument();

    // Summary description
    expect(
      screen.getByText(/A serious malignancy arising from melanocytes/i)
    ).toBeInTheDocument();

    // Verification ensemble badges
    expect(screen.getByText(/ResNet-101/i)).toBeInTheDocument();
    expect(screen.getByText(/DenseNet-121/i)).toBeInTheDocument();
    expect(screen.getByText(/EfficientNet-B3/i)).toBeInTheDocument();

    // Analyzed filename
    expect(screen.getByText(/lesion_scan_01.jpg/i)).toBeInTheDocument();
  });
});

