import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ImageUploadCard from '../../components/features/prediction/ImageUploadCard';

describe('ImageUploadCard Component Unit Tests', () => {
  const defaultProps = {
    selectedImage: null,
    setSelectedImage: jest.fn(),
    imageFile: null,
    setImageFile: jest.fn(),
    setPredictionResult: jest.fn(),
    activeSample: null,
    setActiveSample: jest.fn(),
    loading: false,
    setLoading: jest.fn(),
    setError: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders dropzone with upload prompt when no image is selected', () => {
    render(<ImageUploadCard {...defaultProps} />);

    expect(screen.getByText(/Click to upload/i)).toBeInTheDocument();
    expect(screen.getByText(/or drag lesion image/i)).toBeInTheDocument();
    expect(screen.getByText(/PNG, JPG, or JPEG/i)).toBeInTheDocument();
  });

  test('renders lesion preview image when selectedImage prop is provided', () => {
    render(
      <ImageUploadCard
        {...defaultProps}
        selectedImage="blob:http://localhost/fake-image-preview"
      />
    );

    const img = screen.getByAltText(/Lesion Preview/i);
    expect(img).toBeInTheDocument();
    expect(img).toHaveAttribute('src', 'blob:http://localhost/fake-image-preview');
  });

  test('renders active sample badge when activeSample is present', () => {
    const sample = {
      name: 'Melanoma',
      code: 'mel',
    };

    render(
      <ImageUploadCard
        {...defaultProps}
        selectedImage="blob:http://localhost/sample-mel"
        activeSample={sample}
      />
    );

    expect(screen.getByText(/Melanoma \(mel\)/i)).toBeInTheDocument();
  });
});

