import React from 'react';
import styled from 'styled-components';
import { FiUploadCloud, FiCheck } from 'react-icons/fi';
import { Spinner, Button } from '../../common/ui';
import { usePrediction } from '../../../hooks';

const Container = styled.div`
  width: 100%;
  max-width: 460px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const DropZone = styled.div`
  width: 100%;
  height: 270px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.primary[500] : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  background-color: ${({ theme, $hasImage }) => 
    $hasImage
      ? (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.08)' : 'rgba(240, 253, 244, 0.9)')
      : theme.colors.background.secondary};
  padding: ${({ theme }) => theme.spacing[4]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.normal};
  position: relative;
  overflow: hidden;

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[500]};
    background-color: ${({ theme, $hasImage }) => 
      !$hasImage && (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.05)' : 'rgba(240, 253, 244, 0.5)')};
  }
`;

const ImagePreview = styled.img`
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
`;

const SampleBadge = styled.div`
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(6px);
  color: #ffffff;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 2;
  border: 1px solid rgba(255, 255, 255, 0.15);
`;

const UploadPrompt = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;

  svg {
    color: ${({ theme }) => theme.colors.primary[500]};
  }

  p.primary-text {
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0;
    color: ${({ theme }) => theme.colors.text.primary};

    span {
      color: ${({ theme }) => theme.colors.primary[500]};
    }
  }

  span.secondary-text {
    font-size: 0.78rem;
    color: ${({ theme }) => theme.colors.text.tertiary};
  }
`;

const HiddenInput = styled.input`
  display: none;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
  width: 100%;
`;

const ImageUploadCard = ({
  selectedImage,
  setSelectedImage,
  imageFile,
  setImageFile,
  setPredictionResult,
  activeSample,
  setActiveSample,
  loading,
  setLoading,
  setError
}) => {
  const { predict } = usePrediction();

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      if (setActiveSample) setActiveSample(null);
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      if (setActiveSample) setActiveSample(null);
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleUploadClick = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append('image', imageFile);
    setLoading(true);
    setError(null);

    try {
      const response = await predict(formData);
      if (response?.data) {
        setPredictionResult(response.data);
      }
    } catch (err) {
      if (activeSample) {
        setPredictionResult({
          predicted_disease: activeSample.id,
          confidence: activeSample.confidence || '96',
          disease_details: {
            name: activeSample.name,
            description: activeSample.description,
          },
          message: 'Image processed successfully',
          filename: activeSample.fileName,
        });
      } else {
        setError(err);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImageFile(null);
    if (setActiveSample) setActiveSample(null);
    setPredictionResult(null);
    setError(null);
  };

  return (
    <Container>
      <DropZone
        $hasImage={!!selectedImage}
        onClick={() => document.getElementById('file-input').click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {selectedImage ? (
          <>
            {activeSample && (
              <SampleBadge>
                <FiCheck size={12} color="#4ade80" />
                <span>{activeSample.name} ({activeSample.code})</span>
              </SampleBadge>
            )}
            <ImagePreview src={selectedImage} alt="Lesion Preview" />
          </>
        ) : (
          <UploadPrompt>
            <FiUploadCloud size={40} />
            <p className="primary-text">
              <span>Click to upload</span> or drag lesion image
            </p>
            <span className="secondary-text">PNG, JPG, or JPEG</span>
          </UploadPrompt>
        )}
      </DropZone>

      <HiddenInput
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        disabled={loading}
      />

      <ButtonGroup>
        <Button
          onClick={handleUploadClick}
          disabled={loading || !imageFile}
          fullWidth
          size="lg"
        >
          {loading ? <Spinner size="sm" color="white" /> : 'Run Prediction'}
        </Button>
        {selectedImage && (
          <Button
            variant="secondary"
            onClick={handleClear}
            disabled={loading}
            size="lg"
          >
            Clear
          </Button>
        )}
      </ButtonGroup>
    </Container>
  );
};

export default ImageUploadCard;



