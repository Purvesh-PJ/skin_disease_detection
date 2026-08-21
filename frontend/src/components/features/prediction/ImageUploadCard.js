import React, { useState } from 'react';
import styled from 'styled-components';
import { FiZap, FiShuffle, FiCheck, FiInfo } from 'react-icons/fi';
import { Spinner, Button } from '../../common/ui';
import { SmallText } from '../../../styles/typography';
import { usePrediction } from '../../../hooks';
import { SAMPLE_IMAGES } from '../../../constants';
import DiseaseIcon from '../../../assets/icons/disease_icon.png';

const Card = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
  flex: 1;
`;

const DropZone = styled.div`
  width: 100%;
  box-sizing: border-box;
  flex: 1;
  min-height: 250px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.border : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  background-color: ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.bg : theme.colors.background.tertiary};
  padding: ${({ theme }) => theme.spacing[5]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.normal};
  overflow: hidden;
  position: relative;

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[500]};
    background-color: ${({ theme }) => theme.colors.interactive.selected};
    transform: translateY(-2px);
  }
`;

const ImagePreview = styled.img`
  max-width: 100%;
  max-height: 230px;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.paper};
`;

const SelectedSampleBadge = styled.div`
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(8px);
  color: #ffffff;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 2;
  border: 1px solid rgba(255, 255, 255, 0.2);

  span.risk {
    color: ${({ $risk, theme }) => 
      $risk === 'malignant' ? '#f87171' : $risk === 'precancerous' ? '#fbbf24' : '#4ade80'};
  }
`;

const IconWrapper = styled.div`
  width: 56px;
  height: 56px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: ${({ theme }) => theme.spacing[2]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  
  img {
    width: 32px;
    height: 32px;
    object-fit: contain;
    filter: ${({ theme }) => theme.mode === 'dark' ? 'invert(1) brightness(0.9)' : 'none'};
  }
`;

const HiddenInput = styled.input`
  display: none;
`;

const UploadHint = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;
  font-size: 0.95rem;
  
  span {
    color: ${({ theme }) => theme.colors.primary[600]};
    font-weight: 600;
  }
`;

const FileSupportText = styled(SmallText)`
  margin-top: ${({ theme }) => theme.spacing[1] || '4px'};
  font-size: 0.78rem;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

/* Samples Section Styles */
const SampleSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  padding: ${({ theme }) => theme.spacing[3.5] || '14px'};
  background: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
`;

const SampleHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing[2]};

  .title-group {
    display: flex;
    align-items: center;
    gap: 6px;

    .icon-zap {
      color: #eab308;
    }

    h4 {
      font-size: 0.88rem;
      font-weight: 700;
      color: ${({ theme }) => theme.colors.text.primary};
      margin: 0;
    }

    span.subtitle {
      font-size: 0.75rem;
      color: ${({ theme }) => theme.colors.text.tertiary};
      display: none;
      @media (min-width: 640px) {
        display: inline;
      }
    }
  }
`;

const RandomButton = styled.button`
  background: transparent;
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: 4px 10px;
  font-size: 0.75rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    background: ${({ theme }) => theme.colors.interactive.hover};
    color: ${({ theme }) => theme.colors.primary[600]};
    border-color: ${({ theme }) => theme.colors.primary[500]};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const SampleGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
  gap: 8px;

  @media (max-width: 480px) {
    grid-template-columns: repeat(4, 1fr);
    overflow-x: auto;
    padding-bottom: 4px;
  }
`;

const SampleCard = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px;
  background: ${({ theme, $active }) => 
    $active ? theme.colors.status.success.bg : theme.colors.background.primary};
  border: 1.5px solid ${({ theme, $active }) => 
    $active ? theme.colors.primary[500] : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.sm};
  }

  ${({ $active, theme }) => $active && `
    box-shadow: 0 0 0 2px ${theme.colors.primary[200]};
  `}
`;

const SampleThumbnail = styled.img`
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  margin-bottom: 4px;
`;

const SampleLabel = styled.div`
  font-size: 0.72rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  text-align: center;
  line-height: 1.2;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
`;

const RiskBadge = styled.span`
  font-size: 0.62rem;
  font-weight: 700;
  padding: 1px 4px;
  border-radius: 4px;
  margin-top: 3px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  
  ${({ $risk }) => {
    switch ($risk) {
      case 'malignant':
        return `
          background: rgba(239, 68, 68, 0.15);
          color: #ef4444;
        `;
      case 'precancerous':
        return `
          background: rgba(245, 158, 11, 0.15);
          color: #d97706;
        `;
      case 'benign':
      default:
        return `
          background: rgba(34, 197, 94, 0.15);
          color: #16a34a;
        `;
    }
  }}
`;

const CheckIndicator = styled.div`
  position: absolute;
  top: 4px;
  right: 4px;
  width: 16px;
  height: 16px;
  background: ${({ theme }) => theme.colors.primary[500]};
  color: #ffffff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[3]};
  margin-top: ${({ theme }) => theme.spacing[1]};
`;

const WarningBox = styled.div`
  padding: ${({ theme }) => theme.spacing[3] || '12px'};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: flex-start;
  gap: 8px;

  .info-icon {
    color: ${({ theme }) => theme.colors.text.tertiary};
    flex-shrink: 0;
    margin-top: 2px;
  }
`;

const WarningText = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  font-size: 0.78rem;
  line-height: 1.45;
`;

const ImageUploadCard = ({
  selectedImage,
  setSelectedImage,
  imageFile,
  setImageFile,
  setPredictionResult,
  loading,
  setLoading,
  setError
}) => {
  const { predict } = usePrediction();
  const [activeSample, setActiveSample] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      setActiveSample(null);
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
      setActiveSample(null);
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // Convert sample image URL into a File object for the API
  const handleSelectSample = async (sample) => {
    if (loading || loadingSample) return;
    setLoadingSample(true);
    setActiveSample(sample);
    setError(null);
    setPredictionResult(null);

    try {
      const response = await fetch(sample.imagePath);
      if (!response.ok) throw new Error(`Could not load sample image: ${response.statusText}`);
      
      const blob = await response.blob();
      const file = new File([blob], sample.fileName, { type: blob.type || 'image/jpeg' });
      
      setSelectedImage(sample.imagePath);
      setImageFile(file);
    } catch (err) {
      console.error('Failed to load sample image file:', err);
      // Fallback: set preview at least
      setSelectedImage(sample.imagePath);
      setError(err);
    } finally {
      setLoadingSample(false);
    }
  };

  const handleRandomSample = () => {
    if (loading || loadingSample) return;
    const available = SAMPLE_IMAGES.filter((s) => s.id !== activeSample?.id);
    const random = available[Math.floor(Math.random() * available.length)] || SAMPLE_IMAGES[0];
    handleSelectSample(random);
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
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImageFile(null);
    setActiveSample(null);
    setPredictionResult(null);
    setError(null);
  };

  return (
    <Card>
      {/* DropZone for uploading / previewing */}
      <DropZone
        $hasImage={!!selectedImage}
        onClick={() => document.getElementById('file-input').click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {selectedImage ? (
          <>
            {activeSample && (
              <SelectedSampleBadge $risk={activeSample.type}>
                <span className="risk">●</span> Sample: {activeSample.name} ({activeSample.code})
              </SelectedSampleBadge>
            )}
            <ImagePreview src={selectedImage} alt="Preview" />
          </>
        ) : (
          <>
            <IconWrapper>
              <img src={DiseaseIcon} alt="Skin analysis" />
            </IconWrapper>
            <UploadHint>
              <span>Click to upload</span> or drag & drop dermoscopy image
            </UploadHint>
            <FileSupportText>
              Supports PNG, JPG, JPEG (or pick a 1-click sample below)
            </FileSupportText>
          </>
        )}
      </DropZone>

      <HiddenInput
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        disabled={loading || loadingSample}
      />

      {/* 1-Click Demo Samples Section */}
      <SampleSection>
        <SampleHeader>
          <div className="title-group">
            <FiZap className="icon-zap" size={15} />
            <h4>Try Demo Samples (7 Classes)</h4>
            <span className="subtitle">• 1-click test</span>
          </div>
          <RandomButton
            type="button"
            onClick={handleRandomSample}
            disabled={loading || loadingSample}
            title="Select a random benchmark lesion sample"
          >
            <FiShuffle size={12} />
            <span>Random</span>
          </RandomButton>
        </SampleHeader>

        <SampleGrid>
          {SAMPLE_IMAGES.map((sample) => {
            const isSelected = activeSample?.id === sample.id;
            return (
              <SampleCard
                key={sample.id}
                $active={isSelected}
                onClick={() => handleSelectSample(sample)}
                title={`${sample.name} (${sample.code}) - ${sample.typeLabel} - ${sample.description}`}
              >
                {isSelected && (
                  <CheckIndicator>
                    <FiCheck size={11} strokeWidth={3} />
                  </CheckIndicator>
                )}
                <SampleThumbnail src={sample.imagePath} alt={sample.name} loading="lazy" />
                <SampleLabel>{sample.code}</SampleLabel>
                <RiskBadge $risk={sample.type}>
                  {sample.type === 'malignant' ? 'Malignant' : sample.type === 'precancerous' ? 'Pre-Canc.' : 'Benign'}
                </RiskBadge>
              </SampleCard>
            );
          })}
        </SampleGrid>
      </SampleSection>

      {/* Action Buttons */}
      <ButtonGroup>
        <Button
          onClick={handleUploadClick}
          disabled={loading || loadingSample || !imageFile}
          fullWidth
          size="lg"
          variant="brand"
        >
          {loading ? (
            <Spinner size="sm" color="white" />
          ) : (
            `Analyze ${activeSample ? `${activeSample.code} Sample` : 'Lesion Image'}`
          )}
        </Button>
        {selectedImage && (
          <Button
            variant="secondary"
            onClick={handleClear}
            disabled={loading || loadingSample}
            size="lg"
          >
            Clear
          </Button>
        )}
      </ButtonGroup>

      <WarningBox>
        <FiInfo className="info-icon" size={16} />
        <WarningText>
          Specialized for dermoscopic skin lesion classification across 7 ISIC/HAM10000 classes. For clinical triage and research use only.
        </WarningText>
      </WarningBox>
    </Card>
  );
};

export default ImageUploadCard;

