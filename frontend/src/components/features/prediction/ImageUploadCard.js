import React from 'react';
import styled from 'styled-components';
import { Spinner, Button } from '../../common/ui';
import { SmallText } from '../../../styles/typography';
import { usePrediction } from '../../../hooks';
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
  min-height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.border : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  background-color: ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.bg : theme.colors.background.tertiary};
  padding: ${({ theme }) => theme.spacing[6]};
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
  max-height: 260px;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.paper};
`;

const IconWrapper = styled.div`
  width: 64px;
  height: 64px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  
  img {
    width: 36px;
    height: 36px;
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
  margin-top: ${({ theme }) => theme.spacing[1.5] || '6px'};
  font-size: 0.8rem;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[3]};
  margin-top: ${({ theme }) => theme.spacing[1]};
`;

const WarningBox = styled.div`
  padding: ${({ theme }) => theme.spacing[3.5] || '14px'};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
`;

const WarningText = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  font-size: 0.8rem;
  line-height: 1.5;
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

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
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
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
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
    setPredictionResult(null);
    setError(null);
  };

  return (
    <Card>
      <DropZone
        $hasImage={!!selectedImage}
        onClick={() => document.getElementById('file-input').click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {selectedImage ? (
          <ImagePreview src={selectedImage} alt="Preview" />
        ) : (
          <>
            <IconWrapper>
              <img src={DiseaseIcon} alt="Skin analysis" />
            </IconWrapper>
            <UploadHint>
              <span>Click to upload</span> or drag & drop dermoscopy image
            </UploadHint>
            <FileSupportText>
              Supports PNG, JPG, JPEG up to 16MB
            </FileSupportText>
          </>
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
          variant="android"
        >
          {loading ? <Spinner size="sm" color="white" /> : 'Analyze Lesion Image'}
        </Button>
        {selectedImage && (
          <Button variant="secondary" onClick={handleClear} disabled={loading} size="lg">
            Clear
          </Button>
        )}
      </ButtonGroup>

      <WarningBox>
        <WarningText>
          Specialized for dermoscopic skin lesion classification. For clinical triage and research use only.
        </WarningText>
      </WarningBox>
    </Card>
  );
};

export default ImageUploadCard;
