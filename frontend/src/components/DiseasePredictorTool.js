import { useState } from 'react';
import useSkinDiseasePrediction from '../hooks/useSkinDiseasePrediction';
import styled from "styled-components";
import { FiUpload, FiAlertCircle, FiCheckCircle, FiImage } from 'react-icons/fi';
import { Spinner, Button } from './ui';
import { Text, SmallText, H4 } from '../styles/typography';

const Container = styled.div`
  display: flex;
  flex-direction: row;
  width: 100%;
  height: 100%;
  background-color: ${({ theme }) => theme.colors.background.primary};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex-direction: column;
  }
`;

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
`;

const LeftPanel = styled(Panel)`
  width: 45%;
  border-right: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.primary};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    width: 100%;
    height: 50%;
    border-right: none;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  }
`;

const RightPanel = styled(Panel)`
  width: 55%;
  background-color: ${({ theme }) => theme.colors.background.secondary};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    width: 100%;
    height: 50%;
  }
`;

const PanelHeader = styled.div`
  padding: ${({ theme }) => theme.spacing[5]} ${({ theme }) => theme.spacing[6]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

const PanelHeaderContent = styled.div`
  padding-left: ${({ theme }) => theme.spacing[3]};
  border-left: 3px solid ${({ theme }) => theme.colors.primary[500]};
`;

const PanelTitle = styled(H4)`
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const PanelSubtitle = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const PanelContent = styled.div`
  flex: 1;
  padding: ${({ theme }) => theme.spacing[6]};
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const DropZone = styled.div`
  height: 280px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.border : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  background-color: ${({ theme, $hasImage }) => 
    $hasImage ? theme.colors.status.success.bg : theme.colors.background.tertiary};
  padding: ${({ theme }) => theme.spacing[4]};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  overflow: hidden;

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
    background-color: ${({ theme }) => theme.colors.interactive.selected};
  }
`;

const ImagePreview = styled.img`
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const PlaceholderIcon = styled.div`
  width: 70px;
  height: 70px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const HiddenInput = styled.input`
  display: none;
`;

const UploadHint = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  text-align: center;
  
  span {
    color: ${({ theme }) => theme.colors.primary[500]};
    font-weight: 500;
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[3]};
  margin-top: ${({ theme }) => theme.spacing[5]};
`;

const WarningBox = styled.div`
  display: flex;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.status.warning.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.warning.border};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  margin-top: auto;
  
  svg {
    color: ${({ theme }) => theme.colors.status.warning.icon};
    flex-shrink: 0;
    margin-top: 2px;
  }
`;

const WarningText = styled(SmallText)`
  color: ${({ theme }) => theme.colors.status.warning.text};
`;

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: ${({ theme }) => theme.spacing[8]};
`;

const EmptyIcon = styled.div`
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.status.info.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.info.border};
  border-radius: 50%;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  color: ${({ theme }) => theme.colors.status.info.icon};
`;

const LoadingState = styled(EmptyState)``;

const ResultCard = styled.div`
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  padding: ${({ theme }) => theme.spacing[5]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
`;

const ResultHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  padding-bottom: ${({ theme }) => theme.spacing[4]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const SuccessIcon = styled.div`
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.status.success.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.success.border};
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.status.success.icon};
`;

const DiseaseName = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0;
`;

const ConfidenceBadge = styled.span`
  display: inline-flex;
  align-items: center;
  padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[2]}`};
  background-color: ${({ theme }) => theme.colors.interactive.selected};
  color: ${({ theme }) => theme.colors.primary[500]};
  border: 1px solid ${({ theme }) => theme.colors.primary[400]};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: 0.75rem;
  font-weight: 600;
  margin-left: auto;
`;

const Description = styled(Text)`
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.6;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const DetailGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing[3]};
  
  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    grid-template-columns: 1fr;
  }
`;

const DetailItem = styled.div`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const DetailLabel = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const DetailValue = styled(Text)`
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const ErrorState = styled(EmptyState)``;

const ErrorIcon = styled.div`
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.status.error.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.error.border};
  border-radius: 50%;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  color: ${({ theme }) => theme.colors.status.error.icon};
`;

const DiseasePredictorTool = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const { postImageToPredict, error, loading } = useSkinDiseasePrediction();

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      setPredictionResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      setPredictionResult(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUploadClick = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await postImageToPredict(formData);
      if (response?.data) {
        setPredictionResult(response.data);
      }
    } catch (err) {
      console.error("Error during prediction:", err);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImageFile(null);
    setPredictionResult(null);
  };

  const getDiseaseName = (result) => {
    return result.disease_details?.name || result.predicted_disease;
  };

  const getDescription = (result) => {
    return result.disease_details?.description || null;
  };

  return (
    <Container>
      <LeftPanel>
        <PanelHeader>
          <PanelHeaderContent>
            <PanelTitle>Upload Image</PanelTitle>
            <PanelSubtitle>Drag and drop or click to select</PanelSubtitle>
          </PanelHeaderContent>
        </PanelHeader>

        <PanelContent>
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
                <PlaceholderIcon>
                  <FiImage size={28} />
                </PlaceholderIcon>
                <UploadHint>
                  <span>Click to upload</span> or drag and drop
                </UploadHint>
                <SmallText style={{ marginTop: '8px' }} variant="tertiary">
                  PNG, JPG up to 10MB
                </SmallText>
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
            >
              {loading ? <Spinner size="sm" color="white" /> : "Analyze Image"}
            </Button>
            {selectedImage && (
              <Button variant="secondary" onClick={handleClear} disabled={loading}>
                Clear
              </Button>
            )}
          </ButtonGroup>

          <WarningBox>
            <FiAlertCircle size={16} />
            <WarningText>
              This AI model is trained specifically for skin disease images. Results for other image types may be inaccurate.
            </WarningText>
          </WarningBox>
        </PanelContent>
      </LeftPanel>

      <RightPanel>
        <PanelHeader style={{ backgroundColor: 'transparent' }}>
          <PanelHeaderContent>
            <PanelTitle>Analysis Results</PanelTitle>
            <PanelSubtitle>AI-powered skin condition detection</PanelSubtitle>
          </PanelHeaderContent>
        </PanelHeader>

        <PanelContent>
          {loading ? (
            <LoadingState>
              <Spinner size="lg" />
              <Text style={{ marginTop: '16px' }} variant="secondary">Analyzing your image...</Text>
              <SmallText variant="tertiary">This may take a few seconds</SmallText>
            </LoadingState>
          ) : error ? (
            <ErrorState>
              <ErrorIcon>
                <FiAlertCircle size={32} />
              </ErrorIcon>
              <Text variant="secondary">Analysis failed</Text>
              <SmallText variant="tertiary">
                {error.response?.data?.message || "Please check your connection and try again."}
              </SmallText>
            </ErrorState>
          ) : predictionResult ? (
            <ResultCard>
              <ResultHeader>
                <SuccessIcon>
                  <FiCheckCircle size={24} />
                </SuccessIcon>
                <div>
                  <DiseaseName>{getDiseaseName(predictionResult)}</DiseaseName>
                  <SmallText variant="tertiary">Detected condition</SmallText>
                </div>
                <ConfidenceBadge>{predictionResult.confidence}% confidence</ConfidenceBadge>
              </ResultHeader>

              {getDescription(predictionResult) && (
                <Description>{getDescription(predictionResult)}</Description>
              )}

              <DetailGrid>
                {Object.entries(predictionResult)
                  .filter(([key]) => !['predicted_disease', 'confidence', 'disease_details', 'message'].includes(key))
                  .slice(0, 4)
                  .map(([key, value]) => (
                    <DetailItem key={key}>
                      <DetailLabel>
                        {key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                      </DetailLabel>
                      <DetailValue>
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </DetailValue>
                    </DetailItem>
                  ))}
              </DetailGrid>
            </ResultCard>
          ) : (
            <EmptyState>
              <EmptyIcon>
                <FiUpload size={32} />
              </EmptyIcon>
              <Text variant="secondary">No results yet</Text>
              <SmallText variant="tertiary">Upload an image to get started</SmallText>
            </EmptyState>
          )}
        </PanelContent>
      </RightPanel>
    </Container>
  );
};

export default DiseasePredictorTool;
