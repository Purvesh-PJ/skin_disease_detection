import styled from "styled-components";
import { Spinner, Button } from './ui';
import { SmallText } from '../styles/typography';
import useSkinDiseasePrediction from '../hooks/useSkinDiseasePrediction';
import DiseaseIcon from '../resources/icons/disease_icon.png';

const Card = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const DropZone = styled.div`
  width: 100%;
  aspect-ratio: 4 / 3;
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

const IconWrapper = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  
  img {
    width: 64px;
    height: 64px;
    object-fit: contain;
    opacity: 0.7;
    filter: ${({ theme }) => theme.mode === 'dark' ? 'invert(1) brightness(0.9)' : 'none'};
  }
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
  margin-top: ${({ theme }) => theme.spacing[2]};
`;

const WarningBox = styled.div`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.status.warning.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.warning.border};
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const WarningText = styled(SmallText)`
  color: ${({ theme }) => theme.colors.status.warning.text};
  font-size: 0.75rem;
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
  const { postImageToPredict } = useSkinDiseasePrediction();

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
    formData.append("image", imageFile);
    setLoading(true);
    setError(null);

    try {
      const response = await postImageToPredict(formData);
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
              <span>Click to upload</span> or drag and drop
            </UploadHint>
            <SmallText style={{ marginTop: '4px', fontSize: '0.75rem' }} variant="tertiary">
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
          size="lg"
        >
          {loading ? <Spinner size="sm" color="white" /> : "Analyze Image"}
        </Button>
        {selectedImage && (
          <Button variant="secondary" onClick={handleClear} disabled={loading} size="lg">
            Clear
          </Button>
        )}
      </ButtonGroup>

      <WarningBox>
        <WarningText>
          This AI is trained for skin disease images only. Other images may give inaccurate results.
        </WarningText>
      </WarningBox>
    </Card>
  );
};

export default ImageUploadCard;
