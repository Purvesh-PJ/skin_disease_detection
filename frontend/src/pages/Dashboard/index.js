import { useState } from 'react';
import { FiActivity, FiDatabase, FiZap } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import { SAMPLE_IMAGES } from '../../constants';
import {
  Container,
  Main,
  LeftColumn,
  RightColumn,
  TabBar,
  TabButtons,
  TabButton,
  SampleFooterRail,
  SampleRailLabel,
  SampleRailList,
  SampleCard,
} from './styles';

const Dashboard = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [activeSample, setActiveSample] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('analysis');
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [userUpdateKey, setUserUpdateKey] = useState(0);

  const handlePredictionSuccess = (result) => {
    setPredictionResult(result);
    setActiveTab('analysis');
    setHistoryRefreshKey((prev) => prev + 1);
  };

  const handleUserUpdated = () => {
    setUserUpdateKey((prev) => prev + 1);
  };

  const urlToFile = async (url, filename) => {
    try {
      const response = await fetch(url);
      if (response.ok) {
        const blob = await response.blob();
        if (blob.type && blob.type.startsWith('image/')) {
          return new File([blob], filename, { type: blob.type });
        }
      }
    } catch (e) {
      console.warn('Canvas fallback');
    }

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth || 400;
        canvas.height = img.naturalHeight || 300;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(new File([blob], filename, { type: 'image/jpeg' }));
          } else {
            reject(new Error('Canvas error'));
          }
        }, 'image/jpeg', 0.9);
      };
      img.onerror = () => reject(new Error('Sample load error'));
      img.src = url;
    });
  };

  const handleSelectSample = async (sample) => {
    if (loading) return;
    setActiveSample(sample);
    setSelectedImage(sample.imagePath);
    setError(null);
    setPredictionResult(null);

    try {
      const file = await urlToFile(sample.imagePath, sample.fileName);
      setImageFile(file);
    } catch (err) {
      console.error('Failed to convert sample image:', err);
    }
  };

  return (
    <Container key={userUpdateKey}>
      <Header onUserUpdated={handleUserUpdated} />

      <Main>
        {/* Left Column: Image Dropzone (Centered) & Action Buttons */}
        <LeftColumn>
          <ImageUploadCard
            selectedImage={selectedImage}
            setSelectedImage={setSelectedImage}
            imageFile={imageFile}
            setImageFile={setImageFile}
            setPredictionResult={handlePredictionSuccess}
            activeSample={activeSample}
            setActiveSample={setActiveSample}
            loading={loading}
            setLoading={setLoading}
            setError={setError}
          />
        </LeftColumn>

        {/* Right Column: Analysis Results or History */}
        <RightColumn>
          <TabBar>
            <TabButtons>
              <TabButton
                $active={activeTab === 'analysis'}
                onClick={() => setActiveTab('analysis')}
                type="button"
              >
                <FiActivity size={14} />
                Analysis Result
              </TabButton>
              <TabButton
                $active={activeTab === 'history'}
                onClick={() => setActiveTab('history')}
                type="button"
              >
                <FiDatabase size={14} />
                Saved Scans
              </TabButton>
            </TabButtons>
          </TabBar>

          {activeTab === 'analysis' ? (
            <ResultsCard
              predictionResult={predictionResult}
              loading={loading}
              error={error}
            />
          ) : (
            <HistoryList refreshTrigger={historyRefreshKey} />
          )}
        </RightColumn>
      </Main>

      {/* Bottom Full-Width Sample Rail */}
      <SampleFooterRail>
        <SampleRailLabel>
          <FiZap color="#eab308" size={14} />
          <span>Demo Samples:</span>
        </SampleRailLabel>

        <SampleRailList>
          {SAMPLE_IMAGES.map((sample) => {
            const isSelected = activeSample?.id === sample.id;
            return (
              <SampleCard
                key={sample.id}
                $active={isSelected}
                onClick={() => handleSelectSample(sample)}
                type="button"
                title={`${sample.name} (${sample.code})`}
              >
                <img src={sample.imagePath} alt={sample.code} />
                <div className="meta">
                  <span className="name">{sample.code}</span>
                  <span className="type">{sample.typeLabel || sample.type}</span>
                </div>
              </SampleCard>
            );
          })}
        </SampleRailList>
      </SampleFooterRail>
    </Container>
  );
};

export default Dashboard;




