import { useState } from 'react';
import { FiActivity, FiDatabase } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import {
  Container,
  Main,
  LeftColumn,
  RightColumn,
  TabBar,
  TabButtons,
  TabButton,
} from './styles';

const Dashboard = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
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

  return (
    <Container key={userUpdateKey}>
      <Header onUserUpdated={handleUserUpdated} />

      <Main>
        {/* Left Column: Image Dropzone & Fast Samples */}
        <LeftColumn>
          <ImageUploadCard
            selectedImage={selectedImage}
            setSelectedImage={setSelectedImage}
            imageFile={imageFile}
            setImageFile={setImageFile}
            setPredictionResult={handlePredictionSuccess}
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
    </Container>
  );
};

export default Dashboard;



