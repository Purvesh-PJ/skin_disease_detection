import { useState } from 'react';
import { FiUploadCloud, FiActivity, FiDatabase } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import {
  Container,
  Main,
  LeftPanel,
  RightPanel,
  PanelHeader,
  PanelTitle,
  PanelContent,
  TabGroup,
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
        <LeftPanel>
          <PanelHeader>
            <PanelTitle>
              <FiUploadCloud size={18} />
              <h3>Dermoscopy Image Input</h3>
            </PanelTitle>
          </PanelHeader>
          <PanelContent>
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
          </PanelContent>
        </LeftPanel>

        <RightPanel>
          <PanelHeader>
            <PanelTitle>
              {activeTab === 'analysis' ? (
                <>
                  <FiActivity size={18} />
                  <h3>Diagnostic Analysis</h3>
                </>
              ) : (
                <>
                  <FiDatabase size={18} color="#16a34a" />
                  <h3>MongoDB Saved Scans</h3>
                </>
              )}
            </PanelTitle>

            <TabGroup>
              <TabButton
                $active={activeTab === 'analysis'}
                onClick={() => setActiveTab('analysis')}
                type="button"
              >
                <FiActivity size={14} />
                Live Analysis
              </TabButton>
              <TabButton
                $active={activeTab === 'history'}
                onClick={() => setActiveTab('history')}
                type="button"
              >
                <FiDatabase size={14} />
                Scan History
              </TabButton>
            </TabGroup>
          </PanelHeader>
          <PanelContent>
            {activeTab === 'analysis' ? (
              <ResultsCard
                predictionResult={predictionResult}
                loading={loading}
                error={error}
              />
            ) : (
              <HistoryList refreshTrigger={historyRefreshKey} />
            )}
          </PanelContent>
        </RightPanel>
      </Main>
    </Container>
  );
};

export default Dashboard;

