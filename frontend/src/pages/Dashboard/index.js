import { useState } from 'react';
import { FiUploadCloud, FiActivity, FiDatabase, FiCpu, FiLayers } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import {
  Container,
  Main,
  WorkbenchToolbar,
  TechBadge,
  WorkbenchContainer,
  WorkbenchSection,
  SectionHeader,
  SectionTitle,
  SectionBody,
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
        <WorkbenchToolbar>
          <div className="toolbar-left">
            <h2>
              <FiActivity color="#16a34a" size={20} />
              Dermoscopic Diagnostic Studio
            </h2>
            <span className="live-dot">
              <span className="pulse" />
              Live AI Pipeline Active
            </span>
          </div>

          <div className="toolbar-right">
            <TechBadge>
              <FiLayers size={13} color="#16a34a" />
              <span>Ensemble: ResNet101 • DenseNet121 • EfficientNetB3</span>
            </TechBadge>
            <TechBadge>
              <FiCpu size={13} color="#0284c7" />
              <span>HAM10000 7-Class Model</span>
            </TechBadge>
            <TechBadge>
              <FiDatabase size={13} color="#16a34a" />
              <span>MongoDB Atlas Connected</span>
            </TechBadge>
          </div>
        </WorkbenchToolbar>

        {/* Seamless Unified Workbench */}
        <WorkbenchContainer>
          {/* Left Section: Image Input & Benchmarks */}
          <WorkbenchSection className="left-section">
            <SectionHeader>
              <SectionTitle>
                <FiUploadCloud size={18} />
                <h3>Dermoscopy Image Input</h3>
              </SectionTitle>
            </SectionHeader>
            <SectionBody>
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
            </SectionBody>
          </WorkbenchSection>

          {/* Right Section: Diagnostics & History */}
          <WorkbenchSection className="right-section">
            <SectionHeader>
              <SectionTitle>
                {activeTab === 'analysis' ? (
                  <>
                    <FiActivity size={18} />
                    <h3>Diagnostic Intelligence</h3>
                  </>
                ) : (
                  <>
                    <FiDatabase size={18} color="#16a34a" />
                    <h3>MongoDB Saved Scans</h3>
                  </>
                )}
              </SectionTitle>

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
            </SectionHeader>

            <SectionBody>
              {activeTab === 'analysis' ? (
                <ResultsCard
                  predictionResult={predictionResult}
                  loading={loading}
                  error={error}
                />
              ) : (
                <HistoryList refreshTrigger={historyRefreshKey} />
              )}
            </SectionBody>
          </WorkbenchSection>
        </WorkbenchContainer>
      </Main>
    </Container>
  );
};

export default Dashboard;


