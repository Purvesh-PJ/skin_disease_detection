import { FiActivity, FiClock, FiZap } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import { SAMPLE_IMAGES } from '../../constants';
import { useDashboard } from '../../hooks';
import {
  Container,
  Main,
  LeftColumn,
  LeftContentArea,
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
  const { state, actions } = useDashboard();
  const {
    selectedImage,
    imageFile,
    activeSample,
    predictionResult,
    loading,
    error,
    activeTab,
    historyRefreshKey,
    userUpdateKey,
    hasScans,
  } = state;

  const {
    setSelectedImage,
    setImageFile,
    setActiveSample,
    setLoading,
    setError,
    setActiveTab,
    handlePredictionSuccess,
    handleUserUpdated,
    handleSelectSample,
  } = actions;

  return (
    <Container key={userUpdateKey}>
      <Header onUserUpdated={handleUserUpdated} />

      <Main>
        {/* Left Column: Image Dropzone & Bottom Sample Rail strictly on the Left */}
        <LeftColumn>
          <LeftContentArea>
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
          </LeftContentArea>

          {/* Bottom Sample Rail strictly inside Left Column */}
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
                onClick={() => {
                  if (hasScans) {
                    setActiveTab('history');
                  }
                }}
                disabled={!hasScans}
                title={hasScans ? 'View diagnostic scan history' : 'Perform an image scan to see history'}
                type="button"
              >
                <FiClock size={14} />
                See History
              </TabButton>
            </TabButtons>
          </TabBar>

          <div style={{ display: activeTab === 'analysis' ? 'flex' : 'none', flexDirection: 'column', height: '100%' }}>
            <ResultsCard
              predictionResult={predictionResult}
              loading={loading}
              error={error}
            />
          </div>

          <div style={{ display: activeTab === 'history' ? 'flex' : 'none', flexDirection: 'column', height: '100%' }}>
            <HistoryList refreshTrigger={historyRefreshKey} />
          </div>
        </RightColumn>
      </Main>
    </Container>
  );
};

export default Dashboard;
