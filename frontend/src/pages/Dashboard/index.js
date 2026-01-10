import { useState } from 'react';
import { FiUploadCloud, FiActivity } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import {
  Container,
  Main,
  LeftPanel,
  RightPanel,
  PanelHeader,
  PanelTitle,
  PanelContent,
} from './styles';

const Dashboard = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  return (
    <Container>
      <Header />

      <Main>
        <LeftPanel>
          <PanelHeader>
            <PanelTitle>
              <FiUploadCloud size={18} />
              <h3>Upload Image</h3>
            </PanelTitle>
          </PanelHeader>
          <PanelContent>
            <ImageUploadCard
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              imageFile={imageFile}
              setImageFile={setImageFile}
              setPredictionResult={setPredictionResult}
              loading={loading}
              setLoading={setLoading}
              setError={setError}
            />
          </PanelContent>
        </LeftPanel>

        <RightPanel>
          <PanelHeader>
            <PanelTitle>
              <FiActivity size={18} />
              <h3>Analysis Results</h3>
            </PanelTitle>
          </PanelHeader>
          <PanelContent>
            <ResultsCard
              predictionResult={predictionResult}
              loading={loading}
              error={error}
            />
          </PanelContent>
        </RightPanel>
      </Main>
    </Container>
  );
};

export default Dashboard;
