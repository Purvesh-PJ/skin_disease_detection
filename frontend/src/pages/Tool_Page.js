import styled from 'styled-components';
import ImageUpload from '../components/ImageUpload';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  min-height: 100vh;
`;

const Section = styled.section`
  display: flex;
  flex-direction: row;
  gap: 1px;
  width: 100%;
  max-width: 1250px;
  height: 800px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.subtle};
  background-color: ${({ theme }) => theme.colors.neutral[200]};
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex-direction: column;
    height: auto;
  }
`;

const ImageUploadSection = styled.div`
  display: flex;
  justify-content: center;
  width: 50%;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-top-left-radius: ${({ theme }) => theme.borderRadius.lg};
  border-bottom-left-radius: ${({ theme }) => theme.borderRadius.lg};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    width: 100%;
    border-radius: ${({ theme }) => theme.borderRadius.lg} ${({ theme }) => theme.borderRadius.lg} 0 0;
  }
`;

const PredictedResultSection = styled.div`
  width: 50%;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-top-right-radius: ${({ theme }) => theme.borderRadius.lg};
  border-bottom-right-radius: ${({ theme }) => theme.borderRadius.lg};
  box-sizing: border-box;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    width: 100%;
    border-radius: 0 0 ${({ theme }) => theme.borderRadius.lg} ${({ theme }) => theme.borderRadius.lg};
  }
`;

const ToolPage = () => {
  return (
    <Container>
      <Section>
        <ImageUploadSection>
          <ImageUpload />
        </ImageUploadSection>
        <PredictedResultSection />
      </Section>
    </Container>
  );
};

export default ToolPage;
