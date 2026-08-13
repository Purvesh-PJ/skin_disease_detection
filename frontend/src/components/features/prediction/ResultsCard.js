import styled, { keyframes } from 'styled-components';
import { FiAlertCircle, FiCheckCircle, FiShield, FiFileText } from 'react-icons/fi';
import { Spinner } from '../../common/ui';
import { Text, SmallText } from '../../../styles/typography';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
`;

const fillProgress = keyframes`
  from { width: 0%; }
  to { width: var(--progress-width); }
`;

const Card = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: ${({ theme }) => theme.spacing[6]};
`;

const StateIcon = styled.div`
  width: 72px;
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme, $variant }) => {
    if ($variant === 'error') return theme.colors.status.error.bg;
    if ($variant === 'success') return theme.colors.status.success.bg;
    return theme.colors.status.info.bg;
  }};
  border: 1px solid ${({ theme, $variant }) => {
    if ($variant === 'error') return theme.colors.status.error.border;
    if ($variant === 'success') return theme.colors.status.success.border;
    return theme.colors.status.info.border;
  }};
  border-radius: 50%;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  color: ${({ theme, $variant }) => {
    if ($variant === 'error') return theme.colors.status.error.icon;
    if ($variant === 'success') return theme.colors.status.success.icon;
    return theme.colors.status.info.icon;
  }};
  box-shadow: ${({ theme }) => theme.shadows.sm};
`;

const ResultContainer = styled.div`
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  padding: ${({ theme }) => theme.spacing[6]};
  flex: 1;
  overflow-y: auto;
  animation: ${fadeIn} 0.3s ease-out forwards;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const ResultHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  padding-bottom: ${({ theme }) => theme.spacing[4]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
`;

const SuccessIcon = styled.div`
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.status.success.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.success.border};
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.status.success.icon};
  flex-shrink: 0;
`;

const DiseaseName = styled.h3`
  font-size: 1.25rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0 0 2px 0;
  letter-spacing: -0.01em;
`;

const SubText = styled(SmallText)`
  font-size: 0.8rem;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const ConfidenceBadge = styled.span`
  display: inline-flex;
  align-items: center;
  padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[3]}`};
  background-color: ${({ theme }) => theme.colors.primary[50]};
  color: ${({ theme }) => theme.colors.primary[600]};
  border: 1px solid ${({ theme }) => theme.colors.primary[200]};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  font-size: 0.85rem;
  font-weight: 700;
  margin-left: auto;
`;

const ProgressSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const ProgressLabelRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ProgressTitle = styled(SmallText)`
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const ProgressPercentText = styled(SmallText)`
  font-weight: 700;
  color: ${({ theme }) => theme.colors.primary[500]};
`;

const ProgressBarContainer = styled.div`
  width: 100%;
  height: 8px;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  overflow: hidden;
`;

const ProgressBarFill = styled.div`
  height: 100%;
  background: ${({ theme }) => theme.gradients.progressBar};
  border-radius: ${({ theme }) => theme.borderRadius.full};
  --progress-width: ${({ $percentage }) => `${$percentage}%`};
  animation: ${fillProgress} 0.8s ease-out forwards;
`;

const DescriptionBox = styled.div`
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
`;

const DescriptionTitle = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  font-size: 0.85rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};

  svg {
    color: ${({ theme }) => theme.colors.primary[500]};
  }
`;

const Description = styled(Text)`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 0.875rem;
  line-height: 1.6;
`;

const DetailGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

const DetailItem = styled.div`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
`;

const DetailLabel = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  font-size: 0.75rem;
  margin-bottom: 4px;
`;

const DetailValue = styled(Text)`
  font-weight: 600;
  font-size: 0.9rem;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const LoadingText = styled(Text)`
  margin-top: ${({ theme }) => theme.spacing[5]};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const LoadingSubText = styled(SmallText)`
  margin-top: ${({ theme }) => theme.spacing[1]};
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const ErrorTitle = styled(Text)`
  font-weight: 600;
  font-size: 1.05rem;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const ErrorSubText = styled(SmallText)`
  margin-top: ${({ theme }) => theme.spacing[1]};
  max-width: 320px;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const EmptyTitle = styled(Text)`
  font-weight: 600;
  font-size: 1.05rem;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const EmptySubText = styled(SmallText)`
  margin-top: ${({ theme }) => theme.spacing[1]};
  max-width: 280px;
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

const ResultsCard = ({ predictionResult, loading, error }) => {
  const getDiseaseName = (result) => {
    return result.disease_details?.name || result.predicted_disease;
  };

  const getDescription = (result) => {
    return result.disease_details?.description || null;
  };

  const confidenceValue = parseInt(predictionResult?.confidence || '0', 10);

  if (loading) {
    return (
      <Card>
        <EmptyState>
          <Spinner size="lg" />
          <LoadingText>Analyzing Skin Lesion...</LoadingText>
          <LoadingSubText>Running ensemble deep neural network inference</LoadingSubText>
        </EmptyState>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <EmptyState>
          <StateIcon $variant="error">
            <FiAlertCircle size={32} />
          </StateIcon>
          <ErrorTitle>Analysis Failed</ErrorTitle>
          <ErrorSubText>
            {error.response?.data?.message || 'Unable to process image. Please verify input and try again.'}
          </ErrorSubText>
        </EmptyState>
      </Card>
    );
  }

  if (predictionResult) {
    return (
      <Card>
        <ResultContainer>
          <ResultHeader>
            <SuccessIcon>
              <FiCheckCircle size={24} />
            </SuccessIcon>
            <div>
              <DiseaseName>{getDiseaseName(predictionResult)}</DiseaseName>
              <SubText>AI Diagnosis Output</SubText>
            </div>
            <ConfidenceBadge>{confidenceValue}% Match</ConfidenceBadge>
          </ResultHeader>

          <ProgressSection>
            <ProgressLabelRow>
              <ProgressTitle>Ensemble Confidence Score</ProgressTitle>
              <ProgressPercentText>{confidenceValue}%</ProgressPercentText>
            </ProgressLabelRow>
            <ProgressBarContainer>
              <ProgressBarFill $percentage={confidenceValue} />
            </ProgressBarContainer>
          </ProgressSection>

          {getDescription(predictionResult) && (
            <DescriptionBox>
              <DescriptionTitle>
                <FiFileText size={16} />
                Condition Summary & Guidance
              </DescriptionTitle>
              <Description>{getDescription(predictionResult)}</Description>
            </DescriptionBox>
          )}

          <DetailGrid>
            {Object.entries(predictionResult)
              .filter(([key]) => !['predicted_disease', 'confidence', 'disease_details', 'message'].includes(key))
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
        </ResultContainer>
      </Card>
    );
  }

  return (
    <Card>
      <EmptyState>
        <StateIcon>
          <FiShield size={32} />
        </StateIcon>
        <EmptyTitle>No Diagnostic Data</EmptyTitle>
        <EmptySubText>
          Upload a skin lesion image on the left panel to execute multi-model ensemble analysis.
        </EmptySubText>
      </EmptyState>
    </Card>
  );
};

export default ResultsCard;
