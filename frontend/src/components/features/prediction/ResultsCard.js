import styled from 'styled-components';
import { FiUpload, FiAlertCircle, FiCheckCircle } from 'react-icons/fi';
import { Spinner } from '../../common/ui';
import { Text, SmallText } from '../../../styles/typography';

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
  width: 70px;
  height: 70px;
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
`;

const ResultContainer = styled.div`
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: ${({ theme }) => theme.spacing[4]};
  flex: 1;
  overflow-y: auto;
`;

const ResultHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  padding-bottom: ${({ theme }) => theme.spacing[3]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  margin-bottom: ${({ theme }) => theme.spacing[3]};
`;

const SuccessIcon = styled.div`
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.status.success.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.success.border};
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.status.success.icon};
`;

const DiseaseName = styled.h3`
  font-size: 1.1rem;
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
  font-size: 0.7rem;
  font-weight: 600;
  margin-left: auto;
`;

const Description = styled(Text)`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 0.85rem;
  line-height: 1.5;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
`;

const DetailGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing[2]};
`;

const DetailItem = styled.div`
  padding: ${({ theme }) => theme.spacing[2]};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.md};
`;

const DetailLabel = styled(SmallText)`
  color: ${({ theme }) => theme.colors.text.tertiary};
  font-size: 0.7rem;
  margin-bottom: 2px;
`;

const DetailValue = styled(Text)`
  font-weight: 500;
  font-size: 0.85rem;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const ResultsCard = ({ predictionResult, loading, error }) => {
  const getDiseaseName = (result) => {
    return result.disease_details?.name || result.predicted_disease;
  };

  const getDescription = (result) => {
    return result.disease_details?.description || null;
  };

  if (loading) {
    return (
      <Card>
        <EmptyState>
          <Spinner size="lg" />
          <Text style={{ marginTop: '16px' }} variant="secondary">Analyzing...</Text>
          <SmallText variant="tertiary">This may take a few seconds</SmallText>
        </EmptyState>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <EmptyState>
          <StateIcon $variant="error">
            <FiAlertCircle size={28} />
          </StateIcon>
          <Text variant="secondary">Analysis failed</Text>
          <SmallText variant="tertiary">
            {error.response?.data?.message || 'Please try again.'}
          </SmallText>
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
              <FiCheckCircle size={20} />
            </SuccessIcon>
            <div>
              <DiseaseName>{getDiseaseName(predictionResult)}</DiseaseName>
              <SmallText variant="tertiary" style={{ fontSize: '0.75rem' }}>Detected condition</SmallText>
            </div>
            <ConfidenceBadge>{predictionResult.confidence}%</ConfidenceBadge>
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
        </ResultContainer>
      </Card>
    );
  }

  return (
    <Card>
      <EmptyState>
        <StateIcon>
          <FiUpload size={28} />
        </StateIcon>
        <Text variant="secondary">No results yet</Text>
        <SmallText variant="tertiary">Upload an image to get started</SmallText>
      </EmptyState>
    </Card>
  );
};

export default ResultsCard;
