import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { ThemeToggle } from '../common/ui';
import { ROUTES } from '../../constants';
import { FiActivity, FiLayers, FiDatabase, FiShield, FiCheckCircle } from 'react-icons/fi';

const PageWrapper = styled.div`
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: ${({ theme }) =>
    theme.mode === 'dark'
      ? 'radial-gradient(ellipse at top left, rgba(22, 163, 74, 0.12), transparent 50%), radial-gradient(ellipse at bottom right, rgba(14, 165, 233, 0.08), transparent 50%), #0d1117'
      : 'radial-gradient(ellipse at top left, rgba(220, 252, 231, 0.6), transparent 50%), radial-gradient(ellipse at bottom right, rgba(224, 242, 254, 0.5), transparent 50%), #f8fafc'};
  color: ${({ theme }) => theme.colors.text.primary};
  overflow-x: hidden;
`;

const AuthHeader = styled.header`
  position: sticky;
  top: 0;
  z-index: 50;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  background-color: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(13, 17, 23, 0.85)' : 'rgba(248, 250, 252, 0.85)'};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[8]}`};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
  }
`;

const Brand = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  text-decoration: none;
`;

const BrandIcon = styled.div`
  width: 34px;
  height: 34px;
  background: ${({ theme }) => theme.colors.primary[600]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 2px 10px rgba(22, 163, 74, 0.3);
`;

const BrandTitle = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.15rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.02em;
`;

const SplitLayout = styled.main`
  flex: 1;
  max-width: 1360px;
  width: 100%;
  margin: 0 auto;
  padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[8]}`};
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
    padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[4]}`};
  }
`;

const ShowcasePanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    display: none;
  }
`;

const BadgePill = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(22, 163, 74, 0.15)' : 'rgba(220, 252, 231, 0.9)'};
  color: ${({ theme }) => theme.colors.primary[600]};
  border: 1px solid ${({ theme }) => theme.colors.primary[500]};
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  width: fit-content;
`;

const ShowcaseTitle = styled.h1`
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.2;
  margin: 0;
  color: ${({ theme }) => theme.colors.text.primary};

  span.highlight {
    background: linear-gradient(135deg, #16a34a 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
`;

const ShowcaseDesc = styled.p`
  font-size: 0.95rem;
  line-height: 1.6;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: 0;
  max-width: 540px;
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing[3]};
  margin-top: ${({ theme }) => theme.spacing[1]};
`;

const FeatureCard = styled.div`
  background: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(22, 27, 34, 0.7)' : 'rgba(255, 255, 255, 0.8)'};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  backdrop-filter: blur(10px);
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[500]};
    transform: translateY(-2px);
  }

  .header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.9rem;
    color: ${({ theme }) => theme.colors.text.primary};

    svg {
      color: ${({ theme }) => theme.colors.primary[500]};
    }
  }

  .desc {
    font-size: 0.78rem;
    color: ${({ theme }) => theme.colors.text.secondary};
    line-height: 1.4;
  }
`;

const FormColumn = styled.div`
  display: flex;
  justify-content: center;
  width: 100%;
`;

const AuthCard = styled.div`
  width: 100%;
  max-width: 460px;
  background: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(22, 27, 34, 0.92)' : 'rgba(255, 255, 255, 0.95)'};
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
  padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[6]}`};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    padding: ${({ theme }) => `${theme.spacing[5]} ${theme.spacing[4]}`};
  }
`;

export const AuthLayout = ({ children }) => {
  return (
    <PageWrapper>
      <AuthHeader>
        <Brand to={ROUTES.HOME}>
          <BrandIcon>
            <FiActivity size={18} />
          </BrandIcon>
          <BrandTitle>Skin Disease AI</BrandTitle>
        </Brand>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Link
            to={ROUTES.HOME}
            style={{
              fontSize: '0.85rem',
              fontWeight: 600,
              color: 'inherit',
              textDecoration: 'none',
              opacity: 0.8,
            }}
          >
            ← Back to Home
          </Link>
          <ThemeToggle />
        </div>
      </AuthHeader>

      <SplitLayout>
        {/* Left Side: Medical AI Tech Highlights */}
        <ShowcasePanel>
          <BadgePill>
            <FiShield size={13} />
            Clinical AI Research • HAM10000 Benchmarks
          </BadgePill>

          <ShowcaseTitle>
            Dermoscopic Lesion Classification with <span className="highlight">Ensemble Deep Learning</span>
          </ShowcaseTitle>

          <ShowcaseDesc>
            Integrated AI diagnostic system trained on 10,015 dermatoscopy images across 7 clinical skin disease categories, utilizing ResNet101, DenseNet121, and EfficientNetB3 with live MongoDB persistence.
          </ShowcaseDesc>

          <FeaturesGrid>
            <FeatureCard>
              <div className="header">
                <FiLayers size={16} />
                <span>3-Model Stacking</span>
              </div>
              <span className="desc">
                Ensemble consensus combining deep feature extractors with a meta-classifier.
              </span>
            </FeatureCard>

            <FeatureCard>
              <div className="header">
                <FiDatabase size={16} />
                <span>MongoDB Atlas Sync</span>
              </div>
              <span className="desc">
                Instant persistence for prediction logs, confidence scores, and evaluator settings.
              </span>
            </FeatureCard>

            <FeatureCard>
              <div className="header">
                <FiCheckCircle size={16} />
                <span>7 Diagnostic Classes</span>
              </div>
              <span className="desc">
                Screening for Melanoma, Basal Cell Carcinoma, Actinic Keratoses, and Benign Nevi.
              </span>
            </FeatureCard>

            <FeatureCard>
              <div className="header">
                <FiShield size={16} />
                <span>Pre-filled Evaluator</span>
              </div>
              <span className="desc">
                Instant test account access for zero-friction recruiter and peer evaluation.
              </span>
            </FeatureCard>
          </FeaturesGrid>
        </ShowcasePanel>

        {/* Right Side: Clean Form Container */}
        <FormColumn>
          <AuthCard>{children}</AuthCard>
        </FormColumn>
      </SplitLayout>
    </PageWrapper>
  );
};

export default AuthLayout;

