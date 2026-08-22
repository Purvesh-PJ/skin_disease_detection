import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { ThemeToggle } from '../common/ui';
import { ROUTES } from '../../constants';
import { FiActivity } from 'react-icons/fi';

const PageWrapper = styled.div`
  height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background-color: ${({ theme }) => theme.colors.background.primary};
  color: ${({ theme }) => theme.colors.text.primary};
  overflow: hidden;
`;

const AuthHeader = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[8]}`};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};

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
  width: 32px;
  height: 32px;
  background: ${({ theme }) => theme.colors.primary[600]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
`;

const BrandTitle = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.1rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.02em;
`;

const SplitLayout = styled.main`
  flex: 1;
  max-width: 1100px;
  width: 100%;
  margin: 0 auto;
  padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[6]}`};
  display: grid;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[8]};
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[4]};
    overflow-y: auto;
  }
`;

const IllustrationSide = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const SvgGraphic = styled.div`
  width: 220px;
  height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;

  svg {
    width: 100%;
    height: 100%;
  }
`;

const IllustrationTitle = styled.h2`
  font-size: 1.4rem;
  font-weight: 700;
  margin: 0;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.02em;
`;

const IllustrationSubtitle = styled.p`
  font-size: 0.88rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: 0;
  max-width: 360px;
  line-height: 1.5;
`;

const FormColumn = styled.div`
  display: flex;
  justify-content: center;
  width: 100%;
`;

const FormBox = styled.div`
  width: 100%;
  max-width: 380px;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
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
            ← Home
          </Link>
          <ThemeToggle />
        </div>
      </AuthHeader>

      <SplitLayout>
        {/* Left Side: Clean Flat Graphic & Simple Intro */}
        <IllustrationSide>
          <SvgGraphic>
            <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="100" cy="100" r="85" fill="#16a34a" fillOpacity="0.08" stroke="#16a34a" strokeWidth="2" strokeDasharray="4 4" />
              <rect x="50" y="55" width="100" height="90" rx="16" fill="#16a34a" fillOpacity="0.15" stroke="#16a34a" strokeWidth="2" />
              <circle cx="100" cy="100" r="28" fill="#16a34a" fillOpacity="0.25" />
              <circle cx="100" cy="100" r="14" fill="#16a34a" />
              <path d="M70 100H86M114 100H130M100 70V86M100 114V130" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" />
              <path d="M60 40L75 55M140 40L125 55M60 160L75 145M140 160L125 145" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </SvgGraphic>

          <IllustrationTitle>Skin Lesion AI Classifier</IllustrationTitle>
          <IllustrationSubtitle>
            Fast, non-invasive dermoscopic screening powered by ensemble neural networks.
          </IllustrationSubtitle>
        </IllustrationSide>

        {/* Right Side: Clean Flat Form */}
        <FormColumn>
          <FormBox>{children}</FormBox>
        </FormColumn>
      </SplitLayout>
    </PageWrapper>
  );
};

export default AuthLayout;


