import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { ThemeToggle } from '../common/ui';
import { ROUTES } from '../../constants';
import { FiActivity } from 'react-icons/fi';

const PageWrapper = styled.div`
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.gradients.authBg};
  transition: background ${({ theme }) => theme.transitions.normal};
`;

const AuthHeader = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[8]}`};

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
  width: 36px;
  height: 36px;
  background: ${({ theme }) => theme.gradients.brandIcon};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
`;

const BrandTitle = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.15rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.02em;
`;

const MainContent = styled.main`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing[4]};
`;

const AuthCard = styled.div`
  width: 100%;
  max-width: 440px;
  background: ${({ theme }) => theme.gradients.authCardBg};
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid ${({ theme }) => theme.gradients.authCardBorder};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  box-shadow: ${({ theme }) => theme.shadows.floating};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[8]}`};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[4]}`};
  }
`;

export const AuthLayout = ({ children }) => {
  return (
    <PageWrapper>
      <AuthHeader>
        <Brand to={ROUTES.HOME}>
          <BrandIcon>
            <FiActivity size={20} />
          </BrandIcon>
          <BrandTitle>Skin AI</BrandTitle>
        </Brand>
        <ThemeToggle />
      </AuthHeader>
      <MainContent>
        <AuthCard>{children}</AuthCard>
      </MainContent>
    </PageWrapper>
  );
};

export default AuthLayout;
