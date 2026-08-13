import styled from 'styled-components';
import { ThemeToggle } from '../common/ui';

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

const Brand = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const BrandIcon = styled.div`
  width: 38px;
  height: 38px;
  background: ${({ theme }) => theme.gradients.brandIcon};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 800;
  font-size: 1rem;
  box-shadow: ${({ theme }) => theme.shadows.md};
`;

const BrandTitle = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.2rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.01em;
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
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid ${({ theme }) => theme.gradients.authCardBorder};
  border-radius: ${({ theme }) => theme.borderRadius['2xl']};
  box-shadow: ${({ theme }) => theme.shadows.xl};
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
        <Brand>
          <BrandIcon>SP</BrandIcon>
          <BrandTitle>Skin AI Detector</BrandTitle>
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
