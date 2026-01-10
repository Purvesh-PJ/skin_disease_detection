import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.secondary};
`;

export const Main = styled.main`
  flex: 1;
  display: flex;
  gap: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[4]};
  
  @media (min-width: 1400px) {
    padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[10]}`};
  }
  
  @media (min-width: 1600px) {
    padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[16]}`};
  }
  
  @media (min-width: 1920px) {
    padding: ${({ theme }) => `${theme.spacing[8]} 80px`};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex-direction: column;
  }
`;

export const Panel = styled.div`
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  box-shadow: ${({ theme }) => theme.shadows.subtle};
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

export const LeftPanel = styled(Panel)`
  flex: 0 0 45%;
  
  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex: none;
  }
`;

export const RightPanel = styled(Panel)`
  flex: 1;
`;

export const PanelHeader = styled.div`
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
`;

export const PanelTitle = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  color: ${({ theme }) => theme.colors.text.primary};
  
  h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
  }
  
  svg {
    color: ${({ theme }) => theme.colors.primary[500]};
  }
`;

export const PanelContent = styled.div`
  flex: 1;
  padding: ${({ theme }) => theme.spacing[4]};
  display: flex;
  flex-direction: column;
`;
