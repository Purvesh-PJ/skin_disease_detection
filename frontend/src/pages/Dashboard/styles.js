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
  gap: ${({ theme }) => theme.spacing[6]};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[8]}`};
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex-direction: column;
    padding: ${({ theme }) => theme.spacing[4]};
  }
`;

export const Panel = styled.div`
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 560px;
  transition: all ${({ theme }) => theme.transitions.normal};
`;

export const LeftPanel = styled(Panel)`
  flex: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex: none;
    min-height: auto;
  }
`;

export const RightPanel = styled(Panel)`
  flex: 1.2;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    flex: none;
    min-height: auto;
  }
`;

export const PanelHeader = styled.div`
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.primary};
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
`;

export const TabGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  padding: 4px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

export const TabButton = styled.button`
  background: ${({ $active, theme }) => 
    $active ? theme.colors.background.primary : 'transparent'};
  color: ${({ $active, theme }) => 
    $active ? theme.colors.primary[600] || '#16a34a' : theme.colors.text.secondary};
  border: none;
  padding: 6px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  box-shadow: ${({ $active }) => ($active ? '0 1px 4px rgba(0,0,0,0.1)' : 'none')};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.text.primary};
  }
`;

export const PanelTitle = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  color: ${({ theme }) => theme.colors.text.primary};
  
  h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  
  svg {
    color: ${({ theme }) => theme.colors.primary[500]};
  }
`;

export const PanelContent = styled.div`
  flex: 1;
  padding: ${({ theme }) => theme.spacing[6]};
  display: flex;
  flex-direction: column;
`;

