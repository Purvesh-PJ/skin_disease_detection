import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: ${({ theme }) =>
    theme.mode === 'dark' ? '#0d1117' : '#f8fafc'};
  color: ${({ theme }) => theme.colors.text.primary};
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    height: auto;
    overflow-y: auto;
  }
`;

export const Main = styled.main`
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  gap: ${({ theme }) => theme.spacing[6]};
  height: calc(100vh - 65px);
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
    height: auto;
    overflow: visible;
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
  }
`;

export const LeftColumn = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
  padding-right: ${({ theme }) => theme.spacing[2]};

  &::-webkit-scrollbar {
    width: 5px;
  }
  &::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.default};
    border-radius: 4px;
  }
`;

export const RightColumn = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  border-left: 1px solid ${({ theme }) => theme.colors.border.light};
  padding-left: ${({ theme }) => theme.spacing[6]};
  overflow-y: auto;

  &::-webkit-scrollbar {
    width: 5px;
  }
  &::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.default};
    border-radius: 4px;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    border-left: none;
    padding-left: 0;
    border-top: 1px solid ${({ theme }) => theme.colors.border.light};
    padding-top: ${({ theme }) => theme.spacing[4]};
  }
`;

export const TabBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  padding-bottom: ${({ theme }) => theme.spacing[2]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
`;

export const TabButtons = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

export const TabButton = styled.button`
  background: ${({ $active, theme }) =>
    $active ? theme.colors.primary[600] : 'transparent'};
  color: ${({ $active, theme }) =>
    $active ? '#ffffff' : theme.colors.text.secondary};
  border: 1px solid ${({ $active, theme }) =>
    $active ? theme.colors.primary[600] : theme.colors.border.default};
  padding: 6px 14px;
  border-radius: 9999px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ $active, theme }) => ($active ? '#ffffff' : theme.colors.text.primary)};
    border-color: ${({ theme }) => theme.colors.primary[500]};
  }
`;



