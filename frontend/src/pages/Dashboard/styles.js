import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.primary};
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
  justify-content: center;
  align-items: center;
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

export const SampleFooterRail = styled.div`
  width: 100%;
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  padding: 10px 24px;
  display: flex;
  align-items: center;
  gap: 16px;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: 8px 12px;
    gap: 10px;
  }
`;

export const SampleRailLabel = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.secondary};
  white-space: nowrap;
`;

export const SampleRailList = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  overflow-x: auto;
  width: 100%;
  padding: 2px 0;

  &::-webkit-scrollbar {
    height: 4px;
  }
  &::-webkit-scrollbar-thumb {
    background: ${({ theme }) => theme.colors.border.default};
    border-radius: 4px;
  }
`;

export const SampleCard = styled.button`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1.5px solid ${({ $active, theme }) =>
    $active ? theme.colors.primary[500] : theme.colors.border.default};
  background: ${({ $active, theme }) =>
    $active
      ? (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(220, 252, 231, 0.9)')
      : theme.colors.background.primary};
  color: ${({ theme }) => theme.colors.text.primary};
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.2s ease;

  img {
    width: 44px;
    height: 44px;
    border-radius: 6px;
    object-fit: cover;
  }

  .meta {
    display: flex;
    flex-direction: column;
    align-items: flex-start;

    span.name {
      font-size: 0.75rem;
      font-weight: 700;
    }

    span.type {
      font-size: 0.65rem;
      color: ${({ theme }) => theme.colors.text.tertiary};
    }
  }

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
    transform: translateY(-2px);
  }
`;




