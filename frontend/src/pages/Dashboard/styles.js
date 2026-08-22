import styled from 'styled-components';

export const Container = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: ${({ theme }) =>
    theme.mode === 'dark' ? '#0d1117' : '#f8fafc'};
  color: ${({ theme }) => theme.colors.text.primary};
`;

export const Main = styled.main`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[6]}`};
  max-width: 1460px;
  margin: 0 auto;
  width: 100%;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[3]}`};
  }
`;

export const WorkbenchToolbar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 2px 4px;

  .toolbar-left {
    display: flex;
    align-items: center;
    gap: 12px;

    h2 {
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin: 0;
      color: ${({ theme }) => theme.colors.text.primary};
      display: flex;
      align-items: center;
      gap: 8px;
    }

    span.live-dot {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      color: ${({ theme }) => theme.colors.primary[600] || '#16a34a'};
      background: ${({ theme }) =>
        theme.mode === 'dark' ? 'rgba(22, 163, 74, 0.15)' : 'rgba(220, 252, 231, 0.8)'};
      border: 1px solid ${({ theme }) => theme.colors.primary[500] || '#16a34a'};
      padding: 2px 10px;
      border-radius: 9999px;

      span.pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #22c55e;
      }
    }
  }

  .toolbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.text.secondary};

    @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
      display: none;
    }
  }
`;

export const TechBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: 0.75rem;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

export const WorkbenchContainer = styled.div`
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
  display: grid;
  grid-template-columns: 1fr 1.15fr;
  min-height: calc(100vh - 160px);
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
    min-height: auto;
  }
`;

export const WorkbenchSection = styled.div`
  display: flex;
  flex-direction: column;
  min-width: 0;

  &.left-section {
    border-right: 1px solid ${({ theme }) => theme.colors.border.default};

    @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
      border-right: none;
      border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
    }
  }

  &.right-section {
    background-color: ${({ theme }) =>
      theme.mode === 'dark' ? 'rgba(22, 27, 34, 0.4)' : 'rgba(248, 250, 252, 0.4)'};
  }
`;

export const SectionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.primary};
  gap: 12px;
  flex-wrap: wrap;
`;

export const SectionTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${({ theme }) => theme.colors.text.primary};

  h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }

  svg {
    color: ${({ theme }) => theme.colors.primary[500]};
  }
`;

export const TabGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  padding: 3px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

export const TabButton = styled.button`
  background: ${({ $active, theme }) =>
    $active ? theme.colors.background.primary : 'transparent'};
  color: ${({ $active, theme }) =>
    $active ? theme.colors.primary[600] || '#16a34a' : theme.colors.text.secondary};
  border: none;
  padding: 5px 12px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: 0.8rem;
  font-weight: 700;
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

export const SectionBody = styled.div`
  flex: 1;
  padding: 20px;
  display: flex;
  flex-direction: column;
  overflow-y: auto;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: 14px;
  }
`;


