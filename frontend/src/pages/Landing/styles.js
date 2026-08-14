import styled from 'styled-components';

export const LandingPageWrapper = styled.div`
  min-height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.secondary};
  color: ${({ theme }) => theme.colors.text.primary};
  overflow-x: hidden;
`;

// Hero Section
export const HeroSection = styled.section`
  position: relative;
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[6]} ${theme.spacing[16]}`};
  max-width: 1280px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[4]} ${theme.spacing[10]}`};
  }
`;

export const HeroGlow = styled.div`
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 100vw;
  height: 480px;
  background: ${({ theme }) => theme.gradients.heroGlow};
  pointer-events: none;
  z-index: 0;
`;

export const HeroBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: 6px 16px;
  background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(24, 24, 24, 0.8)' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#bbf7d0')};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#15803d')};
  font-size: 0.8125rem;
  font-weight: 600;
  margin-bottom: ${({ theme }) => theme.spacing[6]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  position: relative;
  z-index: 1;
`;

export const HeroTitle = styled.h1`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 3.5rem;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  max-width: 920px;
  margin: 0 auto ${({ theme }) => theme.spacing[6]};
  color: ${({ theme }) => theme.colors.text.primary};
  position: relative;
  z-index: 1;

  span.highlight {
    color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: 2.5rem;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 1.95rem;
  }
`;

export const HeroSubtitle = styled.p`
  font-size: 1.15rem;
  line-height: 1.65;
  color: ${({ theme }) => theme.colors.text.secondary};
  max-width: 720px;
  margin: 0 auto ${({ theme }) => theme.spacing[8]};
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 1rem;
  }
`;

export const HeroCtaRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing[4]};
  margin-bottom: ${({ theme }) => theme.spacing[10]};
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    flex-direction: column;
    width: 100%;

    button, a {
      width: 100%;
    }
  }
`;

// Large Outlined Architecture Pipeline Container
export const PipelineSvgContainer = styled.div`
  width: 100%;
  max-width: 1140px;
  background: ${({ theme }) => (theme.mode === 'dark' ? '#141414' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[6]}`};
  position: relative;
  z-index: 1;
  overflow-x: auto;
  transition: border-color ${({ theme }) => theme.transitions.normal};

  svg {
    width: 100%;
    min-width: 900px;
    height: auto;
    display: block;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => theme.spacing[4]};
  }
`;

// Common Section Styles (Smooth Natural Flow)
export const SectionWrapper = styled.section`
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[6]}`};
  background-color: ${({ theme, $alt }) =>
    $alt ? (theme.mode === 'dark' ? '#121212' : '#ffffff') : 'transparent'};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[12]} ${theme.spacing[4]}`};
  }
`;

export const Container = styled.div`
  max-width: 1240px;
  margin: 0 auto;
`;

export const SectionHeader = styled.div`
  text-align: center;
  max-width: 760px;
  margin: 0 auto ${({ theme }) => theme.spacing[10]};
`;

export const SectionTag = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : '#f5f5f5')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 0.8125rem;
  font-weight: 600;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
`;

export const SectionTitle = styled.h2`
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    font-size: 1.85rem;
  }
`;

export const SectionDescription = styled.p`
  font-size: 1.05rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.6;
`;

// Dataset & Stats Grid
export const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing[5]};
  margin-bottom: ${({ theme }) => theme.spacing[10]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: repeat(2, 1fr);
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

export const StatCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[4]}`};
  text-align: center;
  transition: transform ${({ theme }) => theme.transitions.fast}, border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => (theme.mode === 'dark' ? '#404040' : '#d4d4d4')};
    transform: translateY(-2px);
  }
`;

export const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 800;
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  letter-spacing: -0.02em;
  margin-bottom: 4px;
`;

export const StatLabel = styled.div`
  font-size: 0.85rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  font-weight: 500;
`;

// Dataset 3-Step Process
export const DatasetProcessGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const ProcessCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => theme.spacing[7]};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  transition: transform ${({ theme }) => theme.transitions.fast}, border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => (theme.mode === 'dark' ? '#404040' : '#d4d4d4')};
    transform: translateY(-2px);
  }
`;

export const ProcessIcon = styled.div`
  width: 42px;
  height: 42px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.12)' : '#f0fdf4')};
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.25)' : '#bbf7d0')};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.15rem;
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

// Models Architecture Grid
export const ModelsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};
  margin-bottom: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const ModelCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => theme.spacing[7]};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
  transition: transform ${({ theme }) => theme.transitions.fast}, border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => (theme.mode === 'dark' ? '#404040' : '#d4d4d4')};
    transform: translateY(-2px);
  }
`;

export const ModelHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

export const ModelBadge = styled.span`
  display: inline-block;
  padding: 4px 12px;
  background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.12)' : '#f0fdf4')};
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#15803d')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.25)' : '#bbf7d0')};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.75rem;
  font-weight: 700;
`;

export const ModelSvgWrapper = styled.div`
  width: 100%;
  background: ${({ theme }) => (theme.mode === 'dark' ? '#121212' : '#f9fafb')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => theme.spacing[4]};
  display: flex;
  align-items: center;
  justify-content: center;

  svg {
    width: 100%;
    max-height: 120px;
    display: block;
  }
`;

// Ensemble Highlight Banner
export const EnsembleBanner = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#bbf7d0')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[8]}`};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: ${({ theme }) => theme.spacing[6]};
  align-items: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    padding: ${({ theme }) => theme.spacing[6]};
  }
`;

// Conditions Grid
export const ConditionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: ${({ theme }) => theme.spacing[5]};
`;

export const ConditionCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => theme.spacing[6]};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.4)' : '#86efac')};
    transform: translateY(-2px);
  }
`;

export const ConditionPill = styled.span`
  display: inline-block;
  padding: 3px 10px;
  background: ${({ theme, $type }) => {
    if ($type === 'danger') return theme.colors.status.error.bg;
    if ($type === 'warning') return theme.colors.status.warning.bg;
    return theme.colors.status.success.bg;
  }};
  color: ${({ theme, $type }) => {
    if ($type === 'danger') return theme.colors.status.error.text;
    if ($type === 'warning') return theme.colors.status.warning.text;
    return theme.colors.status.success.text;
  }};
  border: 1px solid ${({ theme, $type }) => {
    if ($type === 'danger') return theme.colors.status.error.border;
    if ($type === 'warning') return theme.colors.status.warning.border;
    return theme.colors.status.success.border;
  }};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.75rem;
  font-weight: 700;
  width: fit-content;
`;

// Disclaimer Section
export const DisclaimerCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => theme.spacing[8]};
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
    padding: ${({ theme }) => theme.spacing[6]};
  }
`;

export const DisclaimerItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

// CTA Section
export const CtaCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#171717')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#262626')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[14]} ${theme.spacing[8]}`};
  color: white;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  box-shadow: ${({ theme }) => theme.shadows.paper};
`;

// Footer
export const FooterWrapper = styled.footer`
  border-top: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#1f1f1f' : '#e5e5e5')};
  background-color: ${({ theme }) => (theme.mode === 'dark' ? '#121212' : '#ffffff')};
  padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[6]} ${theme.spacing[8]}`};
`;

export const FooterContainer = styled.div`
  max-width: 1240px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing[4]};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    flex-direction: column;
    text-align: center;
  }
`;
