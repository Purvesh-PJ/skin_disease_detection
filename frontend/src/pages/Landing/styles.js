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
  max-width: 1320px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[4]} ${theme.spacing[12]}`};
  }
`;

export const HeroGlow = styled.div`
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 100vw;
  height: 600px;
  background: ${({ theme }) => theme.gradients.heroGlow};
  pointer-events: none;
  z-index: 0;
`;

export const HeroPillBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: 6px 18px;
  background: ${({ theme }) => theme.gradients.heroBadge};
  border: 1px solid ${({ theme }) => theme.colors.border.brand};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => (theme.mode === 'dark' ? theme.colors.emerald.android : '#065f46')};
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: ${({ theme }) => theme.spacing[6]};
  position: relative;
  z-index: 1;

  span.dot {
    width: 8px;
    height: 8px;
    background-color: ${({ theme }) => theme.colors.emerald.android};
    border-radius: 50%;
    box-shadow: 0 0 10px ${({ theme }) => theme.colors.emerald.android};
  }
`;

export const HeroTitle = styled.h1`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 4rem;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: -0.04em;
  max-width: 980px;
  margin: 0 auto ${({ theme }) => theme.spacing[6]};
  position: relative;
  z-index: 1;

  span.highlight {
    background: linear-gradient(135deg, #3ddc84 0%, #059669 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: 2.85rem;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 2.25rem;
  }
`;

export const HeroSubtitle = styled.p`
  font-size: 1.25rem;
  line-height: 1.65;
  color: ${({ theme }) => theme.colors.text.secondary};
  max-width: 740px;
  margin: 0 auto ${({ theme }) => theme.spacing[8]};
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 1.05rem;
  }
`;

export const HeroCtaRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing[4]};
  margin-bottom: ${({ theme }) => theme.spacing[16]};
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

// Interactive Sandbox in Hero
export const SandboxWrapper = styled.div`
  width: 100%;
  max-width: 1120px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
`;

export const SandboxCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  box-shadow: ${({ theme }) => theme.shadows.floating};
  padding: ${({ theme }) => `${theme.spacing[6]} ${theme.spacing[8]}`};
  text-align: left;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.spacing[5]};
  }
`;

export const SandboxTopBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: ${({ theme }) => theme.spacing[5]};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  margin-bottom: ${({ theme }) => theme.spacing[6]};
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export const SamplePillsRow = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  flex-wrap: wrap;
`;

export const SamplePillBtn = styled.button`
  padding: 6px 16px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.emerald.androidDark || '#16a34a' : theme.colors.border.default};
  background-color: ${({ theme, $active }) =>
    $active
      ? theme.colors.emerald[100]
      : theme.colors.background.tertiary};
  color: ${({ theme, $active }) =>
    $active ? '#065f46' : theme.colors.text.secondary};

  &:hover {
    border-color: ${({ theme }) => theme.colors.emerald.android};
  }
`;

export const SandboxGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1.3fr;
  gap: ${({ theme }) => theme.spacing[8]};
  align-items: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
  }
`;

export const LesionDisplayCard = styled.div`
  position: relative;
  aspect-ratio: 4 / 3;
  background: ${({ theme }) => theme.gradients.bentoPine};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
  border: 1px solid rgba(61, 220, 132, 0.25);
  box-shadow: ${({ theme }) => theme.shadows.paper};
`;

export const ScanningReticle = styled.div`
  position: absolute;
  inset: 16px;
  border: 2px dashed rgba(61, 220, 132, 0.7);
  border-radius: ${({ theme }) => theme.borderRadius.md};
  pointer-events: none;
`;

export const ScannerHeaderBadge = styled.div`
  position: absolute;
  top: 12px;
  left: 12px;
  background: rgba(7, 48, 66, 0.85);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(61, 220, 132, 0.4);
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.75rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.emerald.android};
`;

export const ConsensusBreakdown = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

export const ConsensusBarRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

export const BarHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};

  span.val {
    font-weight: 800;
    color: ${({ theme }) => theme.colors.text.primary};
  }
`;

export const BarTrack = styled.div`
  width: 100%;
  height: 8px;
  background: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  overflow: hidden;
`;

export const BarFill = styled.div`
  height: 100%;
  background: ${({ $color, theme }) => $color || theme.gradients.progressBar};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  width: ${({ $width }) => `${$width}%`};
  transition: width 0.6s cubic-bezier(0.2, 0, 0, 1);
`;

// Section Shells
export const SectionWrapper = styled.section`
  padding: ${({ theme }) => `${theme.spacing[20]} ${theme.spacing[6]}`};
  background-color: ${({ theme, $bg }) => {
    if ($bg === 'mint') return theme.colors.background.tonalMint;
    if ($bg === 'indigo') return theme.colors.background.tonalIndigo;
    if ($bg === 'sand') return theme.colors.background.tonalSand;
    if ($bg === 'ice') return theme.colors.background.tonalIce;
    return 'transparent';
  }};
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[12]} ${theme.spacing[4]}`};
  }
`;

export const Container = styled.div`
  max-width: 1320px;
  margin: 0 auto;
`;

export const SectionHeader = styled.div`
  text-align: center;
  max-width: 780px;
  margin: 0 auto ${({ theme }) => theme.spacing[12]};
`;

export const SectionTag = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 16px;
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.brand || theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => (theme.mode === 'dark' ? theme.colors.emerald.android : '#065f46')};
  font-size: 0.8125rem;
  font-weight: 700;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
`;

export const SectionTitle = styled.h2`
  font-size: 2.75rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    font-size: 2.15rem;
  }
`;

export const SectionDescription = styled.p`
  font-size: 1.15rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.6;
`;

// Asymmetric Bento Grid
export const BentoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
  }
`;

export const BentoCardLarge = styled.div`
  grid-column: span 7;
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.bento};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.bento};
  display: flex;
  flex-direction: column;
  justify-content: space-between;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-column: span 12;
  }
`;

export const BentoCardSmall = styled.div`
  grid-column: span 5;
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.bento};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.bento};
  display: flex;
  flex-direction: column;
  justify-content: space-between;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-column: span 12;
  }
`;

// Interactive Model Explorer Tabs
export const TabList = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing[2]};
  margin-bottom: ${({ theme }) => theme.spacing[8]};
  flex-wrap: wrap;
`;

export const TabButton = styled.button`
  padding: 10px 24px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.emerald.androidDark || '#16a34a' : theme.colors.border.default};
  background-color: ${({ theme, $active }) =>
    $active ? theme.colors.button.pine.bg : theme.colors.background.primary};
  color: ${({ theme, $active }) => ($active ? '#3ddc84' : theme.colors.text.secondary)};
  box-shadow: ${({ theme, $active }) =>
    $active ? '0 4px 14px rgba(7, 48, 66, 0.2)' : theme.shadows.sm};

  &:hover {
    border-color: ${({ theme }) => theme.colors.emerald.android};
  }
`;

export const TabContentCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[10]}`};
  box-shadow: ${({ theme }) => theme.shadows.floating};
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: ${({ theme }) => theme.spacing[8]};
  align-items: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    padding: ${({ theme }) => theme.spacing[6]};
  }
`;

// Clinical Pathology Atlas
export const AtlasGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: ${({ theme }) => theme.spacing[6]};
`;

export const AtlasCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => theme.spacing[6]};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.emerald.android};
    transform: translateY(-3px);
    box-shadow: ${({ theme }) => theme.shadows.hover};
  }
`;

export const RiskBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: ${({ theme, $type }) => {
    if ($type === 'malignant') return theme.colors.status.error.bg;
    if ($type === 'pre-malignant') return theme.colors.status.warning.bg;
    return theme.colors.status.success.bg;
  }};
  color: ${({ theme, $type }) => {
    if ($type === 'malignant') return theme.colors.status.error.text;
    if ($type === 'pre-malignant') return theme.colors.status.warning.text;
    return theme.colors.status.success.text;
  }};
  border: 1px solid ${({ theme, $type }) => {
    if ($type === 'malignant') return theme.colors.status.error.border;
    if ($type === 'pre-malignant') return theme.colors.status.warning.border;
    return theme.colors.status.success.border;
  }};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.78rem;
  font-weight: 700;
  width: fit-content;
`;

// Workflow Steps
export const WorkflowGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const WorkflowStep = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export const StepIndexPill = styled.div`
  width: 44px;
  height: 44px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  background: ${({ theme }) => theme.colors.button.pine.bg};
  color: ${({ theme }) => theme.colors.emerald.android};
  font-size: 1.1rem;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(7, 48, 66, 0.2);
`;

// Dark Pine Bottom CTA & Footer
export const DarkCtaSection = styled.section`
  max-width: 1320px;
  margin: 0 auto ${({ theme }) => theme.spacing[20]};
  padding: 0 ${({ theme }) => theme.spacing[6]};
`;

export const DarkCtaCard = styled.div`
  background: ${({ theme }) => theme.gradients.bentoPine};
  border: 1px solid rgba(61, 220, 132, 0.3);
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[8]}`};
  color: white;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  box-shadow: ${({ theme }) => theme.shadows.floating};
`;

export const FooterWrapper = styled.footer`
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.primary};
  padding: ${({ theme }) => `${theme.spacing[12]} ${theme.spacing[6]} ${theme.spacing[8]}`};
`;

export const FooterContainer = styled.div`
  max-width: 1320px;
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
