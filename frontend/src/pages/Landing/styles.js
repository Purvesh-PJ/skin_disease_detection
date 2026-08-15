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
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[6]} ${theme.spacing[12]}`};
  max-width: 1240px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[4]} ${theme.spacing[8]}`};
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

export const HeroSplitLayout = styled.div`
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: ${({ theme }) => theme.spacing[8]};
  align-items: center;
  width: 100%;
  max-width: 1240px;
  position: relative;
  z-index: 1;
  margin-bottom: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr;
    text-align: center;
    gap: ${({ theme }) => theme.spacing[8]};
  }
`;

export const HeroContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  text-align: left;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    align-items: center;
    text-align: center;
  }
`;

export const HeroVisual = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  position: relative;
`;

export const VisualCard = styled.div`
  width: 100%;
  max-width: 500px;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  box-shadow: none;
  position: relative;

  svg {
    width: 100%;
    height: auto;
    display: block;
  }
`;

export const HeroBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: 6px 16px;
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#ffffff')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#bbf7d0')};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#15803d')};
  font-size: 0.8125rem;
  font-weight: 600;
  margin-bottom: ${({ theme }) => theme.spacing[5]};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  position: relative;
  z-index: 1;
`;

export const HeroTitle = styled.h1`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 3.15rem;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  max-width: 680px;
  margin: 0 0 ${({ theme }) => theme.spacing[4]};
  color: ${({ theme }) => theme.colors.text.primary};
  position: relative;
  z-index: 1;

  span.highlight {
    color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: 2.35rem;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 1.95rem;
  }
`;

export const HeroSubtitle = styled.p`
  font-size: 1.1rem;
  line-height: 1.65;
  color: ${({ theme }) => theme.colors.text.secondary};
  max-width: 620px;
  margin: 0 0 ${({ theme }) => theme.spacing[6]};
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 0.95rem;
  }
`;

export const HeroCtaRow = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[4]};
  margin-bottom: 0;
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    justify-content: center;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    flex-direction: column;
    width: 100%;

    button, a {
      width: 100%;
    }
  }
`;

// Hero Overview Highlights (4 Key Pillars)
export const HeroPillarsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};
  width: 100%;
  max-width: 1040px;
  margin-top: ${({ theme }) => theme.spacing[4]};
  position: relative;
  z-index: 1;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: repeat(2, 1fr);
    gap: ${({ theme }) => theme.spacing[4]};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

export const HeroPillar = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => `${theme.spacing[2]} 0`};
  background: transparent;
  border: none;
  text-align: left;
`;

export const PillarIcon = styled.div`
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.35rem;
  flex-shrink: 0;
  background: transparent;
`;

export const PillarText = styled.div`
  display: flex;
  flex-direction: column;

  strong {
    font-size: 0.85rem;
    font-weight: 700;
    color: ${({ theme }) => theme.colors.text.primary};
  }

  span {
    font-size: 0.75rem;
    color: ${({ theme }) => theme.colors.text.secondary};
  }
`;

// Common Section Wrapper
export const SectionWrapper = styled.section`
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[6]}`};
  background-color: ${({ theme, $alt }) =>
    $alt ? (theme.mode === 'dark' ? '#121212' : '#ffffff') : 'transparent'};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[12]} ${theme.spacing[4]}`};
  }
`;

export const Container = styled.div`
  max-width: 1140px;
  margin: 0 auto;
`;

export const SectionHeader = styled.div`
  text-align: center;
  max-width: 780px;
  margin: 0 auto ${({ theme }) => theme.spacing[10]};
`;

export const SectionTag = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  background: ${({ theme }) => (theme.mode === 'dark' ? '#1e1e1e' : '#f5f5f5')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#2e2e2e' : '#e5e5e5')};
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
  line-height: 1.65;
`;

// Outlined Pipeline SVG Container
export const PipelineSvgContainer = styled.div`
  width: 100%;
  background: ${({ theme }) => (theme.mode === 'dark' ? '#141414' : '#fafafa')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[6]}`};
  margin-top: ${({ theme }) => theme.spacing[8]};
  overflow-x: auto;

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

// Fluid Horizontal Stepper / User Journey
export const ProcessFlow = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};
  position: relative;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: repeat(2, 1fr);
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

export const ProcessStep = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2.5]};
`;

export const StepNumber = styled.div`
  font-size: 0.8125rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
  display: flex;
  align-items: center;
  gap: 8px;

  &::after {
    content: '';
    flex: 1;
    height: 1px;
    background: ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  }
`;

export const StepTitle = styled.h3`
  font-size: 1.15rem;
  font-weight: 700;
  margin: 0;
  color: ${({ theme }) => theme.colors.text.primary};
`;

export const StepDesc = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 0.9rem;
  line-height: 1.6;
  margin: 0;
`;

// Technical Architecture Specs (Model Rows)
export const ModelSpecsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[10]};
`;

export const ModelSpecRow = styled.div`
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: ${({ theme }) => theme.spacing[10]};
  align-items: center;
  padding-bottom: ${({ theme }) => theme.spacing[10]};

  ${({ $reverse }) =>
    $reverse &&
    `
    grid-template-columns: 0.85fr 1.15fr;
    & > :first-child {
      order: 2;
    }
    & > :last-child {
      order: 1;
    }
  `}

  &:not(:last-child) {
    border-bottom: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    grid-template-columns: 1fr !important;
    gap: ${({ theme }) => theme.spacing[6]};

    & > :first-child {
      order: 1 !important;
    }
    & > :last-child {
      order: 2 !important;
    }
  }
`;

export const ModelInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export const ModelTitleRow = styled.div`
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export const ModelBadge = styled.span`
  display: inline-block;
  padding: 3px 10px;
  background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.12)' : '#f0fdf4')};
  color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#15803d')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.25)' : '#bbf7d0')};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.75rem;
  font-weight: 700;
`;

export const ModelSvgPanel = styled.div`
  width: 100%;
  max-width: 440px;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto;

  svg {
    width: 100%;
    height: auto;
    max-height: 140px;
    display: block;
  }
`;

// SECTION 4: DATASET & METHODOLOGY (Card-Free, Open, Spacious)
export const DatasetStatsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};
  padding: ${({ theme }) => `${theme.spacing[6]} 0`};
  border-top: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  border-bottom: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  margin-bottom: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: repeat(2, 1fr);
    gap: ${({ theme }) => theme.spacing[4]};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

export const StatBlock = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;

  .value {
    font-size: 2rem;
    font-weight: 800;
    color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#16a34a')};
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .label {
    font-size: 0.8125rem;
    color: ${({ theme }) => theme.colors.text.secondary};
    font-weight: 600;
  }
`;

export const DistributionContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
  margin-bottom: ${({ theme }) => theme.spacing[10]};

  .dist-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.825rem;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.text.secondary};
  }

  .dist-track {
    display: flex;
    height: 12px;
    border-radius: 6px;
    overflow: hidden;
    background: ${({ theme }) => (theme.mode === 'dark' ? '#262626' : '#e5e5e5')};
  }

  .dist-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    font-size: 0.775rem;
    color: ${({ theme }) => theme.colors.text.secondary};
  }

  .dist-pill-item {
    display: flex;
    align-items: center;
    gap: 6px;

    span.dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
    }
  }
`;

export const PipelineStagesRow = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[8]};
  position: relative;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
  }
`;

export const PipelineStageItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};

  .stage-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.15)' : '#f0fdf4')};
    color: ${({ theme }) => (theme.mode === 'dark' ? '#4ade80' : '#15803d')};
    border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#bbf7d0')};
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 4px;
  }

  h4 {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0;
    color: ${({ theme }) => theme.colors.text.primary};
  }

  p {
    font-size: 0.875rem;
    color: ${({ theme }) => theme.colors.text.secondary};
    line-height: 1.6;
    margin: 0;
  }
`;

// SECTION 5: 7 SUPPORTED SKIN DISEASES (Interactive, Minimalist, Card-Free List)
export const FilterTabsContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: ${({ theme }) => theme.spacing[6]};
  flex-wrap: wrap;
`;

export const FilterTabButton = styled.button`
  padding: 8px 16px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.825rem;
  font-weight: 600;
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  display: flex;
  align-items: center;
  gap: 6px;

  ${({ $active, theme }) =>
    $active
      ? `
    background: ${theme.mode === 'dark' ? '#22c55e' : '#16a34a'};
    color: #ffffff;
    border: 1px solid ${theme.mode === 'dark' ? '#22c55e' : '#16a34a'};
  `
      : `
    background: transparent;
    color: ${theme.colors.text.secondary};
    border: 1px solid ${theme.mode === 'dark' ? '#262626' : '#e5e5e5'};

    &:hover {
      background: ${theme.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.03)'};
      color: ${theme.colors.text.primary};
    }
  `}
`;

export const ConditionsList = styled.div`
  display: flex;
  flex-direction: column;
`;

export const ConditionListRow = styled.div`
  display: grid;
  grid-template-columns: 80px 1fr auto;
  gap: ${({ theme }) => theme.spacing[6]};
  align-items: center;
  padding: ${({ theme }) => `${theme.spacing[4]} 0`};
  border-bottom: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  transition: background-color ${({ theme }) => theme.transitions.fast};

  &:first-child {
    border-top: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};
  }

  .code-col {
    display: flex;
    align-items: center;
    justify-content: flex-start;
  }

  .code-pill {
    font-size: 0.775rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: ${({ theme }) => theme.borderRadius.pill};
    background: ${({ $color }) => `${$color}15`};
    color: ${({ $color }) => $color};
    border: 1px solid ${({ $color }) => `${$color}35`};
    text-transform: uppercase;
  }

  .desc-col {
    display: flex;
    flex-direction: column;
    gap: 3px;

    h4 {
      font-size: 1rem;
      font-weight: 700;
      margin: 0;
      color: ${({ theme }) => theme.colors.text.primary};
    }

    p {
      font-size: 0.85rem;
      color: ${({ theme }) => theme.colors.text.secondary};
      line-height: 1.5;
      margin: 0;
    }
  }

  .risk-col {
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }

  .risk-tag {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: ${({ theme }) => theme.borderRadius.pill};
    background: ${({ $color }) => `${$color}12`};
    color: ${({ $color }) => $color};
    border: 1px solid ${({ $color }) => `${$color}30`};
    white-space: nowrap;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[2]};

    .risk-col {
      justify-content: flex-start;
    }
  }
`;

// Notice Strip
export const NoticeStrip = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[8]};
  padding: ${({ theme }) => `${theme.spacing[6]} 0`};
  border-top: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#222222' : '#eeeeee')};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
  }
`;

export const NoticeItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

// Bottom Action Banner
export const CtaCard = styled.div`
  background: ${({ theme }) => (theme.mode === 'dark' ? '#181818' : '#171717')};
  border: 1px solid ${({ theme }) => (theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : '#262626')};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[12]} ${theme.spacing[8]}`};
  color: white;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

// Footer
export const FooterWrapper = styled.footer`
  border-top: 1px solid ${({ theme }) => (theme.mode === 'dark' ? '#1f1f1f' : '#e5e5e5')};
  background-color: ${({ theme }) => (theme.mode === 'dark' ? '#121212' : '#ffffff')};
  padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[6]} ${theme.spacing[8]}`};
`;

export const FooterContainer = styled.div`
  max-width: 1140px;
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
