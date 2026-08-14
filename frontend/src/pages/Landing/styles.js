import styled from 'styled-components';

export const LandingPageWrapper = styled.div`
  min-height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.secondary};
  color: ${({ theme }) => theme.colors.text.primary};
  overflow-x: hidden;
`;

export const HeroSection = styled.section`
  position: relative;
  padding: ${({ theme }) => `${theme.spacing[16]} ${theme.spacing[6]} ${theme.spacing[20]}`};
  max-width: 1280px;
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
  height: 500px;
  background: ${({ theme }) => theme.gradients.heroGlow};
  pointer-events: none;
  z-index: 0;
`;

export const HeroPillBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: 6px 16px;
  background: ${({ theme }) => theme.gradients.heroBadge};
  border: 1px solid ${({ theme }) => theme.colors.primary[200]};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => theme.colors.primary[700]};
  font-size: 0.8125rem;
  font-weight: 600;
  margin-bottom: ${({ theme }) => theme.spacing[6]};
  position: relative;
  z-index: 1;
`;

export const HeroTitle = styled.h1`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 3.75rem;
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.035em;
  max-width: 900px;
  margin: 0 auto ${({ theme }) => theme.spacing[6]};
  position: relative;
  z-index: 1;

  span {
    background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary[600]}, ${({ theme }) => theme.colors.primary[400]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: 2.75rem;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: 2.15rem;
  }
`;

export const HeroSubtitle = styled.p`
  font-size: 1.2rem;
  line-height: 1.6;
  color: ${({ theme }) => theme.colors.text.secondary};
  max-width: 680px;
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

export const PreviewContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 960px;
  margin: 0 auto;
  z-index: 1;
`;

export const PreviewCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  box-shadow: ${({ theme }) => theme.shadows.floating};
  padding: ${({ theme }) => `${theme.spacing[8]} ${theme.spacing[8]}`};
  text-align: left;
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: ${({ theme }) => theme.spacing[8]};
  align-items: center;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    padding: ${({ theme }) => theme.spacing[6]};
  }
`;

export const PreviewLesionImage = styled.div`
  position: relative;
  aspect-ratio: 4 / 3;
  background: linear-gradient(135deg, #1e293b, #0f172a);
  border-radius: ${({ theme }) => theme.borderRadius.card};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.1);

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
`;

export const ScanningOverlay = styled.div`
  position: absolute;
  inset: 0;
  border: 2px dashed rgba(14, 165, 233, 0.6);
  border-radius: ${({ theme }) => theme.borderRadius.card};
  margin: 12px;
  pointer-events: none;
`;

export const PreviewDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

export const PreviewPillTag = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: ${({ theme }) => theme.colors.status.success.bg};
  border: 1px solid ${({ theme }) => theme.colors.status.success.border};
  color: ${({ theme }) => theme.colors.status.success.text};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.8rem;
  font-weight: 700;
  width: fit-content;
`;

export const MetricsRow = styled.div`
  max-width: 1100px;
  margin: 0 auto ${({ theme }) => theme.spacing[20]};
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: ${({ theme }) => theme.spacing[4]};
  padding: 0 ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: repeat(2, 1fr);
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    grid-template-columns: 1fr;
  }
`;

export const MetricCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => theme.spacing[6]};
  text-align: center;
  box-shadow: ${({ theme }) => theme.shadows.paper};
  transition: transform ${({ theme }) => theme.transitions.fast};

  &:hover {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.hover};
  }
`;

export const MetricValue = styled.div`
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: 4px;
`;

export const MetricLabel = styled.div`
  font-size: 0.875rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  font-weight: 500;
`;

export const Section = styled.section`
  max-width: 1280px;
  margin: 0 auto ${({ theme }) => theme.spacing[24]};
  padding: 0 ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: 0 ${({ theme }) => theme.spacing[4]};
    margin-bottom: ${({ theme }) => theme.spacing[16]};
  }
`;

export const SectionHeader = styled.div`
  text-align: center;
  max-width: 720px;
  margin: 0 auto ${({ theme }) => theme.spacing[12]};
`;

export const SectionPill = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  background: ${({ theme }) => theme.colors.interactive.hover};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 0.8125rem;
  font-weight: 600;
  margin-bottom: ${({ theme }) => theme.spacing[3]};
`;

export const SectionTitle = styled.h2`
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[3]};

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    font-size: 2rem;
  }
`;

export const SectionDescription = styled.p`
  font-size: 1.1rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.6;
`;

export const StepsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const StepCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
  transition: all ${({ theme }) => theme.transitions.normal};

  &:hover {
    transform: translateY(-3px);
    box-shadow: ${({ theme }) => theme.shadows.hover};
    border-color: ${({ theme }) => theme.colors.primary[200]};
  }
`;

export const StepNumberPill = styled.div`
  width: 42px;
  height: 42px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  background: ${({ theme }) => theme.colors.interactive.hover};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: 800;
  color: ${({ theme }) => theme.colors.primary[600]};
`;

export const StepTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text.primary};
  letter-spacing: -0.01em;
`;

export const StepText = styled.p`
  font-size: 0.95rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: 1.6;
`;

export const ModelsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[6]};
  margin-bottom: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
  }
`;

export const ModelCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export const EnsembleBannerCard = styled.div`
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.background.primary} 0%, ${({ theme }) => theme.colors.background.tertiary} 100%);
  border: 1px solid ${({ theme }) => theme.colors.primary[200]};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => theme.spacing[8]};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    flex-direction: column;
    align-items: flex-start;
  }
`;

export const ConditionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
`;

export const ConditionCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.card};
  padding: ${({ theme }) => `${theme.spacing[5]} ${theme.spacing[6]}`};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[300]};
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.hover};
  }
`;

export const ConditionCategoryPill = styled.span`
  display: inline-block;
  padding: 3px 10px;
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
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  font-size: 0.75rem;
  font-weight: 700;
  width: fit-content;
`;

export const SafetyCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.container};
  padding: ${({ theme }) => `${theme.spacing[10]} ${theme.spacing[8]}`};
  box-shadow: ${({ theme }) => theme.shadows.paper};
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: ${({ theme }) => theme.spacing[8]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    grid-template-columns: 1fr;
    gap: ${({ theme }) => theme.spacing[6]};
  }
`;

export const SafetyItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

export const CtaSection = styled.section`
  max-width: 1280px;
  margin: 0 auto ${({ theme }) => theme.spacing[20]};
  padding: 0 ${({ theme }) => theme.spacing[6]};
`;

export const CtaCard = styled.div`
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neutral[900]}, #0c4a6e);
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
  max-width: 1280px;
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
