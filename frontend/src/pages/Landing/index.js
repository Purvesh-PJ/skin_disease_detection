import React from 'react';
import { Link } from 'react-router-dom';
import { ROUTES } from '../../constants';
import { Button } from '../../components/common/ui';
import LandingNavbar from '../../components/layout/LandingNavbar';
import {
  FiArrowRight,
  FiCpu,
  FiLayers,
  FiShield,
  FiZap,
  FiCheckCircle,
  FiLock,
  FiTrendingUp,
  FiHelpCircle,
} from 'react-icons/fi';
import {
  LandingPageWrapper,
  HeroSection,
  HeroGlow,
  HeroPillBadge,
  HeroTitle,
  HeroSubtitle,
  HeroCtaRow,
  PreviewContainer,
  PreviewCard,
  PreviewLesionImage,
  ScanningOverlay,
  PreviewDetails,
  PreviewPillTag,
  MetricsRow,
  MetricCard,
  MetricValue,
  MetricLabel,
  Section,
  SectionHeader,
  SectionPill,
  SectionTitle,
  SectionDescription,
  StepsGrid,
  StepCard,
  StepNumberPill,
  StepTitle,
  StepText,
  ModelsGrid,
  ModelCard,
  EnsembleBannerCard,
  ConditionsGrid,
  ConditionCard,
  ConditionCategoryPill,
  SafetyCard,
  SafetyItem,
  CtaSection,
  CtaCard,
  FooterWrapper,
  FooterContainer,
} from './styles';

const SUPPORTED_CONDITIONS = [
  {
    code: 'mel',
    name: 'Melanoma',
    category: 'Malignant',
    type: 'malignant',
    description: 'High-risk melanocytic malignancy requiring immediate histological biopsy and staging.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    category: 'Malignant',
    type: 'malignant',
    description: 'Most common non-melanoma skin cancer arising from the basal layer of the epidermis.',
  },
  {
    code: 'akiec',
    name: 'Actinic Keratoses',
    category: 'Pre-Malignant',
    type: 'pre-malignant',
    description: 'Dysplastic epidermal lesion caused by chronic ultraviolet radiation exposure.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    category: 'Benign',
    type: 'benign',
    description: 'Non-cancerous epithelial tumors including seborrheic keratosis and solar lentigines.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    category: 'Benign',
    type: 'benign',
    description: 'Common benign cutaneous fibrohistiocytic lesion showing the dimple sign.',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    category: 'Benign',
    type: 'benign',
    description: 'Benign proliferations of melanocytes characterized by symmetrical architectural growth.',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    category: 'Benign',
    type: 'benign',
    description: 'Angiomas, pyogenic granulomas, and vascular malformations of the cutaneous vessels.',
  },
];

const Landing = ({ isAuthenticated }) => {
  const ctaRoute = isAuthenticated ? ROUTES.DASHBOARD : ROUTES.SIGNUP;
  const ctaText = isAuthenticated ? 'Open Diagnosis Dashboard' : 'Start Free AI Analysis';

  return (
    <LandingPageWrapper id="overview">
      {/* Sticky Navigation */}
      <LandingNavbar isAuthenticated={isAuthenticated} />

      {/* Hero Section */}
      <HeroSection>
        <HeroGlow />

        <HeroPillBadge>
          <FiCpu size={14} />
          <span>Deep Neural Ensemble v2.0 • Triple CNN Pipeline</span>
        </HeroPillBadge>

        <HeroTitle>
          Precision Skin Lesion Diagnostics with <span>Multi-Model AI</span>
        </HeroTitle>

        <HeroSubtitle>
          Harness the combined diagnostic power of ResNet-101, DenseNet-121, and EfficientNet-B3.
          Designed to assist medical triage with instantaneous classification and confidence metrics.
        </HeroSubtitle>

        <HeroCtaRow>
          <Button asChild variant="accent" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
          <Button asChild variant="secondary" size="lg">
            <a href="#how-it-works">How It Works</a>
          </Button>
        </HeroCtaRow>

        {/* Live Interactive Hero Demo Card */}
        <PreviewContainer>
          <PreviewCard>
            <PreviewLesionImage>
              <ScanningOverlay />
              <div style={{ textAlign: 'center', padding: '16px', zIndex: 1 }}>
                <FiZap size={32} style={{ color: '#38bdf8', marginBottom: '8px' }} />
                <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>Dermoscopic Ingestion Matrix</div>
                <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>224 x 224 RGB Tensor Input</div>
              </div>
            </PreviewLesionImage>

            <PreviewDetails>
              <PreviewPillTag>
                <FiCheckCircle size={14} />
                <span>98.4% Ensemble Confidence Match</span>
              </PreviewPillTag>

              <div>
                <h3 style={{ fontSize: '1.4rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                  Vascular Lesion (vasc)
                </h3>
                <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>
                  Angiomas, pyogenic granulomas, and benign vascular malformations.
                </p>
              </div>

              {/* Mini Architecture Consensus */}
              <div
                style={{
                  background: 'rgba(0, 0, 0, 0.03)',
                  borderRadius: '16px',
                  padding: '14px',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '8px',
                  textAlign: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>ResNet-101</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0284c7' }}>97.9%</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>DenseNet-121</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0284c7' }}>98.6%</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>EfficientNet-B3</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0284c7' }}>98.8%</div>
                </div>
              </div>
            </PreviewDetails>
          </PreviewCard>
        </PreviewContainer>
      </HeroSection>

      {/* Metrics Row */}
      <MetricsRow>
        <MetricCard>
          <MetricValue>3</MetricValue>
          <MetricLabel>SOTA Deep Neural Networks</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>7</MetricValue>
          <MetricLabel>Dermoscopic Disease Classes</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>&lt; 1.5s</MetricValue>
          <MetricLabel>Real-Time Inference Speed</MetricLabel>
        </MetricCard>
        <MetricCard>
          <MetricValue>94.8%</MetricValue>
          <MetricLabel>Multi-Class Benchmark Score</MetricLabel>
        </MetricCard>
      </MetricsRow>

      {/* Section: How It Works */}
      <Section id="how-it-works">
        <SectionHeader>
          <SectionPill>
            <FiLayers size={14} />
            <span>Workflow & Pipeline</span>
          </SectionPill>
          <SectionTitle>How Clinical-Grade Analysis Works</SectionTitle>
          <SectionDescription>
            A continuous automated medical imaging pipeline engineered to process raw dermoscopic photographs into structured diagnostic findings.
          </SectionDescription>
        </SectionHeader>

        <StepsGrid>
          <StepCard>
            <StepNumberPill>01</StepNumberPill>
            <StepTitle>Image Normalization</StepTitle>
            <StepText>
              High-resolution skin lesion photographs are automatically normalized with CLAHE contrast adjustments and cropped to standard 224x224 tensor matrices.
            </StepText>
          </StepCard>

          <StepCard>
            <StepNumberPill>02</StepNumberPill>
            <StepTitle>Parallel Multi-CNN Inference</StepTitle>
            <StepText>
              Tensors are simultaneously passed through ResNet-101, DenseNet-121, and EfficientNet-B3 models to extract complementary architectural representations.
            </StepText>
          </StepCard>

          <StepCard>
            <StepNumberPill>03</StepNumberPill>
            <StepTitle>Stacked Ensemble Consensus</StepTitle>
            <StepText>
              Our custom Meta-Classifier stacks the prediction probability distributions from all three base models to deliver an authoritative classification score.
            </StepText>
          </StepCard>
        </StepsGrid>
      </Section>

      {/* Section: AI Architectures */}
      <Section id="models">
        <SectionHeader>
          <SectionPill>
            <FiCpu size={14} />
            <span>Neural Network Topology</span>
          </SectionPill>
          <SectionTitle>Tri-Model Deep Learning Engine</SectionTitle>
          <SectionDescription>
            Instead of relying on a single neural network, our system combines three distinct architectural paradigms for robust feature generalization.
          </SectionDescription>
        </SectionHeader>

        <ModelsGrid>
          <ModelCard>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ padding: '8px', background: '#f0f9ff', borderRadius: '12px', color: '#0284c7' }}>
                <FiTrendingUp size={20} />
              </div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>ResNet-101</h3>
            </div>
            <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
              Employs 101 layers of deep residual connections that bypass vanishing gradients, allowing fine-grained pigment boundary extraction.
            </p>
          </ModelCard>

          <ModelCard>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ padding: '8px', background: '#f0f9ff', borderRadius: '12px', color: '#0284c7' }}>
                <FiLayers size={20} />
              </div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>DenseNet-121</h3>
            </div>
            <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
              Connects every layer to every other downstream layer in a feed-forward fashion, maximizing feature reuse across subtle lesion margins.
            </p>
          </ModelCard>

          <ModelCard>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ padding: '8px', background: '#f0f9ff', borderRadius: '12px', color: '#0284c7' }}>
                <FiZap size={20} />
              </div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>EfficientNet-B3</h3>
            </div>
            <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
              Balances network depth, width, and image resolution using compound scaling for exceptional efficiency and accuracy.
            </p>
          </ModelCard>
        </ModelsGrid>

        <EnsembleBannerCard>
          <div>
            <h3 style={{ fontSize: '1.35rem', fontWeight: 700, margin: '0 0 8px 0' }}>
              Stacked Logistic Regression Meta-Classifier
            </h3>
            <p style={{ margin: 0, color: '#64748b', fontSize: '0.95rem', maxWidth: '640px' }}>
              The output logits of all three models are fused through a meta-learner that calculates weighted confidence probabilities, virtually eliminating false positives.
            </p>
          </div>
          <Button asChild variant="primary" size="md">
            <Link to={ctaRoute}>Test the Ensemble Live</Link>
          </Button>
        </EnsembleBannerCard>
      </Section>

      {/* Section: Supported Conditions */}
      <Section id="conditions">
        <SectionHeader>
          <SectionPill>
            <FiHelpCircle size={14} />
            <span>Pathology Coverage</span>
          </SectionPill>
          <SectionTitle>Supported Skin Disease Classes</SectionTitle>
          <SectionDescription>
            Trained on the comprehensive HAM10000 dermatoscopy benchmark across 7 major clinical lesion classifications.
          </SectionDescription>
        </SectionHeader>

        <ConditionsGrid>
          {SUPPORTED_CONDITIONS.map((cond) => (
            <ConditionCard key={cond.code}>
              <ConditionCategoryPill $type={cond.type}>{cond.category}</ConditionCategoryPill>
              <h4 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '4px 0 0 0' }}>{cond.name}</h4>
              <p style={{ fontSize: '0.875rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
                {cond.description}
              </p>
            </ConditionCard>
          ))}
        </ConditionsGrid>
      </Section>

      {/* Section: Safety & Ethics */}
      <Section id="safety">
        <SectionHeader>
          <SectionPill>
            <FiShield size={14} />
            <span>Clinical Responsibility</span>
          </SectionPill>
          <SectionTitle>Ethics & Privacy Principles</SectionTitle>
          <SectionDescription>
            Built from the ground up to respect patient confidentiality and reinforce physician clinical judgment.
          </SectionDescription>
        </SectionHeader>

        <SafetyCard>
          <SafetyItem>
            <div style={{ color: '#0284c7', marginBottom: '8px' }}>
              <FiShield size={24} />
            </div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>Physician-in-the-Loop</h4>
            <p style={{ fontSize: '0.9rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
              This platform is an assistive triage tool and does not substitute professional medical diagnosis or biopsy confirmation.
            </p>
          </SafetyItem>

          <SafetyItem>
            <div style={{ color: '#0284c7', marginBottom: '8px' }}>
              <FiLock size={24} />
            </div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>Zero-Retention Privacy</h4>
            <p style={{ fontSize: '0.9rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
              Uploaded images are processed in-memory for inference and are not retained on our servers without user consent.
            </p>
          </SafetyItem>

          <SafetyItem>
            <div style={{ color: '#0284c7', marginBottom: '8px' }}>
              <FiCheckCircle size={24} />
            </div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0 }}>Transparent Confidence</h4>
            <p style={{ fontSize: '0.9rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
              Every diagnosis includes raw probability distributions across all candidate conditions for full clinical explainability.
            </p>
          </SafetyItem>
        </SafetyCard>
      </Section>

      {/* Bottom CTA Banner */}
      <CtaSection>
        <CtaCard>
          <h2 style={{ fontSize: '2.5rem', fontWeight: 800, margin: '0 0 16px 0', letterSpacing: '-0.02em' }}>
            Ready to experience next-generation skin AI?
          </h2>
          <p style={{ fontSize: '1.1rem', opacity: 0.8, maxWidth: '580px', margin: '0 0 32px 0', lineHeight: 1.6 }}>
            Upload any dermoscopic image and receive instantaneous multi-model diagnostic confidence scores.
          </p>
          <Button asChild variant="accent" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
        </CtaCard>
      </CtaSection>

      {/* Minimalist Footer */}
      <FooterWrapper>
        <FooterContainer>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#111827' }}>
              Skin AI Predictor
            </div>
            <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '4px' }}>
              Advanced Deep Learning Ensemble for Dermatological Classification.
            </div>
          </div>

          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            © {new Date().getFullYear()} Skin AI Research. Assistive AI Tool.
          </div>
        </FooterContainer>
      </FooterWrapper>
    </LandingPageWrapper>
  );
};

export default Landing;
