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
  FiDatabase,
  FiCheckCircle,
  FiSliders,
  FiInfo,
  FiExternalLink,
} from 'react-icons/fi';
import {
  LandingPageWrapper,
  HeroSection,
  HeroGlow,
  HeroBadge,
  HeroTitle,
  HeroSubtitle,
  HeroCtaRow,
  PipelineSvgContainer,
  SectionWrapper,
  Container,
  SectionHeader,
  SectionTag,
  SectionTitle,
  SectionDescription,
  StatsGrid,
  StatCard,
  StatValue,
  StatLabel,
  DatasetProcessGrid,
  ProcessCard,
  ProcessIcon,
  ModelsGrid,
  ModelCard,
  ModelHeader,
  ModelBadge,
  ModelSvgWrapper,
  EnsembleBanner,
  ConditionsGrid,
  ConditionCard,
  ConditionPill,
  DisclaimerCard,
  DisclaimerItem,
  CtaCard,
  FooterWrapper,
  FooterContainer,
} from './styles';

// 7 Supported Skin Diseases Data
const CONDITIONS_LIST = [
  {
    code: 'mel',
    name: 'Melanoma',
    type: 'danger',
    tag: 'Malignant',
    description: 'A serious type of skin cancer originating in pigment-producing melanocytes. Early detection is crucial.',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Common moles or birthmarks formed by clusters of melanocyte cells. Typically non-cancerous.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    type: 'danger',
    tag: 'Malignant',
    description: 'The most common form of skin cancer. Usually slow-growing and treatable when identified early.',
  },
  {
    code: 'akiec',
    name: 'Actinic Keratoses',
    type: 'warning',
    tag: 'Pre-Cancerous',
    description: 'Rough, dry, scaly patches on skin caused by long-term UV sun exposure. Can progress if untreated.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Non-cancerous skin growths like seborrheic keratosis, commonly developing with age.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Small, firm, non-cancerous fibrous nodules commonly found on the arms and legs.',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Benign blood vessel spots including cherry angiomas and vascular malformations.',
  },
];

const Landing = ({ isAuthenticated }) => {
  const ctaRoute = ROUTES.DASHBOARD;
  const ctaText = 'Try Image Detection';

  return (
    <LandingPageWrapper id="overview">
      {/* Top Navigation */}
      <LandingNavbar isAuthenticated={isAuthenticated} />

      {/* Hero Section */}
      <HeroSection>
        <HeroGlow />

        <HeroBadge>
          <FiCpu size={14} />
          <span>Deep Learning Architecture • HAM10000 Dataset</span>
        </HeroBadge>

        <HeroTitle>
          Skin Disease Detection via <span className="highlight">Ensemble Deep Learning</span>
        </HeroTitle>

        <HeroSubtitle>
          An academic engineering project trained on 10,015 dermatoscopic images. We combine ResNet-101,
          DenseNet-121, and EfficientNet-B3 with stacked meta-learning to classify 7 skin conditions.
        </HeroSubtitle>

        <HeroCtaRow>
          <Button asChild variant="brand" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
          <Button asChild variant="secondary" size="lg">
            <a href="#pipeline">View System Architecture</a>
          </Button>
        </HeroCtaRow>

        {/* Clean Outlined Sequential Pipeline Diagram (Blueprint Style) */}
        <PipelineSvgContainer id="pipeline">
          <svg viewBox="0 0 1020 320" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Step Track Line */}
            <path d="M 130 160 L 900 160" stroke="#2a2a2a" strokeWidth="1.5" strokeDasharray="4 4" />

            {/* Connecting Directed Arrows */}
            <path d="M 170 160 L 220 160" stroke="#16a34a" strokeWidth="1.5" markerEnd="url(#arrow)" />
            <path d="M 370 160 L 420 160" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 420 160 L 460 90" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 420 160 L 460 160" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 420 160 L 460 230" stroke="#16a34a" strokeWidth="1.5" />

            <path d="M 640 90 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 640 160 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 640 230 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
            <path d="M 820 160 L 860 160" stroke="#16a34a" strokeWidth="1.5" />

            {/* STAGE 01: IMAGE INPUT */}
            <g transform="translate(30, 90)">
              <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
              <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">01</text>
              
              {/* Outlined Camera Icon */}
              <circle cx="70" cy="65" r="22" stroke="#16a34a" strokeWidth="1.5" strokeDasharray="3 3" />
              <circle cx="70" cy="65" r="10" stroke="#4ade80" strokeWidth="1.5" />
              
              <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Dermoscopic Image</text>
              <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">Raw RGB Photo</text>
            </g>

            {/* STAGE 02: PREPROCESSING */}
            <g transform="translate(230, 90)">
              <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
              <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">02</text>

              {/* Outlined Matrix Icon */}
              <rect x="50" y="45" width="40" height="40" rx="6" stroke="#4ade80" strokeWidth="1.5" />
              <path d="M 50 65 L 90 65" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 2" />
              <path d="M 70 45 L 70 85" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 2" />

              <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">CLAHE & Scaling</text>
              <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">224 × 224 Normalization</text>
            </g>

            {/* STAGE 03: TRI-MODEL ENSEMBLE BACKBONE */}
            {/* Model A: ResNet-101 */}
            <g transform="translate(460, 55)">
              <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">ResNet-101</text>
              <text x="16" y="44" fill="#888888" fontSize="10">Residual Skip Connections</text>
              <rect x="16" y="50" width="70" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
              <text x="51" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">44.5M PARAMS</text>
            </g>

            {/* Model B: DenseNet-121 */}
            <g transform="translate(460, 130)">
              <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">DenseNet-121</text>
              <text x="16" y="44" fill="#888888" fontSize="10">Dense Feature Concatenation</text>
              <rect x="16" y="50" width="65" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
              <text x="48" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">8.0M PARAMS</text>
            </g>

            {/* Model C: EfficientNet-B3 */}
            <g transform="translate(460, 205)">
              <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">EfficientNet-B3</text>
              <text x="16" y="44" fill="#888888" fontSize="10">Compound Scaling Depth/Width</text>
              <rect x="16" y="50" width="70" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
              <text x="51" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">12.2M PARAMS</text>
            </g>

            {/* STAGE 04: META-CLASSIFIER STACKING */}
            <g transform="translate(680, 90)">
              <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
              <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
              <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">04</text>

              {/* Stacking Node Icon */}
              <circle cx="70" cy="65" r="18" stroke="#4ade80" strokeWidth="1.5" />
              <text x="70" y="70" textAnchor="middle" fill="#4ade80" fontSize="14" fontWeight="800">∑</text>

              <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Meta-Classifier</text>
              <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">Softmax Fusion</text>
            </g>

            {/* STAGE 05: DIAGNOSIS OUTPUT */}
            <g transform="translate(860, 90)">
              <rect width="130" height="140" rx="16" fill="rgba(34,197,94,0.06)" stroke="#16a34a" strokeWidth="1.5" />
              <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.15)" stroke="rgba(34,197,94,0.4)" strokeWidth="1" />
              <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">05</text>

              <circle cx="65" cy="65" r="16" fill="rgba(34,197,94,0.2)" stroke="#16a34a" strokeWidth="1.5" />
              <text x="65" y="70" textAnchor="middle" fill="#4ade80" fontSize="13">✓</text>

              <text x="65" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Classification</text>
              <text x="65" y="124" textAnchor="middle" fill="#4ade80" fontSize="10" fontWeight="600">7 Classes + Prob %</text>
            </g>
          </svg>
        </PipelineSvgContainer>
      </HeroSection>

      {/* Section 2: Kaggle Dataset & Data Preprocessing */}
      <SectionWrapper id="dataset" $alt>
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiDatabase size={14} />
              <span>Dataset & Ingestion</span>
            </SectionTag>
            <SectionTitle>Trained on Kaggle's HAM10000 Dataset</SectionTitle>
            <SectionDescription>
              HAM10000 ("Human Against Machine with 10,000 training images") is an established academic benchmark
              dataset collected across multiple dermatological clinics for skin lesion evaluation.
            </SectionDescription>
          </SectionHeader>

          {/* Key Metrics */}
          <StatsGrid>
            <StatCard>
              <StatValue>10,015</StatValue>
              <StatLabel>Dermoscopic Images</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>7</StatValue>
              <StatLabel>Skin Disease Classes</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>224 × 224</StatValue>
              <StatLabel>Input Matrix Resolution</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>CLAHE</StatValue>
              <StatLabel>Contrast Equalization</StatLabel>
            </StatCard>
          </StatsGrid>

          {/* 3-Step Data Preparation Pipeline */}
          <DatasetProcessGrid>
            <ProcessCard>
              <ProcessIcon>
                <FiDatabase />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                1. Data Cleaning & Splitting
              </h3>
              <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Images are partitioned into train, validation, and test subsets with patient-level isolation
                to ensure reliable real-world evaluation.
              </p>
            </ProcessCard>

            <ProcessCard>
              <ProcessIcon>
                <FiSliders />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                2. Data Augmentation
              </h3>
              <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Random rotations, horizontal and vertical flips, and zoom crops address class imbalance
                among rarer malignant conditions like Dermatofibroma and Vascular lesions.
              </p>
            </ProcessCard>

            <ProcessCard>
              <ProcessIcon>
                <FiCheckCircle />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                3. RGB Normalization
              </h3>
              <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Pixel intensities are mapped to standard [0, 1] tensor distributions and standardized
                across color channels to maximize backpropagation stability.
              </p>
            </ProcessCard>
          </DatasetProcessGrid>
        </Container>
      </SectionWrapper>

      {/* Section 3: Ensemble AI Strategy & Model Architectures */}
      <SectionWrapper id="models">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiLayers size={14} />
              <span>Model Architectures</span>
            </SectionTag>
            <SectionTitle>Tri-Model Ensemble Backbone</SectionTitle>
            <SectionDescription>
              Combining three distinct neural network topologies allows the ensemble to capture complementary
              spatial features while minimizing individual model blind spots.
            </SectionDescription>
          </SectionHeader>

          <ModelsGrid>
            {/* Model 1: ResNet-101 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 2px 0' }}>ResNet-101</h3>
                  <span style={{ fontSize: '0.825rem', color: '#888888' }}>Residual Deep CNN</span>
                </div>
                <ModelBadge>44.5M Params</ModelBadge>
              </ModelHeader>

              {/* Clean Outlined ResNet Diagram */}
              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* Conv Layer 1 */}
                  <rect x="25" y="25" width="60" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="55" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Conv Layer</text>

                  {/* Flow Arrow */}
                  <path d="M 85 40 L 125 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Conv Layer 2 */}
                  <rect x="125" y="25" width="60" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="155" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Conv Layer</text>

                  {/* Flow Arrow */}
                  <path d="M 185 40 L 225 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Addition Node */}
                  <circle cx="240" cy="40" r="14" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="240" y="45" textAnchor="middle" fontSize="14" fontWeight="700" fill="#4ade80">+</text>

                  {/* Output Flow */}
                  <path d="M 254 40 L 295 40" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="295" y="44" fontSize="10" fontWeight="700" fill="#4ade80">F(x)+x</text>

                  {/* Residual Skip Connection */}
                  <path d="M 55 25 C 55 8, 240 8, 240 26" stroke="#4ade80" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <text x="145" y="14" textAnchor="middle" fontSize="8" fontWeight="600" fill="#4ade80">Residual Skip Highway</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#888888', fontSize: '0.9rem', lineHeight: 1.6, margin: 0 }}>
                Identity shortcuts allow gradient signals to travel directly across 101 layers without attenuation,
                capturing fine-grained lesion borders and subtle pigment networks.
              </p>
            </ModelCard>

            {/* Model 2: DenseNet-121 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 2px 0' }}>DenseNet-121</h3>
                  <span style={{ fontSize: '0.825rem', color: '#888888' }}>Dense Feature Reuse</span>
                </div>
                <ModelBadge>8.0M Params</ModelBadge>
              </ModelHeader>

              {/* Clean Outlined DenseNet Diagram */}
              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* Layer 1 */}
                  <rect x="25" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="52" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 1</text>

                  {/* Layer 2 */}
                  <rect x="130" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="157" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 2</text>

                  {/* Layer 3 */}
                  <rect x="235" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="262" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 3</text>

                  {/* Interconnections */}
                  <path d="M 80 40 L 130 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 185 40 L 235 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Dense Curved Connectors */}
                  <path d="M 52 25 C 52 8, 262 8, 262 25" stroke="#4ade80" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <text x="157" y="14" textAnchor="middle" fontSize="8" fontWeight="600" fill="#4ade80">Dense Cross-Layer Concatenation</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#888888', fontSize: '0.9rem', lineHeight: 1.6, margin: 0 }}>
                Every layer receives direct inputs from all preceding layers. Encourages extensive feature reuse
                and provides strong gradient flow with a compact parameter footprint.
              </p>
            </ModelCard>

            {/* Model 3: EfficientNet-B3 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 2px 0' }}>EfficientNet-B3</h3>
                  <span style={{ fontSize: '0.825rem', color: '#888888' }}>Compound Scaling CNN</span>
                </div>
                <ModelBadge>12.2M Params</ModelBadge>
              </ModelHeader>

              {/* Clean Outlined EfficientNet Diagram */}
              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* Depth */}
                  <rect x="35" y="25" width="45" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="57" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Depth (d)</text>

                  <path d="M 80 40 L 120 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Width */}
                  <rect x="120" y="18" width="60" height="44" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="150" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Width (w)</text>

                  <path d="M 180 40 L 220 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Resolution */}
                  <rect x="220" y="12" width="70" height="56" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="255" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Res (r)</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#888888', fontSize: '0.9rem', lineHeight: 1.6, margin: 0 }}>
                Scales network depth, channel width, and resolution simultaneously using a compound coefficient,
                achieving high diagnostic accuracy without excessive computational overhead.
              </p>
            </ModelCard>

            {/* Model 4: Stacking Meta-Classifier */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 2px 0' }}>Meta-Classifier</h3>
                  <span style={{ fontSize: '0.825rem', color: '#888888' }}>Logistic Stacking Layer</span>
                </div>
                <ModelBadge>Probability Fusion</ModelBadge>
              </ModelHeader>

              {/* Clean Outlined Meta-Classifier Diagram */}
              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* Inputs */}
                  <rect x="20" y="10" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="22" textAnchor="middle" fontSize="8" fill="currentColor">P(ResNet)</text>

                  <rect x="20" y="31" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="43" textAnchor="middle" fontSize="8" fill="currentColor">P(DenseNet)</text>

                  <rect x="20" y="52" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="64" textAnchor="middle" fontSize="8" fill="currentColor">P(EfficientNet)</text>

                  {/* Convergence Lines */}
                  <path d="M 90 19 L 140 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 90 40 L 140 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 90 61 L 140 40" stroke="#16a34a" strokeWidth="1.5" />

                  {/* Stacking Node */}
                  <rect x="140" y="20" width="90" height="40" rx="8" fill="rgba(34,197,94,0.1)" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="185" y="38" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4ade80">Meta-Learner</text>
                  <text x="185" y="50" textAnchor="middle" fontSize="8" fill="#888888">Weighted Voting</text>

                  {/* Output */}
                  <path d="M 230 40 L 260 40" stroke="#16a34a" strokeWidth="1.5" />
                  <rect x="260" y="25" width="50" height="30" rx="6" fill="rgba(34,197,94,0.15)" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="285" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4ade80">Class %</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#888888', fontSize: '0.9rem', lineHeight: 1.6, margin: 0 }}>
                Aggregates output prediction logits from all three neural networks using a second-stage meta-classifier,
                averaging out individual model variance for calibrated confidence rankings.
              </p>
            </ModelCard>
          </ModelsGrid>

          {/* Balanced Decision Stacking Banner */}
          <EnsembleBanner>
            <div>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                Parallel Inference & Stacking Execution
              </h3>
              <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                When an image is submitted, the Flask backend executes inference across all 3 architectures
                simultaneously and calculates the weighted softmax consensus.
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <Button asChild variant="brand" size="md">
                <Link to={ctaRoute}>
                  Launch Detection Tool
                  <FiArrowRight size={14} />
                </Link>
              </Button>
            </div>
          </EnsembleBanner>
        </Container>
      </SectionWrapper>

      {/* Section 4: Supported Skin Conditions (7 Classes) */}
      <SectionWrapper id="conditions" $alt>
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiCheckCircle size={14} />
              <span>Diagnostic Classes</span>
            </SectionTag>
            <SectionTitle>7 Supported Skin Conditions</SectionTitle>
            <SectionDescription>
              The model classifies dermatoscopic photos into 7 distinct categories present in the HAM10000 dataset.
            </SectionDescription>
          </SectionHeader>

          <ConditionsGrid>
            {CONDITIONS_LIST.map((cond) => (
              <ConditionCard key={cond.code}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <ConditionPill $type={cond.type}>{cond.tag}</ConditionPill>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#888888' }}>
                    {cond.code.toUpperCase()}
                  </span>
                </div>

                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: '4px 0 0 0' }}>
                  {cond.name}
                </h3>

                <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.55, margin: 0 }}>
                  {cond.description}
                </p>
              </ConditionCard>
            ))}
          </ConditionsGrid>
        </Container>
      </SectionWrapper>

      {/* Section 5: Project Scope & Limitations */}
      <SectionWrapper id="disclaimer">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiShield size={14} />
              <span>Project Transparency</span>
            </SectionTag>
            <SectionTitle>Project Scope & AI Limitations</SectionTitle>
            <SectionDescription>
              Technical considerations regarding how this deep learning tool was engineered and how results should be evaluated.
            </SectionDescription>
          </SectionHeader>

          <DisclaimerCard>
            <DisclaimerItem>
              <div style={{ color: '#16a34a', marginBottom: '4px' }}>
                <FiInfo size={22} />
              </div>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                Academic Project Scope
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                This tool is an engineering project developed to evaluate ensemble deep learning on dermatoscopic images.
                It is not a commercial medical diagnostic system.
              </p>
            </DisclaimerItem>

            <DisclaimerItem>
              <div style={{ color: '#f59e0b', marginBottom: '4px' }}>
                <FiShield size={22} />
              </div>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                AI Model Limitations
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                Deep learning models can make errors, especially on blurry photos, non-standard lighting,
                or skin lesions outside the HAM10000 dataset distribution.
              </p>
            </DisclaimerItem>

            <DisclaimerItem>
              <div style={{ color: '#16a34a', marginBottom: '4px' }}>
                <FiCheckCircle size={22} />
              </div>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                Consult Qualified Doctors
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                Always consult a certified dermatologist for actual clinical evaluation, dermoscopy,
                or biopsy confirmation of any concerning skin spot.
              </p>
            </DisclaimerItem>
          </DisclaimerCard>
        </Container>
      </SectionWrapper>

      {/* Section 6: Bottom Call to Action */}
      <SectionWrapper>
        <Container>
          <CtaCard>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 800, margin: '0 0 10px 0', letterSpacing: '-0.02em' }}>
              Ready to test the Ensemble Model?
            </h2>
            <p style={{ fontSize: '1.05rem', color: '#a3a3a3', maxWidth: '600px', margin: '0 0 24px 0', lineHeight: 1.6 }}>
              Upload any skin lesion photo to view the predicted condition and probability breakdown across all 7 categories.
            </p>
            <Button asChild variant="brand" size="lg">
              <Link to={ctaRoute}>
                {ctaText}
                <FiArrowRight size={16} />
              </Link>
            </Button>
          </CtaCard>
        </Container>
      </SectionWrapper>

      {/* Minimalist Clean Footer */}
      <FooterWrapper>
        <FooterContainer>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1rem', color: 'currentColor' }}>
              Skin Disease Classification Project
            </div>
            <div style={{ fontSize: '0.825rem', color: '#888888', marginTop: '4px' }}>
              Deep Learning Ensemble (ResNet-101 + DenseNet-121 + EfficientNet-B3) on HAM10000.
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.825rem', color: '#888888' }}>
            <span>Educational ML Project</span>
            <span>•</span>
            <a
              href="https://github.com/Purvesh-PJ/skin_disease_detection"
              target="_blank"
              rel="noopener noreferrer"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#16a34a', fontWeight: 600, textDecoration: 'none' }}
            >
              GitHub Repository
              <FiExternalLink size={12} />
            </a>
          </div>
        </FooterContainer>
      </FooterWrapper>
    </LandingPageWrapper>
  );
};

export default Landing;
