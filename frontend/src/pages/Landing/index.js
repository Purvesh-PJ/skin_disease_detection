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
  FiInfo,
  FiActivity,
  FiExternalLink,
  FiZap,
} from 'react-icons/fi';
import {
  LandingPageWrapper,
  HeroSection,
  HeroGlow,
  HeroBadge,
  HeroTitle,
  HeroSubtitle,
  HeroCtaRow,
  HeroPillarsGrid,
  HeroPillar,
  PillarIcon,
  PillarText,
  PipelineSvgContainer,
  SectionWrapper,
  Container,
  SectionHeader,
  SectionTag,
  SectionTitle,
  SectionDescription,
  ProcessFlow,
  ProcessStep,
  StepNumber,
  StepTitle,
  StepDesc,
  ModelSpecsContainer,
  ModelSpecRow,
  ModelInfo,
  ModelTitleRow,
  ModelBadge,
  ModelSvgPanel,
  StatsStrip,
  StatItem,
  StatValue,
  StatLabel,
  ConditionCategory,
  ConditionsMatrix,
  ConditionRow,
  ConditionPill,
  NoticeStrip,
  NoticeItem,
  CtaCard,
  FooterWrapper,
  FooterContainer,
} from './styles';

// 7 Supported Skin Diseases Grouped by Clinical Risk
const MALIGNANT_CONDITIONS = [
  {
    code: 'mel',
    name: 'Melanoma',
    type: 'danger',
    tag: 'Malignant (High Risk)',
    description: 'A serious form of skin cancer originating in pigment-producing melanocytes. Early identification is vital for successful treatment.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    type: 'danger',
    tag: 'Malignant (High Risk)',
    description: 'The most common form of skin cancer. Arises in basal cells and typically grows slowly without spreading to distant sites.',
  },
];

const PRECANCER_CONDITIONS = [
  {
    code: 'akiec',
    name: 'Actinic Keratoses',
    type: 'warning',
    tag: 'Pre-Cancerous (Medium Risk)',
    description: 'Rough, dry, scaly patches on sun-exposed skin caused by UV damage. Can occasionally progress to squamous cell carcinoma if left untreated.',
  },
];

const BENIGN_CONDITIONS = [
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Common moles or birthmarks formed by clusters of melanocytes. Common in adults and typically non-cancerous.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Non-cancerous skin growths including seborrheic keratoses and solar lentigines that commonly develop with aging.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Small, firm, non-cancerous fibrous skin nodules, most commonly found on the arms and lower legs.',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Benign blood vessel spots including cherry angiomas, angiokeratomas, and vascular malformations.',
  },
];

const Landing = ({ isAuthenticated }) => {
  const ctaRoute = ROUTES.DASHBOARD;
  const ctaText = 'Try Image Detection';

  return (
    <LandingPageWrapper id="overview">
      {/* Top Navigation */}
      <LandingNavbar isAuthenticated={isAuthenticated} />

      {/* SECTION 1: HERO & PROJECT INTRODUCTION */}
      <HeroSection>
        <HeroGlow />

        <HeroBadge>
          <FiCpu size={14} />
          <span>Dermatological AI Research • Ensemble Deep Learning</span>
        </HeroBadge>

        <HeroTitle>
          Skin Disease Classification via <span className="highlight">Ensemble Deep Learning</span>
        </HeroTitle>

        <HeroSubtitle>
          An academic machine learning project designed to assist in early skin disease screening.
          Trained on 10,015 dermatoscopic images from Kaggle's HAM10000 benchmark, our system combines three
          distinct deep CNN topologies with a logistic stacking layer to detect and categorize 7 skin conditions.
        </HeroSubtitle>

        <HeroCtaRow>
          <Button asChild variant="brand" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
          <Button asChild variant="secondary" size="lg">
            <a href="#how-it-works">How It Works</a>
          </Button>
        </HeroCtaRow>

        {/* 4 Core Pillars of the Project */}
        <HeroPillarsGrid>
          <HeroPillar>
            <PillarIcon>
              <FiLayers />
            </PillarIcon>
            <PillarText>
              <strong>Stacked Ensemble</strong>
              <span>ResNet + DenseNet + EfficientNet</span>
            </PillarText>
          </HeroPillar>

          <HeroPillar>
            <PillarIcon>
              <FiDatabase />
            </PillarIcon>
            <PillarText>
              <strong>HAM10000 Dataset</strong>
              <span>10,015 Verified Images</span>
            </PillarText>
          </HeroPillar>

          <HeroPillar>
            <PillarIcon>
              <FiZap />
            </PillarIcon>
            <PillarText>
              <strong>Real-Time Screening</strong>
              <span>Sub-Second Inference API</span>
            </PillarText>
          </HeroPillar>

          <HeroPillar>
            <PillarIcon>
              <FiActivity />
            </PillarIcon>
            <PillarText>
              <strong>7 Disease Classes</strong>
              <span>Malignant & Benign Analysis</span>
            </PillarText>
          </HeroPillar>
        </HeroPillarsGrid>
      </HeroSection>

      {/* SECTION 2: HOW IT WORKS (END-TO-END DATA FLOW) */}
      <SectionWrapper id="how-it-works" $alt>
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiActivity size={14} />
              <span>Step-by-Step Workflow</span>
            </SectionTag>
            <SectionTitle>How the Detection Pipeline Works</SectionTitle>
            <SectionDescription>
              From raw dermatoscopic image ingestion to calibrated probability estimation,
              here is the sequential process executed by the backend on every scan.
            </SectionDescription>
          </SectionHeader>

          {/* 4-Step Process Walkthrough */}
          <ProcessFlow>
            <ProcessStep>
              <StepNumber>STEP 01</StepNumber>
              <StepTitle>Image Ingestion</StepTitle>
              <StepDesc>
                The user uploads a close-up dermatoscopic image of a suspicious skin spot or mole via the web dashboard.
              </StepDesc>
            </ProcessStep>

            <ProcessStep>
              <StepNumber>STEP 02</StepNumber>
              <StepTitle>CLAHE Preprocessing</StepTitle>
              <StepDesc>
                OpenCV resizes the image to 224×224 pixels and applies contrast-limited adaptive histogram equalization to enhance pigment patterns.
              </StepDesc>
            </ProcessStep>

            <ProcessStep>
              <StepNumber>STEP 03</StepNumber>
              <StepTitle>Tri-Model Inference</StepTitle>
              <StepDesc>
                ResNet-101, DenseNet-121, and EfficientNet-B3 simultaneously extract spatial features and generate probability logits.
              </StepDesc>
            </ProcessStep>

            <ProcessStep>
              <StepNumber>STEP 04</StepNumber>
              <StepTitle>Stacked Prediction</StepTitle>
              <StepDesc>
                A logistic meta-classifier aggregates all 3 outputs, calculating the final disease classification and calibrated confidence score.
              </StepDesc>
            </ProcessStep>
          </ProcessFlow>

          {/* Clean Outlined Pipeline Blueprint Diagram */}
          <PipelineSvgContainer>
            <svg viewBox="0 0 1020 320" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M 130 160 L 900 160" stroke="#2a2a2a" strokeWidth="1.5" strokeDasharray="4 4" />

              <path d="M 170 160 L 220 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 370 160 L 420 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 420 160 L 460 90" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 420 160 L 460 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 420 160 L 460 230" stroke="#16a34a" strokeWidth="1.5" />

              <path d="M 640 90 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 640 160 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 640 230 L 680 160" stroke="#16a34a" strokeWidth="1.5" />
              <path d="M 820 160 L 860 160" stroke="#16a34a" strokeWidth="1.5" />

              {/* Stage 01 */}
              <g transform="translate(30, 90)">
                <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
                <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">01</text>
                <circle cx="70" cy="65" r="22" stroke="#16a34a" strokeWidth="1.5" strokeDasharray="3 3" />
                <circle cx="70" cy="65" r="10" stroke="#4ade80" strokeWidth="1.5" />
                <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Dermoscopy Photo</text>
                <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">Raw RGB Image</text>
              </g>

              {/* Stage 02 */}
              <g transform="translate(230, 90)">
                <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
                <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">02</text>
                <rect x="50" y="45" width="40" height="40" rx="6" stroke="#4ade80" strokeWidth="1.5" />
                <path d="M 50 65 L 90 65" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 2" />
                <path d="M 70 45 L 70 85" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 2" />
                <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">CLAHE & Scaling</text>
                <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">224 × 224 Normalization</text>
              </g>

              {/* Stage 03 */}
              <g transform="translate(460, 55)">
                <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">ResNet-101</text>
                <text x="16" y="44" fill="#888888" fontSize="10">Residual Skip Highway</text>
                <rect x="16" y="50" width="70" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
                <text x="51" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">44.5M PARAMS</text>
              </g>

              <g transform="translate(460, 130)">
                <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">DenseNet-121</text>
                <text x="16" y="44" fill="#888888" fontSize="10">Dense Feature Reuse</text>
                <rect x="16" y="50" width="65" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
                <text x="48" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">8.0M PARAMS</text>
              </g>

              <g transform="translate(460, 205)">
                <rect width="180" height="68" rx="12" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <text x="16" y="28" fill="currentColor" fontSize="13" fontWeight="700">EfficientNet-B3</text>
                <text x="16" y="44" fill="#888888" fontSize="10">Compound Scaling</text>
                <rect x="16" y="50" width="70" height="12" rx="6" fill="rgba(34,197,94,0.1)" />
                <text x="51" y="59" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="700">12.2M PARAMS</text>
              </g>

              {/* Stage 04 */}
              <g transform="translate(680, 90)">
                <rect width="140" height="140" rx="16" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.5" />
                <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
                <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">04</text>
                <circle cx="70" cy="65" r="18" stroke="#4ade80" strokeWidth="1.5" />
                <text x="70" y="70" textAnchor="middle" fill="#4ade80" fontSize="14" fontWeight="800">∑</text>
                <text x="70" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Meta-Classifier</text>
                <text x="70" y="124" textAnchor="middle" fill="#888888" fontSize="10">Softmax Fusion</text>
              </g>

              {/* Output */}
              <g transform="translate(860, 90)">
                <rect width="130" height="140" rx="16" fill="rgba(34,197,94,0.06)" stroke="#16a34a" strokeWidth="1.5" />
                <rect x="12" y="12" width="40" height="18" rx="9" fill="rgba(34,197,94,0.15)" stroke="rgba(34,197,94,0.4)" strokeWidth="1" />
                <text x="32" y="24" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">05</text>
                <circle cx="65" cy="65" r="16" fill="rgba(34,197,94,0.2)" stroke="#16a34a" strokeWidth="1.5" />
                <text x="65" y="70" textAnchor="middle" fill="#4ade80" fontSize="13">✓</text>
                <text x="65" y="108" textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="700">Diagnosis</text>
                <text x="65" y="124" textAnchor="middle" fill="#4ade80" fontSize="10" fontWeight="600">7 Classes + Prob %</text>
              </g>
            </svg>
          </PipelineSvgContainer>
        </Container>
      </SectionWrapper>

      {/* SECTION 3: DEEP ENSEMBLE AI ARCHITECTURES */}
      <SectionWrapper id="architecture">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiLayers size={14} />
              <span>Multi-Model Topologies</span>
            </SectionTag>
            <SectionTitle>Why Ensemble Stacking?</SectionTitle>
            <SectionDescription>
              A single convolutional neural network often exhibits inductive bias and can struggle with diverse skin phototypes.
              By stacking three architecturally distinct CNN backbones, our model achieves superior generalizability and consensus calibration.
            </SectionDescription>
          </SectionHeader>

          <ModelSpecsContainer>
            {/* Model 1: ResNet-101 */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>ResNet-101</h3>
                  <ModelBadge>44.5M Parameters</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.85rem', color: '#888888' }}>Deep Residual Network with Skip Connections</span>
                <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  Uses identity shortcuts (<code>F(x) + x</code>) to pass gradients directly across 101 layers,
                  preventing vanishing gradient degradation and capturing complex structural lesion borders.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 320 80" fill="none">
                  <rect x="25" y="25" width="60" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="55" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Conv Layer</text>

                  <path d="M 85 40 L 125 40" stroke="#16a34a" strokeWidth="1.5" />

                  <rect x="125" y="25" width="60" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="155" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Conv Layer</text>

                  <path d="M 185 40 L 225 40" stroke="#16a34a" strokeWidth="1.5" />

                  <circle cx="240" cy="40" r="14" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="240" y="45" textAnchor="middle" fontSize="14" fontWeight="700" fill="#4ade80">+</text>

                  <path d="M 254 40 L 295 40" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="295" y="44" fontSize="10" fontWeight="700" fill="#4ade80">F(x)+x</text>

                  <path d="M 55 25 C 55 8, 240 8, 240 26" stroke="#4ade80" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <text x="145" y="14" textAnchor="middle" fontSize="8" fontWeight="600" fill="#4ade80">Residual Skip Highway</text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 2: DenseNet-121 */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>DenseNet-121</h3>
                  <ModelBadge>8.0M Parameters</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.85rem', color: '#888888' }}>Densely Connected Convolutional Network</span>
                <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  Connects every layer to all subsequent layers in a feed-forward fashion, encouraging maximum
                  feature reuse and high-gradient flow with a very compact parameter footprint.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 320 80" fill="none">
                  <rect x="25" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="52" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 1</text>

                  <rect x="130" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="157" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 2</text>

                  <rect x="235" y="25" width="55" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="262" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">Layer 3</text>

                  <path d="M 80 40 L 130 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 185 40 L 235 40" stroke="#16a34a" strokeWidth="1.5" />

                  <path d="M 52 25 C 52 8, 262 8, 262 25" stroke="#4ade80" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <text x="157" y="14" textAnchor="middle" fontSize="8" fontWeight="600" fill="#4ade80">Dense Cross-Layer Concatenation</text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 3: EfficientNet-B3 */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>EfficientNet-B3</h3>
                  <ModelBadge>12.2M Parameters</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.85rem', color: '#888888' }}>Compound Scaling Convolutional Architecture</span>
                <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  Balances network depth (<code>d</code>), width (<code>w</code>), and input resolution (<code>r</code>)
                  simultaneously using a fixed compound coefficient for optimal feature efficiency.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 320 80" fill="none">
                  <rect x="35" y="25" width="45" height="30" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="57" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Depth (d)</text>

                  <path d="M 80 40 L 120 40" stroke="#16a34a" strokeWidth="1.5" />

                  <rect x="120" y="18" width="60" height="44" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="150" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Width (w)</text>

                  <path d="M 180 40 L 220 40" stroke="#16a34a" strokeWidth="1.5" />

                  <rect x="220" y="12" width="70" height="56" rx="6" fill="none" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="255" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="currentColor">Res (r)</text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 4: Meta-Classifier */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>Logistic Meta-Learner</h3>
                  <ModelBadge>Stacking Layer</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.85rem', color: '#888888' }}>Second-Stage Softmax Consensus Model</span>
                <p style={{ color: '#888888', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  Instead of a simple majority vote, the meta-classifier learns the individual reliability weights
                  of each model on validation data, yielding calibrated final probabilities.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 320 80" fill="none">
                  <rect x="20" y="10" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="22" textAnchor="middle" fontSize="8" fill="currentColor">P(ResNet)</text>

                  <rect x="20" y="31" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="43" textAnchor="middle" fontSize="8" fill="currentColor">P(DenseNet)</text>

                  <rect x="20" y="52" width="70" height="18" rx="4" fill="none" stroke="#525252" strokeWidth="1" />
                  <text x="55" y="64" textAnchor="middle" fontSize="8" fill="currentColor">P(EfficientNet)</text>

                  <path d="M 90 19 L 140 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 90 40 L 140 40" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 90 61 L 140 40" stroke="#16a34a" strokeWidth="1.5" />

                  <rect x="140" y="20" width="90" height="40" rx="8" fill="rgba(34,197,94,0.1)" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="185" y="38" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4ade80">Meta-Learner</text>
                  <text x="185" y="50" textAnchor="middle" fontSize="8" fill="#888888">Weighted Voting</text>

                  <path d="M 230 40 L 260 40" stroke="#16a34a" strokeWidth="1.5" />
                  <rect x="260" y="25" width="50" height="30" rx="6" fill="rgba(34,197,94,0.15)" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="285" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4ade80">Class %</text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>
          </ModelSpecsContainer>
        </Container>
      </SectionWrapper>

      {/* SECTION 4: DATASET & TRAINING METHODOLOGY */}
      <SectionWrapper id="dataset" $alt>
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiDatabase size={14} />
              <span>Training Corpus</span>
            </SectionTag>
            <SectionTitle>Trained on Kaggle's HAM10000 Dataset</SectionTitle>
            <SectionDescription>
              HAM10000 ("Human Against Machine with 10,000 training images") is an academic benchmark
              dataset collected across multiple international dermatological clinics.
            </SectionDescription>
          </SectionHeader>

          {/* Borderless Metric Strip */}
          <StatsStrip>
            <StatItem>
              <StatValue>10,015</StatValue>
              <StatLabel>Dermatoscopic Images</StatLabel>
            </StatItem>
            <StatItem>
              <StatValue>7</StatValue>
              <StatLabel>Disease Categories</StatLabel>
            </StatItem>
            <StatItem>
              <StatValue>224 × 224</StatValue>
              <StatLabel>Input Matrix Resolution</StatLabel>
            </StatItem>
            <StatItem>
              <StatValue>CLAHE</StatValue>
              <StatLabel>Contrast Equalization</StatLabel>
            </StatItem>
          </StatsStrip>

          {/* 3-Step Data Engineering Pipeline */}
          <ProcessFlow>
            <ProcessStep>
              <StepNumber>STAGE 01</StepNumber>
              <StepTitle>Patient-Isolated Splitting</StepTitle>
              <StepDesc>
                Images are partitioned into train, validation, and test subsets with patient-level isolation
                to ensure the model generalizes across unseen patients rather than memorizing lesions.
              </StepDesc>
            </ProcessStep>

            <ProcessStep>
              <StepNumber>STAGE 02</StepNumber>
              <StepTitle>Augmentation & Balancing</StepTitle>
              <StepDesc>
                Rotations, horizontal/vertical flips, and zoom scaling are applied dynamically alongside class weights
                to prevent majority bias toward common moles (NV).
              </StepDesc>
            </ProcessStep>

            <ProcessStep>
              <StepNumber>STAGE 03</StepNumber>
              <StepTitle>RGB Tensor Normalization</StepTitle>
              <StepDesc>
                Pixel intensities are scaled to standard [0, 1] distributions and normalized across RGB color channels
                for stable neural network backpropagation.
              </StepDesc>
            </ProcessStep>
          </ProcessFlow>
        </Container>
      </SectionWrapper>

      {/* SECTION 5: 7 SUPPORTED SKIN CONDITIONS GROUPED BY RISK */}
      <SectionWrapper id="conditions">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiCheckCircle size={14} />
              <span>Diagnostic Scope</span>
            </SectionTag>
            <SectionTitle>7 Supported Skin Disease Conditions</SectionTitle>
            <SectionDescription>
              The ensemble model is trained to classify dermatoscopic lesions into 7 distinct categories
              spanning malignant cancers, pre-cancerous growths, and benign lesions.
            </SectionDescription>
          </SectionHeader>

          {/* Group 1: Malignant (High Risk) */}
          <ConditionCategory>
            <h3>
              <span style={{ color: '#ef4444' }}>●</span>
              Malignant Skin Cancers (High Priority)
            </h3>
            <ConditionsMatrix>
              {MALIGNANT_CONDITIONS.map((cond) => (
                <ConditionRow key={cond.code}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>{cond.name}</h4>
                      <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#888888' }}>
                        ({cond.code.toUpperCase()})
                      </span>
                    </div>
                    <ConditionPill $type={cond.type}>{cond.tag}</ConditionPill>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#888888', lineHeight: 1.55, margin: 0 }}>
                    {cond.description}
                  </p>
                </ConditionRow>
              ))}
            </ConditionsMatrix>
          </ConditionCategory>

          {/* Group 2: Pre-Cancerous (Medium Risk) */}
          <ConditionCategory>
            <h3>
              <span style={{ color: '#f59e0b' }}>●</span>
              Pre-Cancerous Lesions
            </h3>
            <ConditionsMatrix>
              {PRECANCER_CONDITIONS.map((cond) => (
                <ConditionRow key={cond.code}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>{cond.name}</h4>
                      <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#888888' }}>
                        ({cond.code.toUpperCase()})
                      </span>
                    </div>
                    <ConditionPill $type={cond.type}>{cond.tag}</ConditionPill>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#888888', lineHeight: 1.55, margin: 0 }}>
                    {cond.description}
                  </p>
                </ConditionRow>
              ))}
            </ConditionsMatrix>
          </ConditionCategory>

          {/* Group 3: Benign / Non-Cancerous (Low Risk) */}
          <ConditionCategory>
            <h3>
              <span style={{ color: '#22c55e' }}>●</span>
              Benign Non-Cancerous Conditions
            </h3>
            <ConditionsMatrix>
              {BENIGN_CONDITIONS.map((cond) => (
                <ConditionRow key={cond.code}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>{cond.name}</h4>
                      <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#888888' }}>
                        ({cond.code.toUpperCase()})
                      </span>
                    </div>
                    <ConditionPill $type={cond.type}>{cond.tag}</ConditionPill>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#888888', lineHeight: 1.55, margin: 0 }}>
                    {cond.description}
                  </p>
                </ConditionRow>
              ))}
            </ConditionsMatrix>
          </ConditionCategory>
        </Container>
      </SectionWrapper>

      {/* SECTION 6: PROJECT SCOPE & AI LIMITATIONS */}
      <SectionWrapper id="transparency" $alt>
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiShield size={14} />
              <span>Project Transparency</span>
            </SectionTag>
            <SectionTitle>Project Scope & AI Limitations</SectionTitle>
            <SectionDescription>
              Technical considerations regarding how this deep learning tool was engineered and how its outputs should be interpreted.
            </SectionDescription>
          </SectionHeader>

          <NoticeStrip>
            <NoticeItem>
              <div style={{ color: '#16a34a', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FiInfo size={18} />
                <h4 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'currentColor' }}>
                  Academic Research Project
                </h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                This tool is an engineering project developed to evaluate ensemble deep learning on dermatoscopic images.
                It is not an FDA-approved clinical diagnostic device.
              </p>
            </NoticeItem>

            <NoticeItem>
              <div style={{ color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FiShield size={18} />
                <h4 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'currentColor' }}>
                  Sensitivity to Photo Quality
                </h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                Deep learning models can make errors, especially on blurry photos, non-standard smartphone lighting,
                hair occlusions, or lesions outside the HAM10000 distribution.
              </p>
            </NoticeItem>

            <NoticeItem>
              <div style={{ color: '#16a34a', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FiCheckCircle size={18} />
                <h4 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, color: 'currentColor' }}>
                  Consult Certified Doctors
                </h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: '#888888', lineHeight: 1.6, margin: 0 }}>
                Always consult a certified dermatologist for professional clinical examination, dermoscopy,
                or biopsy confirmation of any concerning skin spot.
              </p>
            </NoticeItem>
          </NoticeStrip>
        </Container>
      </SectionWrapper>

      {/* SECTION 7: LIVE DEMO CTA & CLEAN FOOTER */}
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
