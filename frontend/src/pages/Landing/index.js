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
  HeroSplitLayout,
  HeroContent,
  HeroVisual,
  VisualCard,
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
  DatasetLayout,
  DatasetInfographicCard,
  DatasetMetricsGrid,
  MetricBadge,
  ClassDistributionBar,
  DataStagesContainer,
  DataStageCard,
  StageBadge,
  StageContent,
  RiskSpectrumBanner,
  RiskColumnsGrid,
  RiskTierColumn,
  RiskTierHeader,
  DiseaseCompactCard,
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

        <HeroSplitLayout>
          {/* Left Column: Hero Content */}
          <HeroContent>
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
          </HeroContent>

          {/* Right Column: Disease + AI Dermoscopy Scanner Visual SVG */}
          <HeroVisual>
            <VisualCard>
              <svg viewBox="0 0 460 380" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <linearGradient id="aiPulseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#4ade80" />
                    <stop offset="100%" stopColor="#16a34a" />
                  </linearGradient>

                  <linearGradient id="lesionMelanomaGrad" x1="20%" y1="20%" x2="80%" y2="80%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.9" />
                    <stop offset="45%" stopColor="#991b1b" />
                    <stop offset="100%" stopColor="#450a0a" />
                  </linearGradient>

                  <radialGradient id="reticleRadar" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity="0.22" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                  </radialGradient>
                </defs>

                {/* HUD Tech Corner Brackets */}
                <path d="M 18 36 L 18 18 L 36 18" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />
                <path d="M 424 18 L 442 18 L 442 36" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />
                <path d="M 18 335 L 18 353 L 36 353" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />
                <path d="M 424 353 L 442 353 L 442 335" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />

                {/* Header HUD Status Bar */}
                <circle cx="34" cy="30" r="4" fill="#22c55e" />
                <circle cx="34" cy="30" r="7" stroke="#22c55e" strokeWidth="1" opacity="0.4" />
                <text x="48" y="34" fill="currentColor" fontSize="10" fontWeight="700" letterSpacing="0.06em">
                  DERMOSCOPIC AI SCANNER
                </text>
                <rect x="336" y="20" width="88" height="20" rx="4" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.35)" strokeWidth="1" />
                <text x="380" y="33" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">
                  HAM10000 AI
                </text>

                {/* Background Tech Grid Lines */}
                <line x1="40" y1="85" x2="420" y2="85" stroke="rgba(34,197,94,0.08)" strokeDasharray="3 3" />
                <line x1="40" y1="145" x2="420" y2="145" stroke="rgba(34,197,94,0.08)" strokeDasharray="3 3" />
                <line x1="40" y1="205" x2="420" y2="205" stroke="rgba(34,197,94,0.08)" strokeDasharray="3 3" />
                <line x1="40" y1="265" x2="420" y2="265" stroke="rgba(34,197,94,0.08)" strokeDasharray="3 3" />

                {/* Main Dermatoscope Inspection Reticle Circle (Centered at (230, 175)) */}
                <circle cx="230" cy="175" r="92" stroke="#16a34a" strokeWidth="1.5" strokeDasharray="4 6" opacity="0.75" />
                <circle cx="230" cy="175" r="78" stroke="rgba(34,197,94,0.2)" strokeWidth="1" />
                <circle cx="230" cy="175" r="92" fill="url(#reticleRadar)" />

                {/* Reticle Crosshairs with Degree Marks */}
                <line x1="125" y1="175" x2="335" y2="175" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />
                <line x1="230" y1="70" x2="230" y2="280" stroke="#16a34a" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />
                <text x="230" y="78" textAnchor="middle" fill="#4ade80" fontSize="7" fontWeight="700">0°</text>
                <text x="328" y="178" textAnchor="middle" fill="#4ade80" fontSize="7" fontWeight="700">90°</text>
                <text x="230" y="274" textAnchor="middle" fill="#4ade80" fontSize="7" fontWeight="700">180°</text>
                <text x="132" y="178" textAnchor="middle" fill="#4ade80" fontSize="7" fontWeight="700">270°</text>

                {/* Organic Skin Lesion (Disease Feature) */}
                <path
                  d="M 212 144 C 236 130, 258 140, 265 158 C 273 178, 256 204, 238 210 C 216 216, 194 202, 198 178 C 200 158, 198 152, 212 144 Z"
                  fill="url(#lesionMelanomaGrad)"
                  stroke="#991b1b"
                  strokeWidth="1.5"
                />
                {/* Granular Pigment Micro-structures */}
                <circle cx="225" cy="164" r="2" fill="#450a0a" />
                <circle cx="244" cy="172" r="2.5" fill="#450a0a" />
                <circle cx="232" cy="187" r="2" fill="#450a0a" />
                <circle cx="216" cy="180" r="2" fill="#7f1d1d" />

                {/* AI Boundary Segmentation / Computer Vision Tracking Polygon */}
                <polygon
                  points="206,138 238,128 264,138 272,160 266,198 238,216 208,212 190,190 192,160"
                  fill="rgba(34,197,94,0.06)"
                  stroke="#22c55e"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
                {/* Active Tracking Vertices */}
                <rect x="204" y="136" width="4" height="4" fill="#4ade80" />
                <rect x="236" y="126" width="4" height="4" fill="#4ade80" />
                <rect x="262" y="136" width="4" height="4" fill="#4ade80" />
                <rect x="270" y="158" width="4" height="4" fill="#4ade80" />
                <rect x="264" y="196" width="4" height="4" fill="#4ade80" />
                <rect x="236" y="214" width="4" height="4" fill="#4ade80" />
                <rect x="206" y="210" width="4" height="4" fill="#4ade80" />
                <rect x="188" y="188" width="4" height="4" fill="#4ade80" />

                {/* Multi-CNN Neural Nodes Overlay */}
                {/* ResNet-101 Node */}
                <g transform="translate(28, 75)">
                  <rect width="98" height="32" rx="7" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="10" y="16" fill="currentColor" fontSize="10" fontWeight="700">ResNet-101</text>
                  <text x="10" y="26" fill="#4ade80" fontSize="8" fontWeight="600">Skip Highway</text>
                </g>
                <path d="M 126 91 C 156 91, 185 134, 206 138" stroke="#16a34a" strokeWidth="1.2" strokeDasharray="3 3" />

                {/* DenseNet-121 Node */}
                <g transform="translate(28, 245)">
                  <rect width="102" height="32" rx="7" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="10" y="16" fill="currentColor" fontSize="10" fontWeight="700">DenseNet-121</text>
                  <text x="10" y="26" fill="#4ade80" fontSize="8" fontWeight="600">Feature Reuse</text>
                </g>
                <path d="M 130 261 C 158 261, 185 216, 206 210" stroke="#16a34a" strokeWidth="1.2" strokeDasharray="3 3" />

                {/* EfficientNet-B3 Node */}
                <g transform="translate(325, 75)">
                  <rect width="105" height="32" rx="7" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="10" y="16" fill="currentColor" fontSize="10" fontWeight="700">EfficientNet-B3</text>
                  <text x="10" y="26" fill="#4ade80" fontSize="8" fontWeight="600">Compound Scale</text>
                </g>
                <path d="M 325 91 C 295 91, 280 134, 264 138" stroke="#16a34a" strokeWidth="1.2" strokeDasharray="3 3" />

                {/* Meta-Classifier Stacking Node */}
                <g transform="translate(310, 235)">
                  <rect width="120" height="42" rx="9" fill="rgba(34,197,94,0.15)" stroke="#22c55e" strokeWidth="1.5" />
                  <text x="10" y="17" fill="#4ade80" fontSize="10" fontWeight="800">Logistic Stacking</text>
                  <text x="10" y="30" fill="currentColor" fontSize="9" fontWeight="600">7-Class Softmax</text>
                  <circle cx="104" cy="21" r="7" fill="rgba(34,197,94,0.25)" />
                  <text x="104" y="25" textAnchor="middle" fill="#4ade80" fontSize="10" fontWeight="800">∑</text>
                </g>
                <path d="M 266 198 C 282 210, 292 235, 310 255" stroke="#22c55e" strokeWidth="1.5" strokeDasharray="3 3" />

                {/* Bottom Diagnostic HUD Status Strip */}
                <g transform="translate(26, 312)">
                  <rect width="408" height="28" rx="7" fill="rgba(255,255,255,0.03)" stroke="rgba(34,197,94,0.25)" strokeWidth="1" />
                  <text x="12" y="18" fill="#888888" fontSize="9">Diagnostic Scope:</text>
                  <text x="106" y="18" fill="#4ade80" fontSize="9" fontWeight="700">7 Disease Classes • Stacking Consensus</text>
                  <text x="396" y="18" textAnchor="end" fill="#22c55e" fontSize="9" fontWeight="700">✓ READY</text>
                </g>
              </svg>
            </VisualCard>
          </HeroVisual>
        </HeroSplitLayout>

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
            {/* Model 1: ResNet-101 (Left Content | Right SVG) */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>ResNet-101</h3>
                  <ModelBadge>44.5M Params • Structural Specialist</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#4ade80' }}>
                  Identity Skip Highways for Lesion Borders & Asymmetry
                </span>
                <p style={{ color: '#a3a3a3', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  <strong>How it works:</strong> Standard deep networks suffer from signal loss as layers get deeper.
                  ResNet-101 solves this with <em>shortcut connections</em> (<code>F(x) + x</code>) that let gradients
                  travel directly across 101 layers without vanishing.
                </p>
                <p style={{ color: '#888888', fontSize: '0.875rem', lineHeight: 1.6, margin: 0 }}>
                  <strong>Clinical Role:</strong> Melanoma and malignant tumors often have jagged, asymmetrical borders.
                  ResNet’s ultra-deep receptive field is specifically tuned to detect these macro-structural contour anomalies.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {/* Skip Highway Arc */}
                  <path d="M 42 40 C 42 12, 277 12, 277 43" stroke="#4ade80" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <rect x="110" y="4" width="100" height="16" rx="8" fill="rgba(34,197,94,0.12)" stroke="rgba(34,197,94,0.3)" strokeWidth="1" />
                  <text x="160" y="15" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">Residual Skip Highway</text>

                  {/* Nodes & Conv Layers */}
                  <rect x="15" y="40" width="55" height="30" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="42" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Input X</text>

                  <path d="M 70 55 L 95 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="95" y="40" width="60" height="30" rx="6" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="125" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Conv 1</text>

                  <path d="M 155 55 L 180 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="180" y="40" width="60" height="30" rx="6" fill="rgba(255,255,255,0.02)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="210" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Conv 2</text>

                  <path d="M 240 55 L 265 55" stroke="#16a34a" strokeWidth="1.2" />

                  <circle cx="277" cy="55" r="12" fill="rgba(34,197,94,0.15)" stroke="#22c55e" strokeWidth="1.5" />
                  <text x="277" y="60" textAnchor="middle" fill="#4ade80" fontSize="13" fontWeight="800">+</text>

                  <path d="M 289 55 L 310 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="310" y="40" width="40" height="30" rx="6" fill="rgba(34,197,94,0.12)" stroke="#22c55e" strokeWidth="1.2" />
                  <text x="330" y="58" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="800">F(x)+x</text>

                  <text x="180" y="98" textAnchor="middle" fill="#888888" fontSize="8.5">
                    101 layers • Zero degradation signal propagation
                  </text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 2: DenseNet-121 (Left SVG | Right Content) */}
            <ModelSpecRow $reverse>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>DenseNet-121</h3>
                  <ModelBadge>8.0M Params • Texture Specialist</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#4ade80' }}>
                  Cross-Layer Concatenation for Micro-Pigment Patterns
                </span>
                <p style={{ color: '#a3a3a3', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  <strong>How it works:</strong> Instead of summing outputs, every layer in DenseNet directly connects to
                  <em>all subsequent layers</em> in the network. This maximizes gradient reuse and ensures no subtle details
                  get lost in transit.
                </p>
                <p style={{ color: '#888888', fontSize: '0.875rem', lineHeight: 1.6, margin: 0 }}>
                  <strong>Clinical Role:</strong> Benign keratoses and subtle vascular spots are differentiated by tiny cellular
                  pigment granules. DenseNet excels at recognizing these fine dermatoscopic texture patterns.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {/* Dense Links */}
                  <path d="M 47 40 C 47 14, 247 14, 247 40" stroke="#4ade80" strokeWidth="1.2" strokeDasharray="3 3" fill="none" />
                  <path d="M 47 70 C 47 92, 330 92, 330 72" stroke="#16a34a" strokeWidth="1.2" strokeDasharray="3 3" fill="none" />
                  <path d="M 147 40 C 147 22, 330 22, 330 38" stroke="#22c55e" strokeWidth="1.2" strokeDasharray="2 2" fill="none" />
                  <text x="147" y="11" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">Dense Cross-Layer Feature Reuse</text>

                  {/* Layers */}
                  <rect x="20" y="40" width="55" height="30" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="47" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Layer 1</text>

                  <path d="M 75 55 L 120 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="120" y="40" width="55" height="30" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="147" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Layer 2</text>

                  <path d="M 175 55 L 220 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="220" y="40" width="55" height="30" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="247" y="58" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700">Layer 3</text>

                  <path d="M 275 55 L 310 55" stroke="#16a34a" strokeWidth="1.2" />

                  <rect x="310" y="38" width="40" height="34" rx="6" fill="rgba(34,197,94,0.15)" stroke="#22c55e" strokeWidth="1.2" />
                  <text x="330" y="53" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="800">[x0..x2]</text>
                  <text x="330" y="64" textAnchor="middle" fill="#888888" fontSize="7">Concat</text>

                  <text x="180" y="104" textAnchor="middle" fill="#888888" fontSize="8.5">
                    High parameter efficiency with direct cross-layer gradient flow
                  </text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 3: EfficientNet-B3 (Left Content | Right SVG) */}
            <ModelSpecRow>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>EfficientNet-B3</h3>
                  <ModelBadge>12.2M Params • Scale Specialist</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#4ade80' }}>
                  Compound Scaling across Depth, Width & Image Resolution
                </span>
                <p style={{ color: '#a3a3a3', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  <strong>How it works:</strong> Rather than arbitrarily making the network deeper or wider, EfficientNet scales
                  <em>Depth (d)</em>, <em>Width (w)</em>, and <em>Resolution (r)</em> simultaneously using a mathematically principled
                  compound scaling coefficient.
                </p>
                <p style={{ color: '#888888', fontSize: '0.875rem', lineHeight: 1.6, margin: 0 }}>
                  <strong>Clinical Role:</strong> Dermatological images vary widely in zoom and lesion size. EfficientNet balances
                  both macro lesion overview and micro zoom details with supreme computational speed.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <text x="180" y="10" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">
                    Compound Coefficient Principle: α · β² · γ² ≈ 2
                  </text>

                  {/* Depth */}
                  <rect x="25" y="30" width="48" height="42" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="49" y="49" textAnchor="middle" fill="currentColor" fontSize="8.5" fontWeight="700">Depth</text>
                  <text x="49" y="61" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">d = αᵠ</text>

                  <path d="M 73 51 L 110 51" stroke="#16a34a" strokeWidth="1.2" />

                  {/* Width */}
                  <rect x="110" y="24" width="60" height="54" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="140" y="48" textAnchor="middle" fill="currentColor" fontSize="8.5" fontWeight="700">Width</text>
                  <text x="140" y="60" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">w = βᵠ</text>

                  <path d="M 170 51 L 205 51" stroke="#16a34a" strokeWidth="1.2" />

                  {/* Resolution */}
                  <rect x="205" y="18" width="70" height="66" rx="6" fill="rgba(34,197,94,0.08)" stroke="#16a34a" strokeWidth="1.2" />
                  <text x="240" y="48" textAnchor="middle" fill="currentColor" fontSize="8.5" fontWeight="700">Resolution</text>
                  <text x="240" y="60" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">r = γᵠ (224px)</text>

                  <path d="M 275 51 L 305 51" stroke="#16a34a" strokeWidth="1.2" />

                  {/* Optimal Output */}
                  <rect x="305" y="30" width="45" height="42" rx="6" fill="rgba(34,197,94,0.15)" stroke="#22c55e" strokeWidth="1.2" />
                  <text x="327" y="49" textAnchor="middle" fill="#4ade80" fontSize="8" fontWeight="800">Optimal</text>
                  <text x="327" y="60" textAnchor="middle" fill="currentColor" fontSize="7.5" fontWeight="600">FLOPs</text>

                  <text x="180" y="104" textAnchor="middle" fill="#888888" fontSize="8.5">
                    Unified multi-dimension scaling yields maximum feature efficiency
                  </text>
                </svg>
              </ModelSvgPanel>
            </ModelSpecRow>

            {/* Model 4: Meta-Classifier (Left SVG | Right Content) */}
            <ModelSpecRow $reverse>
              <ModelInfo>
                <ModelTitleRow>
                  <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: 0 }}>Logistic Meta-Learner</h3>
                  <ModelBadge>Stacking Arbiter • Softmax Consensus</ModelBadge>
                </ModelTitleRow>
                <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#4ade80' }}>
                  Learned Probability Weighting for Calibrated Diagnoses
                </span>
                <p style={{ color: '#a3a3a3', fontSize: '0.925rem', lineHeight: 1.65, margin: 0 }}>
                  <strong>How it works:</strong> Rather than taking a simple unweighted average, a second-stage Logistic Regression
                  meta-model learns the individual reliability and accuracy weights ($w_1, w_2, w_3$) of each CNN on validation data.
                </p>
                <p style={{ color: '#888888', fontSize: '0.875rem', lineHeight: 1.6, margin: 0 }}>
                  <strong>Clinical Role:</strong> In healthcare, confidence calibration is essential. The meta-learner ensures that
                  when a high risk prediction like Melanoma is flagged, it is backed by calibrated multi-model consensus.
                </p>
              </ModelInfo>

              <ModelSvgPanel>
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <text x="190" y="14" textAnchor="middle" fill="#4ade80" fontSize="7.5" fontWeight="700">
                    Supervised 2nd-Stage Logistic Consensus
                  </text>

                  {/* 3 CNN Inputs */}
                  <rect x="15" y="16" width="76" height="20" rx="4" fill="rgba(255,255,255,0.02)" stroke="#525252" strokeWidth="1" />
                  <text x="53" y="29" textAnchor="middle" fill="currentColor" fontSize="7.5">P(ResNet-101)</text>

                  <rect x="15" y="45" width="76" height="20" rx="4" fill="rgba(255,255,255,0.02)" stroke="#525252" strokeWidth="1" />
                  <text x="53" y="58" textAnchor="middle" fill="currentColor" fontSize="7.5">P(DenseNet-121)</text>

                  <rect x="15" y="74" width="76" height="20" rx="4" fill="rgba(255,255,255,0.02)" stroke="#525252" strokeWidth="1" />
                  <text x="53" y="87" textAnchor="middle" fill="currentColor" fontSize="7.5">P(EfficientNet)</text>

                  {/* Weighted Synapse Arcs */}
                  <path d="M 91 26 L 140 55" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 91 55 L 140 55" stroke="#16a34a" strokeWidth="1.5" />
                  <path d="M 91 84 L 140 55" stroke="#16a34a" strokeWidth="1.5" />
                  <text x="110" y="34" fill="#4ade80" fontSize="7" fontWeight="700">w₁</text>
                  <text x="115" y="50" fill="#4ade80" fontSize="7" fontWeight="700">w₂</text>
                  <text x="110" y="78" fill="#4ade80" fontSize="7" fontWeight="700">w₃</text>

                  {/* Meta-Learner Hub */}
                  <rect x="140" y="30" width="105" height="50" rx="10" fill="rgba(34,197,94,0.12)" stroke="#22c55e" strokeWidth="1.5" />
                  <text x="192" y="52" textAnchor="middle" fill="#4ade80" fontSize="9.5" fontWeight="800">Meta-Learner</text>
                  <text x="192" y="65" textAnchor="middle" fill="#888888" fontSize="7.5">Learned Softmax Fusion</text>

                  {/* Output */}
                  <path d="M 245 55 L 280 55" stroke="#22c55e" strokeWidth="1.5" />

                  <rect x="280" y="35" width="65" height="40" rx="8" fill="rgba(34,197,94,0.18)" stroke="#22c55e" strokeWidth="1.2" />
                  <text x="312" y="52" textAnchor="middle" fill="#4ade80" fontSize="9" fontWeight="800">7 Classes</text>
                  <text x="312" y="64" textAnchor="middle" fill="currentColor" fontSize="7.5" fontWeight="600">Calibrated %</text>

                  <text x="180" y="104" textAnchor="middle" fill="#888888" fontSize="8.5">
                    Weighs individual model reliability per class for verified consensus
                  </text>
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
              <span>Training Corpus & Preprocessing</span>
            </SectionTag>
            <SectionTitle>Trained on Kaggle's HAM10000 Dataset</SectionTitle>
            <SectionDescription>
              A benchmark corpus of 10,015 dermatoscopic images collected across international dermatological clinics,
              curated for dermatological deep learning research.
            </SectionDescription>
          </SectionHeader>

          <DatasetLayout>
            {/* Left: Dataset Infographic & Class Distribution */}
            <DatasetInfographicCard>
              <DatasetMetricsGrid>
                <MetricBadge>
                  <strong>10,015</strong>
                  <span>Verified Images</span>
                </MetricBadge>
                <MetricBadge>
                  <strong>7 Classes</strong>
                  <span>Skin Conditions</span>
                </MetricBadge>
                <MetricBadge>
                  <strong>224 × 224</strong>
                  <span>RGB Tensor Size</span>
                </MetricBadge>
                <MetricBadge>
                  <strong>CLAHE</strong>
                  <span>Contrast Tuned</span>
                </MetricBadge>
              </DatasetMetricsGrid>

              {/* Graphical Stacked Distribution */}
              <ClassDistributionBar>
                <div className="bar-header">
                  <span>Class Distribution & Imbalance</span>
                  <span>10,015 Total Scans</span>
                </div>
                <div className="stacked-bar">
                  <div style={{ width: '67%', background: '#22c55e' }} title="NV: 6,705 (67%)" />
                  <div style={{ width: '11%', background: '#ef4444' }} title="MEL: 1,113 (11%)" />
                  <div style={{ width: '11%', background: '#10b981' }} title="BKL: 1,099 (11%)" />
                  <div style={{ width: '5%', background: '#dc2626' }} title="BCC: 514 (5%)" />
                  <div style={{ width: '3%', background: '#f59e0b' }} title="AKIEC: 327 (3%)" />
                  <div style={{ width: '1.5%', background: '#3b82f6' }} title="VASC: 142 (1.5%)" />
                  <div style={{ width: '1.5%', background: '#8b5cf6' }} title="DF: 115 (1.5%)" />
                </div>
                <div className="bar-legend">
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#22c55e' }} />
                    <span>NV: 67% (Nevi)</span>
                  </div>
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#ef4444' }} />
                    <span>MEL: 11% (Melanoma)</span>
                  </div>
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#10b981' }} />
                    <span>BKL: 11% (Keratosis)</span>
                  </div>
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#dc2626' }} />
                    <span>BCC: 5% (Carcinoma)</span>
                  </div>
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#f59e0b' }} />
                    <span>AKIEC: 3% (Actinic)</span>
                  </div>
                  <div className="legend-item">
                    <span className="dot" style={{ background: '#3b82f6' }} />
                    <span>VASC/DF: 3%</span>
                  </div>
                </div>
              </ClassDistributionBar>

              <p style={{ fontSize: '0.8rem', color: '#888888', lineHeight: 1.55, margin: 0 }}>
                💡 <em>Imbalance Mitigation:</em> To prevent model bias toward dominant benign moles (NV),
                we apply dynamic class weighting and multi-axis data augmentation during backpropagation.
              </p>
            </DatasetInfographicCard>

            {/* Right: 3-Stage Data Engineering Flow */}
            <DataStagesContainer>
              <DataStageCard>
                <StageBadge>STAGE 01</StageBadge>
                <StageContent>
                  <h4>Patient-Isolated Splitting (70/15/15)</h4>
                  <p>
                    Images are partitioned by unique patient IDs. Multiple lesions from the same patient never overlap
                    between train and test sets, strictly preventing artificial accuracy inflation.
                  </p>
                </StageContent>
              </DataStageCard>

              <DataStageCard>
                <StageBadge>STAGE 02</StageBadge>
                <StageContent>
                  <h4>Dynamic Augmentation & Rebalancing</h4>
                  <p>
                    Rotations (0°–360°), horizontal/vertical flips, and zoom crops are dynamically generated on-the-fly,
                    multiplying the effective variety of rare malignant classes like Melanoma and BCC.
                  </p>
                </StageContent>
              </DataStageCard>

              <DataStageCard>
                <StageBadge>STAGE 03</StageBadge>
                <StageContent>
                  <h4>CLAHE Contrast & RGB Tensor Normalization</h4>
                  <p>
                    Contrast-Limited Adaptive Histogram Equalization sharpens subtle lesion boundaries, followed by
                    channel-wise RGB scaling to [0, 1] for smooth, stable gradient descent.
                  </p>
                </StageContent>
              </DataStageCard>
            </DataStagesContainer>
          </DatasetLayout>
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
              The ensemble model classifies skin lesions across 3 distinct clinical risk tiers,
              spanning high-priority malignant cancers to harmless benign growths.
            </SectionDescription>
          </SectionHeader>

          {/* Visual Risk Spectrum Meter */}
          <RiskSpectrumBanner>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ color: '#16a34a' }}>●</span>
              <span>Clinical Risk Spectrum:</span>
            </div>
            <div className="spectrum-bar">
              <span style={{ color: '#22c55e' }}>🟢 Low Risk (Benign)</span>
              <span>→</span>
              <span style={{ color: '#f59e0b' }}>🟡 Medium Risk (Pre-Cancer)</span>
              <span>→</span>
              <span style={{ color: '#ef4444' }}>🔴 High Priority (Malignant)</span>
            </div>
          </RiskSpectrumBanner>

          {/* 3 Risk Columns */}
          <RiskColumnsGrid>
            {/* Column 1: Malignant (High Risk) */}
            <RiskTierColumn>
              <RiskTierHeader $color="#ef4444">
                <h3>
                  <span>🔴</span> Malignant Cancers
                </h3>
                <span className="count">2 Diseases</span>
              </RiskTierHeader>

              {MALIGNANT_CONDITIONS.map((cond) => (
                <DiseaseCompactCard key={cond.code} $color="#ef4444">
                  <div className="card-top">
                    <h4>{cond.name}</h4>
                    <span className="code-badge">{cond.code.toUpperCase()}</span>
                  </div>
                  <p>{cond.description}</p>
                </DiseaseCompactCard>
              ))}
            </RiskTierColumn>

            {/* Column 2: Pre-Cancerous (Medium Risk) */}
            <RiskTierColumn>
              <RiskTierHeader $color="#f59e0b">
                <h3>
                  <span>🟡</span> Pre-Cancerous
                </h3>
                <span className="count">1 Disease</span>
              </RiskTierHeader>

              {PRECANCER_CONDITIONS.map((cond) => (
                <DiseaseCompactCard key={cond.code} $color="#f59e0b">
                  <div className="card-top">
                    <h4>{cond.name}</h4>
                    <span className="code-badge">{cond.code.toUpperCase()}</span>
                  </div>
                  <p>{cond.description}</p>
                </DiseaseCompactCard>
              ))}
            </RiskTierColumn>

            {/* Column 3: Benign (Low Risk) */}
            <RiskTierColumn>
              <RiskTierHeader $color="#22c55e">
                <h3>
                  <span>🟢</span> Benign Conditions
                </h3>
                <span className="count">4 Diseases</span>
              </RiskTierHeader>

              {BENIGN_CONDITIONS.map((cond) => (
                <DiseaseCompactCard key={cond.code} $color="#22c55e">
                  <div className="card-top">
                    <h4>{cond.name}</h4>
                    <span className="code-badge">{cond.code.toUpperCase()}</span>
                  </div>
                  <p>{cond.description}</p>
                </DiseaseCompactCard>
              ))}
            </RiskTierColumn>
          </RiskColumnsGrid>
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
