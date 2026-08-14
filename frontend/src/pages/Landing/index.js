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
    description: 'A serious type of skin cancer that begins in melanocytes. Early identification is critical.',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Common moles or birthmarks formed by clusters of pigment cells. Typically harmless.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    type: 'danger',
    tag: 'Malignant',
    description: 'The most common form of skin cancer. Usually slow-growing and treatable when detected early.',
  },
  {
    code: 'akiec',
    name: 'Actinic Keratoses',
    type: 'warning',
    tag: 'Pre-Cancerous',
    description: 'Rough, dry, scaly patches on skin caused by long-term sun exposure. Can progress if untreated.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Non-cancerous skin growths like seborrheic keratosis, commonly appearing with aging.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    type: 'success',
    tag: 'Benign (Harmless)',
    description: 'Small, firm, non-cancerous skin nodules often found on the lower legs.',
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
          <span>Deep Learning Project • Ensemble Multi-Model Architecture</span>
        </HeroBadge>

        <HeroTitle>
          Skin Disease Classification Using <span className="highlight">Ensemble Deep Learning</span>
        </HeroTitle>

        <HeroSubtitle>
          An engineering project trained on Kaggle's HAM10000 dataset. We combine ResNet-101,
          DenseNet-121, and EfficientNet-B3 into a stacked ensemble to detect and classify 7 common skin conditions.
        </HeroSubtitle>

        <HeroCtaRow>
          <Button asChild variant="brand" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
          <Button asChild variant="secondary" size="lg">
            <a href="#pipeline">Explore AI Pipeline</a>
          </Button>
        </HeroCtaRow>

        {/* Large Visual Neural Network Pipeline SVG Diagram */}
        <PipelineSvgContainer id="pipeline">
          <svg viewBox="0 0 1000 360" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Background Grid Accent */}
            <defs>
              <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="100%" stopColor="#1d4ed8" />
              </linearGradient>
              <linearGradient id="cardGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#ffffff" />
                <stop offset="100%" stopColor="#f8fafc" />
              </linearGradient>
              <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
                <feDropShadow dx="0" dy="4" stdDeviation="6" floodOpacity="0.06" />
              </filter>
            </defs>

            {/* Connecting Flow Lines */}
            <path d="M 170 180 L 260 180" stroke="#94a3b8" strokeWidth="2.5" strokeDasharray="5 5" />
            <path d="M 400 180 L 480 90" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 400 180 L 480 180" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 400 180 L 480 270" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 660 90 L 730 180" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 660 180 L 730 180" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 660 270 L 730 180" stroke="#3b82f6" strokeWidth="2.5" />
            <path d="M 870 180 L 910 180" stroke="#10b981" strokeWidth="3" />

            {/* Stage 1: Raw Image Ingest */}
            <g transform="translate(30, 110)">
              <rect width="140" height="140" rx="16" fill="url(#cardGrad)" stroke="#cbd5e1" strokeWidth="1.5" filter="url(#shadow)" />
              <rect x="20" y="20" width="100" height="70" rx="8" fill="#e2e8f0" />
              <circle cx="70" cy="55" r="16" fill="#94a3b8" />
              <text x="70" y="112" textAnchor="middle" fill="#0f172a" fontSize="13" fontWeight="700">Skin Photo</text>
              <text x="70" y="128" textAnchor="middle" fill="#64748b" fontSize="11">Raw Input</text>
            </g>

            {/* Stage 2: Preprocessing */}
            <g transform="translate(260, 110)">
              <rect width="140" height="140" rx="16" fill="url(#cardGrad)" stroke="#cbd5e1" strokeWidth="1.5" filter="url(#shadow)" />
              <rect x="20" y="20" width="100" height="70" rx="8" fill="#eff6ff" stroke="#bfdbfe" />
              <text x="70" y="50" textAnchor="middle" fill="#2563eb" fontSize="12" fontWeight="700">224 x 224</text>
              <text x="70" y="68" textAnchor="middle" fill="#64748b" fontSize="11">CLAHE Resize</text>
              <text x="70" y="112" textAnchor="middle" fill="#0f172a" fontSize="13" fontWeight="700">Preprocessing</text>
              <text x="70" y="128" textAnchor="middle" fill="#64748b" fontSize="11">RGB Normalization</text>
            </g>

            {/* Stage 3: Three Parallel Models */}
            {/* Model A: ResNet-101 */}
            <g transform="translate(480, 45)">
              <rect width="180" height="90" rx="14" fill="#ffffff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#shadow)" />
              <circle cx="30" cy="45" r="14" fill="#eff6ff" />
              <text x="30" y="50" textAnchor="middle" fill="#2563eb" fontSize="11" fontWeight="800">R</text>
              <text x="56" y="38" fill="#0f172a" fontSize="14" fontWeight="700">ResNet-101</text>
              <text x="56" y="56" fill="#64748b" fontSize="11">Residual Skip Layers</text>
              <text x="56" y="72" fill="#2563eb" fontSize="11" fontWeight="600">44.5M Parameters</text>
            </g>

            {/* Model B: DenseNet-121 */}
            <g transform="translate(480, 140)">
              <rect width="180" height="80" rx="14" fill="#ffffff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#shadow)" />
              <circle cx="30" cy="40" r="14" fill="#eff6ff" />
              <text x="30" y="45" textAnchor="middle" fill="#2563eb" fontSize="11" fontWeight="800">D</text>
              <text x="56" y="34" fill="#0f172a" fontSize="14" fontWeight="700">DenseNet-121</text>
              <text x="56" y="52" fill="#64748b" fontSize="11">Dense Feature Reuse</text>
              <text x="56" y="68" fill="#2563eb" fontSize="11" fontWeight="600">8.0M Parameters</text>
            </g>

            {/* Model C: EfficientNet-B3 */}
            <g transform="translate(480, 230)">
              <rect width="180" height="85" rx="14" fill="#ffffff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#shadow)" />
              <circle cx="30" cy="42" r="14" fill="#eff6ff" />
              <text x="30" y="47" textAnchor="middle" fill="#2563eb" fontSize="11" fontWeight="800">E</text>
              <text x="56" y="34" fill="#0f172a" fontSize="14" fontWeight="700">EfficientNet-B3</text>
              <text x="56" y="52" fill="#64748b" fontSize="11">Compound Scaling</text>
              <text x="56" y="68" fill="#2563eb" fontSize="11" fontWeight="600">12.2M Parameters</text>
            </g>

            {/* Stage 4: Ensemble Meta-Classifier */}
            <g transform="translate(730, 110)">
              <rect width="140" height="140" rx="16" fill="url(#blueGrad)" filter="url(#shadow)" />
              <text x="70" y="50" textAnchor="middle" fill="#ffffff" fontSize="13" fontWeight="800">Ensemble</text>
              <text x="70" y="70" textAnchor="middle" fill="#dbeafe" fontSize="11">Stacking Layer</text>
              <rect x="25" y="90" width="90" height="30" rx="6" fill="rgba(255,255,255,0.2)" />
              <text x="70" y="110" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="700">Softmax Fusion</text>
            </g>

            {/* Stage 5: Prediction Output */}
            <g transform="translate(910, 110)">
              <rect width="70" height="140" rx="16" fill="#f0fdf4" stroke="#86efac" strokeWidth="1.5" filter="url(#shadow)" />
              <circle cx="35" cy="45" r="16" fill="#dcfce7" />
              <text x="35" y="50" textAnchor="middle" fill="#16a34a" fontSize="14">✓</text>
              <text x="35" y="90" textAnchor="middle" fill="#166534" fontSize="11" fontWeight="800">Output</text>
              <text x="35" y="110" textAnchor="middle" fill="#15803d" fontSize="10">Class & %</text>
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
              <span>Dataset & Training</span>
            </SectionTag>
            <SectionTitle>Trained on Kaggle's HAM10000 Dataset</SectionTitle>
            <SectionDescription>
              HAM10000 ("Human Against Machine with 10,000 training images") is a benchmark dermatoscopic
              dataset collected from multiple clinical institutions to train dermatological deep learning models.
            </SectionDescription>
          </SectionHeader>

          {/* Dataset Key Metrics */}
          <StatsGrid>
            <StatCard>
              <StatValue>10,015</StatValue>
              <StatLabel>Dermoscopic Images</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>7</StatValue>
              <StatLabel>Disease Classes</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>224 x 224</StatValue>
              <StatLabel>Input Matrix Resolution</StatLabel>
            </StatCard>
            <StatCard>
              <StatValue>CLAHE</StatValue>
              <StatLabel>Contrast Enhancement</StatLabel>
            </StatCard>
          </StatsGrid>

          {/* Preprocessing Steps */}
          <DatasetProcessGrid>
            <ProcessCard>
              <ProcessIcon>
                <FiDatabase />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 6px 0' }}>
                1. Data Cleaning & Splitting
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.95rem', lineHeight: 1.6, margin: 0 }}>
                Images are partitioned into standard training, validation, and test subsets to evaluate
                real-world generalization across diverse skin types.
              </p>
            </ProcessCard>

            <ProcessCard>
              <ProcessIcon>
                <FiSliders />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 6px 0' }}>
                2. Data Augmentation
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.95rem', lineHeight: 1.6, margin: 0 }}>
                Random rotations, horizontal and vertical flips, and zoom scaling are applied during training
                to address class imbalance in rarer skin conditions.
              </p>
            </ProcessCard>

            <ProcessCard>
              <ProcessIcon>
                <FiCheckCircle />
              </ProcessIcon>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 6px 0' }}>
                3. Image Normalization
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.95rem', lineHeight: 1.6, margin: 0 }}>
                Pixels are scaled to standard [0, 1] tensor distributions and standardized across RGB channels
                to accelerate network convergence.
              </p>
            </ProcessCard>
          </DatasetProcessGrid>
        </Container>
      </SectionWrapper>

      {/* Section 3: Ensemble AI Strategy & Architectures */}
      <SectionWrapper id="models">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiLayers size={14} />
              <span>Model Architectures</span>
            </SectionTag>
            <SectionTitle>Why Ensemble Deep Learning?</SectionTitle>
            <SectionDescription>
              A single convolutional neural network can have architectural blind spots. By combining three
              diverse models with different strengths, our ensemble achieves more reliable predictions.
            </SectionDescription>
          </SectionHeader>

          <ModelsGrid>
            {/* Model 1: ResNet-101 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 2px 0' }}>ResNet-101</h3>
                  <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Residual Deep CNN</span>
                </div>
                <ModelBadge>44.5M Params</ModelBadge>
              </ModelHeader>

              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* ResNet Residual Skip Representation */}
                  <rect x="20" y="25" width="60" height="30" rx="6" fill="#dbeafe" stroke="#3b82f6" />
                  <text x="50" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#1d4ed8">Conv Layer</text>

                  <rect x="130" y="25" width="60" height="30" rx="6" fill="#dbeafe" stroke="#3b82f6" />
                  <text x="160" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#1d4ed8">Conv Layer</text>

                  <circle cx="250" cy="40" r="14" fill="#eff6ff" stroke="#2563eb" />
                  <text x="250" y="45" textAnchor="middle" fontSize="14" fontWeight="700" fill="#2563eb">+</text>

                  {/* Main Flow */}
                  <path d="M 80 40 L 130 40" stroke="#2563eb" strokeWidth="2" />
                  <path d="M 190 40 L 236 40" stroke="#2563eb" strokeWidth="2" />
                  <path d="M 264 40 L 300 40" stroke="#2563eb" strokeWidth="2" />

                  {/* Skip Connection Curve */}
                  <path d="M 50 25 C 50 5, 250 5, 250 26" stroke="#f59e0b" strokeWidth="2" strokeDasharray="3 3" fill="none" />
                  <text x="150" y="12" textAnchor="middle" fontSize="9" fontWeight="600" fill="#d97706">Skip Highway (Identity Shortcut)</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#64748b', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Uses identity shortcut connections to allow gradient signals to travel directly across 101 layers,
                capturing intricate lesion boundaries without vanishing gradient issues.
              </p>
            </ModelCard>

            {/* Model 2: DenseNet-121 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 2px 0' }}>DenseNet-121</h3>
                  <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Dense Feature Reuse</span>
                </div>
                <ModelBadge>8.0M Params</ModelBadge>
              </ModelHeader>

              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* DenseNet Interconnection Representation */}
                  <rect x="20" y="25" width="55" height="30" rx="6" fill="#e0e7ff" stroke="#6366f1" />
                  <text x="47" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4338ca">Layer 1</text>

                  <rect x="130" y="25" width="55" height="30" rx="6" fill="#e0e7ff" stroke="#6366f1" />
                  <text x="157" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4338ca">Layer 2</text>

                  <rect x="240" y="25" width="55" height="30" rx="6" fill="#e0e7ff" stroke="#6366f1" />
                  <text x="267" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#4338ca">Layer 3</text>

                  {/* Dense Interconnections */}
                  <path d="M 75 40 L 130 40" stroke="#6366f1" strokeWidth="2" />
                  <path d="M 185 40 L 240 40" stroke="#6366f1" strokeWidth="2" />
                  <path d="M 47 25 C 47 5, 267 5, 267 25" stroke="#ec4899" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />
                  <text x="157" y="12" textAnchor="middle" fontSize="9" fontWeight="600" fill="#db2777">Dense Cross-Layer Concatenation</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#64748b', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Directly connects all layers to each subsequent layer. Reuses low-level edge features
                alongside high-level texture patterns with a compact parameter count.
              </p>
            </ModelCard>

            {/* Model 3: EfficientNet-B3 */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 2px 0' }}>EfficientNet-B3</h3>
                  <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Compound Scaling CNN</span>
                </div>
                <ModelBadge>12.2M Params</ModelBadge>
              </ModelHeader>

              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* EfficientNet Compound Scaling */}
                  <rect x="30" y="30" width="40" height="20" rx="4" fill="#dcfce7" stroke="#22c55e" />
                  <text x="50" y="43" textAnchor="middle" fontSize="8" fontWeight="700" fill="#15803d">Width</text>

                  <rect x="130" y="20" width="50" height="40" rx="4" fill="#dcfce7" stroke="#22c55e" />
                  <text x="155" y="44" textAnchor="middle" fontSize="9" fontWeight="700" fill="#15803d">Depth</text>

                  <rect x="230" y="10" width="60" height="60" rx="6" fill="#dcfce7" stroke="#22c55e" />
                  <text x="260" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#15803d">Resolution</text>

                  <path d="M 70 40 L 130 40" stroke="#16a34a" strokeWidth="2" />
                  <path d="M 180 40 L 230 40" stroke="#16a34a" strokeWidth="2" />
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#64748b', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Scales network depth, width, and resolution simultaneously using a compound coefficient,
                delivering high representational accuracy at minimal computational cost.
              </p>
            </ModelCard>

            {/* Model 4: Stacking Meta-Classifier */}
            <ModelCard>
              <ModelHeader>
                <div>
                  <h3 style={{ fontSize: '1.3rem', fontWeight: 700, margin: '0 0 2px 0' }}>Meta-Classifier</h3>
                  <span style={{ fontSize: '0.85rem', color: '#64748b' }}>Logistic Stacking Layer</span>
                </div>
                <ModelBadge>Probability Fusion</ModelBadge>
              </ModelHeader>

              <ModelSvgWrapper>
                <svg viewBox="0 0 320 80" fill="none">
                  {/* Softmax Stacking */}
                  <rect x="20" y="10" width="70" height="18" rx="4" fill="#f1f5f9" stroke="#94a3b8" />
                  <text x="55" y="22" textAnchor="middle" fontSize="9" fill="#475569">P(ResNet)</text>

                  <rect x="20" y="32" width="70" height="18" rx="4" fill="#f1f5f9" stroke="#94a3b8" />
                  <text x="55" y="44" textAnchor="middle" fontSize="9" fill="#475569">P(DenseNet)</text>

                  <rect x="20" y="54" width="70" height="18" rx="4" fill="#f1f5f9" stroke="#94a3b8" />
                  <text x="55" y="66" textAnchor="middle" fontSize="9" fill="#475569">P(EfficientNet)</text>

                  <path d="M 90 19 L 140 40" stroke="#2563eb" strokeWidth="1.5" />
                  <path d="M 90 41 L 140 40" stroke="#2563eb" strokeWidth="1.5" />
                  <path d="M 90 63 L 140 40" stroke="#2563eb" strokeWidth="1.5" />

                  <rect x="140" y="20" width="90" height="40" rx="8" fill="#2563eb" />
                  <text x="185" y="38" textAnchor="middle" fontSize="10" fontWeight="700" fill="#ffffff">Meta-Learner</text>
                  <text x="185" y="50" textAnchor="middle" fontSize="8" fill="#bfdbfe">Weighted Voting</text>

                  <path d="M 230 40 L 260 40" stroke="#16a34a" strokeWidth="2" />
                  <rect x="260" y="25" width="50" height="30" rx="6" fill="#dcfce7" stroke="#22c55e" />
                  <text x="285" y="44" textAnchor="middle" fontSize="10" fontWeight="700" fill="#15803d">Class %</text>
                </svg>
              </ModelSvgWrapper>

              <p style={{ color: '#64748b', fontSize: '0.925rem', lineHeight: 1.6, margin: 0 }}>
                Combines output logits from all three neural networks using a second-stage meta-classifier,
                averaging out individual model biases for robust consensus.
              </p>
            </ModelCard>
          </ModelsGrid>

          {/* Ensemble Summary Card */}
          <EnsembleBanner>
            <div>
              <h3 style={{ fontSize: '1.4rem', fontWeight: 700, margin: '0 0 6px 0' }}>
                Balanced Decision Stacking
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.95rem', lineHeight: 1.6, margin: 0 }}>
                When an image is evaluated, the backend executes inference across all three architectures
                in parallel threads and applies weighted calibration to produce the final diagnostic ranking.
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
              <span>Diagnostic Scope</span>
            </SectionTag>
            <SectionTitle>7 Supported Skin Conditions</SectionTitle>
            <SectionDescription>
              The model is trained to recognize and differentiate between these 7 specific dermatological
              categories from the HAM10000 dataset.
            </SectionDescription>
          </SectionHeader>

          <ConditionsGrid>
            {CONDITIONS_LIST.map((cond) => (
              <ConditionCard key={cond.code}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <ConditionPill $type={cond.type}>{cond.tag}</ConditionPill>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8' }}>
                    {cond.code.toUpperCase()}
                  </span>
                </div>

                <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '4px 0 0 0' }}>
                  {cond.name}
                </h3>

                <p style={{ fontSize: '0.9rem', color: '#64748b', lineHeight: 1.55, margin: 0 }}>
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
              Important considerations regarding how this deep learning tool was developed and how its
              outputs should be interpreted.
            </SectionDescription>
          </SectionHeader>

          <DisclaimerCard>
            <DisclaimerItem>
              <div style={{ color: '#2563eb', marginBottom: '4px' }}>
                <FiInfo size={22} />
              </div>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, margin: 0 }}>
                Academic Project Scope
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
                This tool is an engineering project created to study ensemble deep learning on dermatoscopic images.
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
              <p style={{ fontSize: '0.875rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
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
              <p style={{ fontSize: '0.875rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
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
            <h2 style={{ fontSize: '2.5rem', fontWeight: 800, margin: '0 0 12px 0', letterSpacing: '-0.02em' }}>
              Ready to test the Ensemble Model?
            </h2>
            <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '600px', margin: '0 0 28px 0', lineHeight: 1.6 }}>
              Upload any skin lesion photo to view the predicted condition and probability breakdown across all 7 categories.
            </p>
            <Button asChild variant="primary" size="lg">
              <Link to={ctaRoute}>
                {ctaText}
                <FiArrowRight size={16} />
              </Link>
            </Button>
          </CtaCard>
        </Container>
      </SectionWrapper>

      {/* Minimalist Footer */}
      <FooterWrapper>
        <FooterContainer>
          <div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
              Skin Disease Classification Project
            </div>
            <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '4px' }}>
              Deep Learning Ensemble (ResNet-101 + DenseNet-121 + EfficientNet-B3) on HAM10000.
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.85rem', color: '#64748b' }}>
            <span>Educational ML Project</span>
            <span>•</span>
            <a
              href="https://github.com/Purvesh-PJ/skin_disease_detection"
              target="_blank"
              rel="noopener noreferrer"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: '#2563eb' }}
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
