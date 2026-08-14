import React, { useState } from 'react';
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
  FiActivity,
  FiCrosshair,
  FiGitBranch,
  FiBarChart2,
  FiCompass,
} from 'react-icons/fi';
import {
  LandingPageWrapper,
  HeroSection,
  HeroGlow,
  HeroPillBadge,
  HeroTitle,
  HeroSubtitle,
  HeroCtaRow,
  SandboxWrapper,
  SandboxCard,
  SandboxTopBar,
  SamplePillsRow,
  SamplePillBtn,
  SandboxGrid,
  LesionDisplayCard,
  ScanningReticle,
  ScannerHeaderBadge,
  ConsensusBreakdown,
  ConsensusBarRow,
  BarHeader,
  BarTrack,
  BarFill,
  SectionWrapper,
  Container,
  SectionHeader,
  SectionTag,
  SectionTitle,
  SectionDescription,
  BentoGrid,
  BentoCardLarge,
  BentoCardSmall,
  TabList,
  TabButton,
  TabContentCard,
  AtlasGrid,
  AtlasCard,
  RiskBadge,
  WorkflowGrid,
  WorkflowStep,
  StepIndexPill,
  DarkCtaSection,
  DarkCtaCard,
  FooterWrapper,
  FooterContainer,
} from './styles';

// Interactive Sample Cases for Hero Sandbox
const SAMPLE_CASES = [
  {
    id: 'sample-1',
    name: 'Melanocytic Nevus (nv)',
    category: 'Benign Proliferation',
    type: 'benign',
    confidence: 98.2,
    resnet: 97.5,
    densenet: 98.6,
    efficientnet: 98.8,
    markers: 'Symmetrical pigment network, regular globular borders.',
    recommendation: 'Routine annual dermatoscopy monitoring recommended.',
  },
  {
    id: 'sample-2',
    name: 'Malignant Melanoma (mel)',
    category: 'High-Risk Malignancy',
    type: 'malignant',
    confidence: 97.4,
    resnet: 96.9,
    densenet: 97.8,
    efficientnet: 98.1,
    markers: 'Asymmetric border irregularity, multiple color variegation.',
    recommendation: 'Immediate surgical excision biopsy and clinical staging.',
  },
  {
    id: 'sample-3',
    name: 'Vascular Angioma (vasc)',
    category: 'Benign Vascular',
    type: 'benign',
    confidence: 99.1,
    resnet: 98.8,
    densenet: 99.2,
    efficientnet: 99.4,
    markers: 'Well-demarcated red-purple lacunae, lack of pigment reticulum.',
    recommendation: 'Benign vascular lesion; no invasive intervention required.',
  },
];

// Interactive Architecture Tabs
const MODEL_TABS = [
  {
    id: 'resnet',
    name: 'ResNet-101',
    family: 'Residual Deep Networks',
    parameters: '44.5 Million Params',
    depth: '101 Convolutional Layers',
    receptiveField: 'Large Receptive Field (Residual Skip Highway)',
    description:
      'Employs identity shortcut connections to solve the vanishing gradient problem in deep neural networks. Excels at extracting fine-grained pigment boundary structures and macro lesion geometry.',
    keyAdvantage: 'Zero-degradation gradient flow across deep architectural layers.',
  },
  {
    id: 'densenet',
    name: 'DenseNet-121',
    family: 'Dense Feature Reuse',
    parameters: '8.0 Million Params',
    depth: '121 Interconnected Layers',
    receptiveField: 'Multi-Scale Dense Concatenation',
    description:
      'Directly connects every single layer to all subsequent layers in a feed-forward topology. Drastically enhances feature propagation and maximizes subtle cellular texture reuse without parameter bloat.',
    keyAdvantage: 'Maximum feature reuse with ultra-compact parameter footprint.',
  },
  {
    id: 'efficientnet',
    name: 'EfficientNet-B3',
    family: 'Compound Scaling CNN',
    parameters: '12.2 Million Params',
    depth: 'Compound Depth & Width Scaled',
    receptiveField: 'MBConv Inverted Residual Blocks',
    description:
      'Uniformly balances network depth, width, and input image resolution using a principled compound scaling coefficient. Delivers superior FLOPs efficiency and top-1 feature representation.',
    keyAdvantage: 'Optimal balance of computational efficiency and accuracy.',
  },
  {
    id: 'meta-ensemble',
    name: 'Logistic Meta-Ensemble',
    family: 'Stacked Probability Classifier',
    parameters: 'Meta-Calibration Matrix',
    depth: 'Softmax Stacking Layer',
    receptiveField: 'Multi-Logit Probability Fusion',
    description:
      'Takes the non-linear probability outputs and uncalibrated logits of all three base models and computes an optimal weighted meta-classification, virtually eliminating false positive rates.',
    keyAdvantage: 'High diagnostic consensus score with minimal variance.',
  },
];

// Pathology Atlas Data
const PATHOLOGY_CONDITIONS = [
  {
    code: 'mel',
    name: 'Malignant Melanoma',
    category: 'Malignant',
    type: 'malignant',
    incidence: 'High Severity',
    features: 'Asymmetric contours, atypical pigment network, blue-white veil.',
    guidance: 'Urgent histological examination and depth staging required.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    category: 'Malignant',
    type: 'malignant',
    incidence: 'Most Common Skin Cancer',
    features: 'Arborizing telangiectasia, translucent pearly border, ulceration.',
    guidance: 'Mohs micrographic surgery or topical therapy per clinical site.',
  },
  {
    code: 'akiec',
    name: 'Actinic Keratoses',
    category: 'Pre-Malignant',
    type: 'pre-malignant',
    incidence: 'UV-Induced Dysplasia',
    features: 'Erythematous scaly patches, keratotic plug, rosette signs.',
    guidance: 'Cryotherapy, photodynamic therapy, or topical 5-FU application.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    category: 'Benign',
    type: 'benign',
    incidence: 'Extremely Common',
    features: 'Stuck-on appearance, comedo-like openings, milia-like cysts.',
    guidance: 'Non-malignant; biopsy only if clinical ambiguity persists.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    category: 'Benign',
    type: 'benign',
    incidence: 'Cutaneous Fibroma',
    features: 'Central white patch, delicate peripheral pigment network, dimple sign.',
    guidance: 'Reassure patient; benign reactive fibrohistiocytic lesion.',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    category: 'Benign',
    type: 'benign',
    incidence: 'Common Mole',
    features: 'Uniform pigmentation, symmetric globular or reticular architecture.',
    guidance: 'Standard dermatoscopic surveillance if stability is confirmed.',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    category: 'Benign',
    type: 'benign',
    incidence: 'Vascular Malformation',
    features: 'Red-blue lacunae, absence of pigment network, hemorrhagic crust.',
    guidance: 'Clinical observation; pulsed dye laser for cosmetic concerns.',
  },
];

const Landing = ({ isAuthenticated }) => {
  const [activeSample, setActiveSample] = useState(SAMPLE_CASES[0]);
  const [activeModelTab, setActiveModelTab] = useState(MODEL_TABS[0]);

  const ctaRoute = isAuthenticated ? ROUTES.DASHBOARD : ROUTES.SIGNUP;
  const ctaText = isAuthenticated ? 'Open Diagnosis Dashboard' : 'Launch Free AI Analysis';

  return (
    <LandingPageWrapper id="overview">
      {/* Sticky Android Developers Pill Navigation */}
      <LandingNavbar isAuthenticated={isAuthenticated} />

      {/* Hero Section */}
      <HeroSection>
        <HeroGlow />

        <HeroPillBadge>
          <span className="dot" />
          <span>Material 3 • Tri-Model Deep Learning Pipeline</span>
        </HeroPillBadge>

        <HeroTitle>
          Clinical-Grade Skin Lesion Analysis with <span className="highlight">Ensemble AI</span>
        </HeroTitle>

        <HeroSubtitle>
          Experience multi-model consensus inference powered by ResNet-101, DenseNet-121, and
          EfficientNet-B3. Engineered to assist medical triage with real-time classification metrics.
        </HeroSubtitle>

        <HeroCtaRow>
          <Button asChild variant="android" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
          <Button asChild variant="secondary" size="lg">
            <a href="#sandbox">Explore Live Sandbox</a>
          </Button>
        </HeroCtaRow>

        {/* Live Interactive Diagnostic Sandbox */}
        <SandboxWrapper id="sandbox">
          <SandboxCard>
            <SandboxTopBar>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FiCrosshair size={20} style={{ color: '#16a34a' }} />
                <span style={{ fontWeight: 800, fontSize: '1.05rem' }}>
                  Interactive Dermoscopy Diagnostic Sandbox
                </span>
              </div>

              {/* Sample Switcher Pills */}
              <SamplePillsRow>
                {SAMPLE_CASES.map((sample) => (
                  <SamplePillBtn
                    key={sample.id}
                    $active={activeSample.id === sample.id}
                    onClick={() => setActiveSample(sample)}
                  >
                    {sample.name.split(' (')[0]}
                  </SamplePillBtn>
                ))}
              </SamplePillsRow>
            </SandboxTopBar>

            <SandboxGrid>
              {/* Lesion Simulator Card */}
              <LesionDisplayCard>
                <ScanningReticle />
                <ScannerHeaderBadge>
                  <FiActivity size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  Live Tensor Ingestion (224x224x3)
                </ScannerHeaderBadge>

                <div style={{ textAlign: 'center', padding: '24px', zIndex: 1 }}>
                  <FiZap size={40} style={{ color: '#3ddc84', marginBottom: '10px' }} />
                  <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#f8fafc' }}>
                    {activeSample.name}
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '4px' }}>
                    {activeSample.category}
                  </div>
                </div>
              </LesionDisplayCard>

              {/* Dynamic Consensus Meter */}
              <ConsensusBreakdown>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <RiskBadge $type={activeSample.type}>{activeSample.category}</RiskBadge>
                    <h3 style={{ fontSize: '1.35rem', fontWeight: 800, margin: '8px 0 2px 0' }}>
                      {activeSample.name}
                    </h3>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '1.6rem', fontWeight: 900, color: '#16a34a' }}>
                      {activeSample.confidence}%
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Ensemble Match</div>
                  </div>
                </div>

                {/* Progress Bars for Models */}
                <ConsensusBarRow>
                  <BarHeader>
                    <span>ResNet-101 Residual Logit</span>
                    <span className="val">{activeSample.resnet}%</span>
                  </BarHeader>
                  <BarTrack>
                    <BarFill $width={activeSample.resnet} $color="#0284c7" />
                  </BarTrack>
                </ConsensusBarRow>

                <ConsensusBarRow>
                  <BarHeader>
                    <span>DenseNet-121 Feature Reuse</span>
                    <span className="val">{activeSample.densenet}%</span>
                  </BarHeader>
                  <BarTrack>
                    <BarFill $width={activeSample.densenet} $color="#6366f1" />
                  </BarTrack>
                </ConsensusBarRow>

                <ConsensusBarRow>
                  <BarHeader>
                    <span>EfficientNet-B3 Scaled Probability</span>
                    <span className="val">{activeSample.efficientnet}%</span>
                  </BarHeader>
                  <BarTrack>
                    <BarFill $width={activeSample.efficientnet} $color="#16a34a" />
                  </BarTrack>
                </ConsensusBarRow>

                {/* Clinical Notes Box */}
                <div
                  style={{
                    background: 'rgba(0, 0, 0, 0.03)',
                    borderRadius: '16px',
                    padding: '12px 16px',
                    fontSize: '0.85rem',
                    color: '#4b5563',
                    lineHeight: 1.5,
                  }}
                >
                  <strong>Dermoscopic Hallmark:</strong> {activeSample.markers}
                </div>
              </ConsensusBreakdown>
            </SandboxGrid>
          </SandboxCard>
        </SandboxWrapper>
      </HeroSection>

      {/* Section 2: Asymmetric Bento Grid (Tonal Mint) */}
      <SectionWrapper id="architecture" $bg="mint">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiLayers size={14} />
              <span>Engineering & Scale</span>
            </SectionTag>
            <SectionTitle>Engineered with Asymmetric Precision</SectionTitle>
            <SectionDescription>
              A high-throughput convolutional pipeline fusing three specialized vision architectures
              to achieve clinical-grade generalization across heterogeneous patient data.
            </SectionDescription>
          </SectionHeader>

          <BentoGrid>
            {/* Bento Card 1 (Large 7-col) */}
            <BentoCardLarge>
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 12px',
                    background: 'rgba(61, 220, 132, 0.15)',
                    borderRadius: '9999px',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    color: '#065f46',
                    marginBottom: '16px',
                  }}
                >
                  <FiGitBranch size={14} />
                  Tri-Model Parallel Pipeline
                </div>
                <h3 style={{ fontSize: '1.75rem', fontWeight: 800, margin: '0 0 12px 0' }}>
                  Parallel Multi-CNN Ingestion Engine
                </h3>
                <p style={{ color: '#64748b', lineHeight: 1.65, margin: '0 0 24px 0' }}>
                  Raw dermoscopic photographs undergo simultaneous CLAHE contrast normalization before
                  being routed into three distinct neural networks in parallel threads, extracting
                  spatial, textural, and boundary features concurrently.
                </p>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '12px',
                  background: 'rgba(0,0,0,0.02)',
                  padding: '16px',
                  borderRadius: '20px',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Network A</div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: '#0284c7' }}>ResNet-101</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Network B</div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: '#6366f1' }}>DenseNet-121</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Network C</div>
                  <div style={{ fontSize: '1rem', fontWeight: 800, color: '#16a34a' }}>
                    EfficientNet-B3
                  </div>
                </div>
              </div>
            </BentoCardLarge>

            {/* Bento Card 2 (Small 5-col) */}
            <BentoCardSmall>
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 12px',
                    background: 'rgba(99, 102, 241, 0.15)',
                    borderRadius: '9999px',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    color: '#4338ca',
                    marginBottom: '16px',
                  }}
                >
                  <FiZap size={14} />
                  Latency Telemetry
                </div>
                <h3 style={{ fontSize: '1.5rem', fontWeight: 800, margin: '0 0 8px 0' }}>
                  &lt; 1.2s Real-Time Inference
                </h3>
                <p style={{ color: '#64748b', lineHeight: 1.6, fontSize: '0.925rem' }}>
                  Optimized for rapid triage with asynchronous batch evaluation and GPU TensorRT execution.
                </p>
              </div>

              <div style={{ marginTop: '20px' }}>
                <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#073042' }}>64.7M</div>
                <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Total Parameters Evaluated</div>
              </div>
            </BentoCardSmall>

            {/* Bento Card 3 (Small 5-col) */}
            <BentoCardSmall>
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 12px',
                    background: 'rgba(245, 158, 11, 0.15)',
                    borderRadius: '9999px',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    color: '#92400e',
                    marginBottom: '16px',
                  }}
                >
                  <FiBarChart2 size={14} />
                  Clinical Benchmark
                </div>
                <h3 style={{ fontSize: '1.5rem', fontWeight: 800, margin: '0 0 8px 0' }}>
                  94.8% HAM10000 F1-Score
                </h3>
                <p style={{ color: '#64748b', lineHeight: 1.6, fontSize: '0.925rem' }}>
                  Validated across 10,015 multi-source dermatoscopic images covering 7 distinct pathologies.
                </p>
              </div>

              <div style={{ marginTop: '20px' }}>
                <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#16a34a' }}>7 Classes</div>
                <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Full Histological Diversity</div>
              </div>
            </BentoCardSmall>

            {/* Bento Card 4 (Large 7-col) */}
            <BentoCardLarge>
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 12px',
                    background: 'rgba(61, 220, 132, 0.15)',
                    borderRadius: '9999px',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    color: '#065f46',
                    marginBottom: '16px',
                  }}
                >
                  <FiCompass size={14} />
                  Meta-Classification Stacking
                </div>
                <h3 style={{ fontSize: '1.75rem', fontWeight: 800, margin: '0 0 12px 0' }}>
                  Stacked Logistic Probability Fusion
                </h3>
                <p style={{ color: '#64748b', lineHeight: 1.65, margin: '0 0 20px 0' }}>
                  Instead of naive majority voting, our second-tier meta-learner computes an optimized
                  weighted consensus across all base logits, suppressing outliers and false positives.
                </p>
              </div>

              <div
                style={{
                  background: 'rgba(7, 48, 66, 0.04)',
                  padding: '14px 20px',
                  borderRadius: '16px',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  color: '#073042',
                  fontWeight: 600,
                }}
              >
                P(Class | x) = σ( W_meta · [P_ResNet, P_DenseNet, P_EfficientNet] + b )
              </div>
            </BentoCardLarge>
          </BentoGrid>
        </Container>
      </SectionWrapper>

      {/* Section 3: Interactive Tabbed Architecture Showcase (Tonal Indigo) */}
      <SectionWrapper id="models" $bg="indigo">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiCpu size={14} />
              <span>Model Architecture Deep Dive</span>
            </SectionTag>
            <SectionTitle>Interactive Vision Topology</SectionTitle>
            <SectionDescription>
              Explore how each neural network in the tri-model ensemble contributes unique receptive
              fields and feature representations.
            </SectionDescription>
          </SectionHeader>

          {/* Android Dev Tab List */}
          <TabList>
            {MODEL_TABS.map((tab) => (
              <TabButton
                key={tab.id}
                $active={activeModelTab.id === tab.id}
                onClick={() => setActiveModelTab(tab)}
              >
                {tab.name}
              </TabButton>
            ))}
          </TabList>

          {/* Dynamic Tab Content Card */}
          <TabContentCard>
            <div>
              <div
                style={{
                  display: 'inline-block',
                  padding: '4px 12px',
                  borderRadius: '9999px',
                  background: 'rgba(99, 102, 241, 0.1)',
                  color: '#4338ca',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  marginBottom: '12px',
                }}
              >
                {activeModelTab.family}
              </div>

              <h3 style={{ fontSize: '2rem', fontWeight: 800, margin: '0 0 12px 0' }}>
                {activeModelTab.name}
              </h3>

              <p style={{ color: '#64748b', fontSize: '1.05rem', lineHeight: 1.65, margin: '0 0 20px 0' }}>
                {activeModelTab.description}
              </p>

              <div
                style={{
                  background: 'rgba(61, 220, 132, 0.1)',
                  border: '1px solid rgba(61, 220, 132, 0.3)',
                  padding: '12px 18px',
                  borderRadius: '16px',
                  fontSize: '0.9rem',
                  color: '#065f46',
                  fontWeight: 600,
                }}
              >
                💡 <strong>Core Advantage:</strong> {activeModelTab.keyAdvantage}
              </div>
            </div>

            {/* Spec Cards Column */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                background: 'rgba(0, 0, 0, 0.02)',
                padding: '24px',
                borderRadius: '24px',
              }}
            >
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Parameter Footprint</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#111827' }}>
                  {activeModelTab.parameters}
                </div>
              </div>

              <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', paddingTop: '12px' }}>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Network Depth</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#111827' }}>
                  {activeModelTab.depth}
                </div>
              </div>

              <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', paddingTop: '12px' }}>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Receptive Field Architecture</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0284c7' }}>
                  {activeModelTab.receptiveField}
                </div>
              </div>
            </div>
          </TabContentCard>
        </Container>
      </SectionWrapper>

      {/* Section 4: Clinical Pathology Atlas (Tonal Sand) */}
      <SectionWrapper id="conditions" $bg="sand">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiCheckCircle size={14} />
              <span>Pathology Atlas</span>
            </SectionTag>
            <SectionTitle>7 Supported Clinical Categories</SectionTitle>
            <SectionDescription>
              Comprehensive diagnostic coverage mapped to the HAM10000 international dermatology benchmark.
            </SectionDescription>
          </SectionHeader>

          <AtlasGrid>
            {PATHOLOGY_CONDITIONS.map((condition) => (
              <AtlasCard key={condition.code}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <RiskBadge $type={condition.type}>{condition.category}</RiskBadge>
                  <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 700 }}>
                    {condition.code.toUpperCase()}
                  </span>
                </div>

                <h4 style={{ fontSize: '1.25rem', fontWeight: 800, margin: '6px 0 0 0' }}>
                  {condition.name}
                </h4>

                <div style={{ fontSize: '0.8rem', color: '#073042', fontWeight: 700 }}>
                  {condition.incidence}
                </div>

                <p style={{ fontSize: '0.875rem', color: '#4b5563', lineHeight: 1.5, margin: 0 }}>
                  <strong>Key Hallmarks:</strong> {condition.features}
                </p>

                <div
                  style={{
                    marginTop: 'auto',
                    paddingTop: '12px',
                    borderTop: '1px solid rgba(0,0,0,0.06)',
                    fontSize: '0.8rem',
                    color: '#6b7280',
                  }}
                >
                  📋 {condition.guidance}
                </div>
              </AtlasCard>
            ))}
          </AtlasGrid>
        </Container>
      </SectionWrapper>

      {/* Section 5: Clinician-in-the-Loop Workflow (Tonal Ice) */}
      <SectionWrapper id="workflow" $bg="ice">
        <Container>
          <SectionHeader>
            <SectionTag>
              <FiShield size={14} />
              <span>Clinical Responsibility</span>
            </SectionTag>
            <SectionTitle>3-Stage Clinician Decision Pipeline</SectionTitle>
            <SectionDescription>
              Designed as an assistive pre-screening tool that complements professional medical review.
            </SectionDescription>
          </SectionHeader>

          <WorkflowGrid>
            <WorkflowStep>
              <StepIndexPill>01</StepIndexPill>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, margin: '8px 0 0 0' }}>
                High-Res Dermoscopy Ingestion
              </h3>
              <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
                High-resolution epiluminescence photographs are captured under polarized light,
                normalized for lighting variations, and uploaded directly to the platform.
              </p>
            </WorkflowStep>

            <WorkflowStep>
              <StepIndexPill>02</StepIndexPill>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, margin: '8px 0 0 0' }}>
                Tri-Model Ensemble Inference
              </h3>
              <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
                ResNet-101, DenseNet-121, and EfficientNet-B3 independently extract feature representations
                and generate cross-calibrated probability distributions in under 1.2 seconds.
              </p>
            </WorkflowStep>

            <WorkflowStep>
              <StepIndexPill>03</StepIndexPill>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, margin: '8px 0 0 0' }}>
                Physician Review & Histology
              </h3>
              <p style={{ fontSize: '0.925rem', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
                The board-certified clinician evaluates the AI consensus report, incorporates patient history,
                and determines next clinical steps or confirmatory surgical excision.
              </p>
            </WorkflowStep>
          </WorkflowGrid>
        </Container>
      </SectionWrapper>

      {/* Section 6: Dark Pine & Android Green CTA */}
      <DarkCtaSection>
        <DarkCtaCard>
          <div
            style={{
              padding: '6px 16px',
              borderRadius: '9999px',
              background: 'rgba(61, 220, 132, 0.15)',
              color: '#3ddc84',
              fontWeight: 700,
              fontSize: '0.85rem',
              marginBottom: '16px',
            }}
          >
            ⚡ Open Access Dermoscopy AI
          </div>

          <h2 style={{ fontSize: '2.75rem', fontWeight: 800, margin: '0 0 16px 0', letterSpacing: '-0.03em' }}>
            Ready to explore clinical-grade lesion detection?
          </h2>

          <p style={{ fontSize: '1.15rem', opacity: 0.85, maxWidth: '640px', margin: '0 0 32px 0', lineHeight: 1.65 }}>
            Upload any dermoscopic photograph and obtain instantaneous tri-model consensus confidence
            breakdowns across 7 disease classifications.
          </p>

          <Button asChild variant="android" size="lg">
            <Link to={ctaRoute}>
              {ctaText}
              <FiArrowRight size={16} />
            </Link>
          </Button>
        </DarkCtaCard>
      </DarkCtaSection>

      {/* Minimalist Footer */}
      <FooterWrapper>
        <FooterContainer>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', color: '#073042' }}>
              SkinAI Diagnostics Pipeline
            </div>
            <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '4px' }}>
              Tri-Model Deep Convolutional Ensemble for Dermoscopic Lesion Classification.
            </div>
          </div>

          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            © {new Date().getFullYear()} Skin AI Research. Assistive Clinical AI Tool.
          </div>
        </FooterContainer>
      </FooterWrapper>
    </LandingPageWrapper>
  );
};

export default Landing;
