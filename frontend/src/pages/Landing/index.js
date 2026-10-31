import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ROUTES } from '../../constants';
import Button from '../../components/ui/Button';
import LandingNavbar from '../../components/layout/LandingNavbar';
import {
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

// 7 Supported Skin Diseases with Clinical Details
const ALL_CONDITIONS = [
  {
    code: 'mel',
    name: 'Melanoma',
    category: 'malignant',
    riskText: 'Malignant (High Priority)',
    riskColor: '#ef4444',
    description: 'A serious malignancy arising from pigment-producing melanocytes. Often presents with asymmetrical contours, irregular borders, and color variations.',
  },
  {
    code: 'bcc',
    name: 'Basal Cell Carcinoma',
    category: 'malignant',
    riskText: 'Malignant (Common)',
    riskColor: '#ef4444',
    description: 'The most frequent form of skin cancer. Typically appears as pearly translucent papules with telangiectasia; rarely metastasizes but causes local tissue invasion.',
  },
  {
    code: 'akiec',
    name: "Actinic Keratoses / Bowen's Disease",
    category: 'precancer',
    riskText: 'Pre-Cancerous',
    riskColor: '#f59e0b',
    description: 'Rough, dry, scaly patches on chronically sun-exposed areas caused by cumulative UV damage. Carries potential for progression to invasive squamous cell carcinoma.',
  },
  {
    code: 'nv',
    name: 'Melanocytic Nevi',
    category: 'benign',
    riskText: 'Benign (Harmless)',
    riskColor: '#22c55e',
    description: 'Common benign moles and birthmarks formed by ordered nests of melanocytes with uniform pigment distribution and symmetrical borders.',
  },
  {
    code: 'bkl',
    name: 'Benign Keratosis',
    category: 'benign',
    riskText: 'Benign (Harmless)',
    riskColor: '#22c55e',
    description: 'Non-malignant epidermal growths, including seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses common in adults.',
  },
  {
    code: 'df',
    name: 'Dermatofibroma',
    category: 'benign',
    riskText: 'Benign (Harmless)',
    riskColor: '#22c55e',
    description: 'Firm, slowly developing non-cancerous fibrous cutaneous nodules, most frequently observed on the lower extremities and shoulder girdle.',
  },
  {
    code: 'vasc',
    name: 'Vascular Lesions',
    category: 'benign',
    riskText: 'Benign (Harmless)',
    riskColor: '#22c55e',
    description: 'Benign vascular anomalies including cherry angiomas, pyogenic granulomas, angiokeratomas, and localized capillary malformations.',
  },
];

const Landing = () => {
  const [selectedFilter, setSelectedFilter] = useState('all');
  const ctaRoute = ROUTES.DASHBOARD;
  const ctaText = 'Try Image Detection';

  const filteredConditions =
    selectedFilter === 'all'
      ? ALL_CONDITIONS
      : ALL_CONDITIONS.filter((c) => c.category === selectedFilter);

  return (
    <div id="overview" className="min-h-screen bg-slate-50 dark:bg-[#0a0a0a] text-slate-900 dark:text-slate-100 overflow-x-hidden">
      {/* Top Navigation */}
      <LandingNavbar />

      {/* SECTION 1: HERO & PROJECT INTRODUCTION */}
      <section className="relative px-4 sm:px-6 pt-10 sm:pt-16 pb-8 sm:pb-12 max-w-[1240px] mx-auto flex flex-col items-center text-center">
        {/* Hero glow ambient overlay */}
        <div
          className="absolute top-0 left-1/2 -translate-x-1/2 w-screen h-[480px] pointer-events-none z-0"
          style={{
            background: 'radial-gradient(ellipse 60% 50% at 50% 0%, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0) 75%)',
          }}
        />

        <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_0.85fr] gap-8 items-center w-full max-w-[1240px] relative z-10 mb-8 text-center lg:text-left">
          {/* Left Column: Hero Content */}
          <div className="flex flex-col items-center lg:items-start">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white dark:bg-[#181818] border border-green-200 dark:border-green-500/30 rounded-full text-green-700 dark:text-green-400 text-[0.8125rem] font-semibold mb-5 shadow-sm relative z-10">
              <FiCpu size={14} />
              <span>Dermatological AI Research • Ensemble Deep Learning</span>
            </div>

            <h1 className="text-3xl sm:text-4xl lg:text-[3.15rem] font-extrabold leading-[1.15] tracking-tight max-w-[680px] mb-4 text-slate-900 dark:text-slate-50 relative z-10">
              Skin Disease Classification via{' '}
              <span className="text-green-600 dark:text-green-400">Ensemble Deep Learning</span>
            </h1>

            <p className="text-[1.1rem] leading-[1.65] text-slate-600 dark:text-slate-400 max-w-[620px] mb-6 relative z-10">
              An academic machine learning project designed to assist in early skin disease screening.
              Trained on 10,015 dermatoscopic images from Kaggle's HAM10000 benchmark, our system combines three
              distinct deep CNN topologies with a logistic stacking layer to detect and categorize 7 skin conditions.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 w-full sm:w-auto relative z-10">
              <Button asChild variant="brand" size="lg" className="w-full sm:w-auto">
                <Link to={ctaRoute}>
                  {ctaText}
                </Link>
              </Button>

              <Button asChild variant="outline" size="lg" className="w-full sm:w-auto">
                <a href="#how-it-works">How It Works</a>
              </Button>
            </div>
          </div>

          {/* Right Column: Disease + AI Dermoscopy Scanner Visual SVG */}
          <div className="flex justify-center items-center w-full relative">
            <div className="w-full max-w-[500px]">
              <svg viewBox="0 0 460 380" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto block">
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
            </div>
          </div>
        </div>

        {/* 4 Core Pillars of the Project */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 w-full max-w-[1040px] mt-4 relative z-10">
          <div className="flex items-center gap-3 py-2 text-left">
            <div className="text-green-600 dark:text-green-400 text-xl shrink-0">
              <FiLayers />
            </div>
            <div className="flex flex-col">
              <strong className="text-xs font-bold text-slate-900 dark:text-slate-100">Stacked Ensemble</strong>
              <span className="text-[0.75rem] text-slate-500 dark:text-slate-400">ResNet + DenseNet + EfficientNet</span>
            </div>
          </div>

          <div className="flex items-center gap-3 py-2 text-left">
            <div className="text-green-600 dark:text-green-400 text-xl shrink-0">
              <FiDatabase />
            </div>
            <div className="flex flex-col">
              <strong className="text-xs font-bold text-slate-900 dark:text-slate-100">HAM10000 Dataset</strong>
              <span className="text-[0.75rem] text-slate-500 dark:text-slate-400">10,015 Verified Images</span>
            </div>
          </div>

          <div className="flex items-center gap-3 py-2 text-left">
            <div className="text-green-600 dark:text-green-400 text-xl shrink-0">
              <FiZap />
            </div>
            <div className="flex flex-col">
              <strong className="text-xs font-bold text-slate-900 dark:text-slate-100">Real-Time Screening</strong>
              <span className="text-[0.75rem] text-slate-500 dark:text-slate-400">Sub-Second Inference API</span>
            </div>
          </div>

          <div className="flex items-center gap-3 py-2 text-left">
            <div className="text-green-600 dark:text-green-400 text-xl shrink-0">
              <FiActivity />
            </div>
            <div className="flex flex-col">
              <strong className="text-xs font-bold text-slate-900 dark:text-slate-100">7 Disease Classes</strong>
              <span className="text-[0.75rem] text-slate-500 dark:text-slate-400">Malignant & Benign Analysis</span>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: HOW IT WORKS (END-TO-END DATA FLOW) */}
      <section id="how-it-works" className="py-12 sm:py-16 px-4 sm:px-6 bg-white dark:bg-[#121212]">
        <div className="max-w-[1140px] mx-auto">
          <div className="text-center max-w-[780px] mx-auto mb-10">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 bg-slate-100 dark:bg-[#1e1e1e] border border-slate-200 dark:border-[#2e2e2e] rounded-full text-slate-600 dark:text-slate-400 text-[0.8125rem] font-semibold mb-3">
              <FiActivity size={14} />
              <span>Step-by-Step Workflow</span>
            </div>
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              How the Detection Pipeline Works
            </h2>
            <p className="text-[1.05rem] text-slate-600 dark:text-slate-400 leading-[1.65]">
              From raw dermatoscopic image ingestion to calibrated probability estimation,
              here is the sequential process executed by the backend on every scan.
            </p>
          </div>

          {/* 4-Step Process Walkthrough */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 relative">
            <div className="flex flex-col gap-2.5">
              <div className="text-[0.8125rem] font-extrabold tracking-wider text-green-600 dark:text-green-400 flex items-center gap-2 after:content-[''] after:flex-1 after:h-[1px] after:bg-slate-200 dark:after:bg-[#262626]">
                STEP 01
              </div>
              <h3 className="text-[1.15rem] font-bold m-0 text-slate-900 dark:text-slate-100">Image Ingestion</h3>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                The user uploads a close-up dermatoscopic image of a suspicious skin spot or mole via the web dashboard.
              </p>
            </div>

            <div className="flex flex-col gap-2.5">
              <div className="text-[0.8125rem] font-extrabold tracking-wider text-green-600 dark:text-green-400 flex items-center gap-2 after:content-[''] after:flex-1 after:h-[1px] after:bg-slate-200 dark:after:bg-[#262626]">
                STEP 02
              </div>
              <h3 className="text-[1.15rem] font-bold m-0 text-slate-900 dark:text-slate-100">CLAHE Preprocessing</h3>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                OpenCV resizes the image to 224×224 pixels and applies contrast-limited adaptive histogram equalization to enhance pigment patterns.
              </p>
            </div>

            <div className="flex flex-col gap-2.5">
              <div className="text-[0.8125rem] font-extrabold tracking-wider text-green-600 dark:text-green-400 flex items-center gap-2 after:content-[''] after:flex-1 after:h-[1px] after:bg-slate-200 dark:after:bg-[#262626]">
                STEP 03
              </div>
              <h3 className="text-[1.15rem] font-bold m-0 text-slate-900 dark:text-slate-100">Tri-Model Inference</h3>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                ResNet-101, DenseNet-121, and EfficientNet-B3 simultaneously extract spatial features and generate probability logits.
              </p>
            </div>

            <div className="flex flex-col gap-2.5">
              <div className="text-[0.8125rem] font-extrabold tracking-wider text-green-600 dark:text-green-400 flex items-center gap-2 after:content-[''] after:flex-1 after:h-[1px] after:bg-slate-200 dark:after:bg-[#262626]">
                STEP 04
              </div>
              <h3 className="text-[1.15rem] font-bold m-0 text-slate-900 dark:text-slate-100">Stacked Prediction</h3>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                A logistic meta-classifier aggregates all 3 outputs, calculating the final disease classification and calibrated confidence score.
              </p>
            </div>
          </div>

          {/* Clean Outlined Pipeline Blueprint Diagram */}
          <div className="w-full bg-[#fafafa] dark:bg-[#141414] border border-slate-200 dark:border-[#262626] rounded-2xl p-4 sm:p-6 lg:p-8 mt-8 overflow-x-auto">
            <svg viewBox="0 0 1020 320" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full min-w-[900px] h-auto block">
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
          </div>
        </div>
      </section>

      {/* SECTION 3: DEEP ENSEMBLE AI ARCHITECTURES */}
      <section id="architecture" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-[1140px] mx-auto">
          <div className="text-center max-w-[780px] mx-auto mb-10">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 bg-slate-100 dark:bg-[#1e1e1e] border border-slate-200 dark:border-[#2e2e2e] rounded-full text-slate-600 dark:text-slate-400 text-[0.8125rem] font-semibold mb-3">
              <FiLayers size={14} />
              <span>Multi-Model Topologies</span>
            </div>
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              Why Ensemble Stacking?
            </h2>
            <p className="text-[1.05rem] text-slate-600 dark:text-slate-400 leading-[1.65]">
              A single convolutional neural network often exhibits inductive bias and can struggle with diverse skin phototypes.
              By stacking three architecturally distinct CNN backbones, our model achieves superior generalizability and consensus calibration.
            </p>
          </div>

          <div className="flex flex-col gap-10">
            {/* Model 1: ResNet-101 */}
            <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_0.85fr] gap-6 lg:gap-10 items-center pb-10 border-b border-slate-200 dark:border-[#222222]">
              <div className="flex flex-col gap-3">
                <div className="flex items-center flex-wrap gap-3">
                  <h3 className="text-[1.35rem] font-extrabold m-0 text-slate-900 dark:text-slate-100">ResNet-101</h3>
                  <span className="inline-block px-2.5 py-0.5 bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/25 rounded-full text-xs font-bold">
                    44.5M Params • Structural Specialist
                  </span>
                </div>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  Identity Skip Highways for Lesion Borders & Asymmetry
                </span>
                <p className="text-slate-600 dark:text-slate-300 text-[0.925rem] leading-[1.65] m-0">
                  <strong>How it works:</strong> Standard deep networks suffer from signal loss as layers get deeper.
                  ResNet-101 solves this with <em>shortcut connections</em> (<code>F(x) + x</code>) that let gradients
                  travel directly across 101 layers without vanishing.
                </p>
                <p className="text-slate-500 dark:text-slate-400 text-sm leading-[1.6] m-0">
                  <strong>Clinical Role:</strong> Melanoma and malignant tumors often have jagged, asymmetrical borders.
                  ResNet’s ultra-deep receptive field is specifically tuned to detect these macro-structural contour anomalies.
                </p>
              </div>

              <div className="w-full max-w-[440px] mx-auto flex items-center justify-center">
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto max-h-[140px] block">
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
              </div>
            </div>

            {/* Model 2: DenseNet-121 */}
            <div className="grid grid-cols-1 lg:grid-cols-[0.85fr_1.15fr] gap-6 lg:gap-10 items-center pb-10 border-b border-slate-200 dark:border-[#222222]">
              <div className="order-2 lg:order-1 w-full max-w-[440px] mx-auto flex items-center justify-center">
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto max-h-[140px] block">
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
              </div>

              <div className="order-1 lg:order-2 flex flex-col gap-3">
                <div className="flex items-center flex-wrap gap-3">
                  <h3 className="text-[1.35rem] font-extrabold m-0 text-slate-900 dark:text-slate-100">DenseNet-121</h3>
                  <span className="inline-block px-2.5 py-0.5 bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/25 rounded-full text-xs font-bold">
                    8.0M Params • Texture Specialist
                  </span>
                </div>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  Cross-Layer Concatenation for Micro-Pigment Patterns
                </span>
                <p className="text-slate-600 dark:text-slate-300 text-[0.925rem] leading-[1.65] m-0">
                  <strong>How it works:</strong> Instead of summing outputs, every layer in DenseNet directly connects to
                  <em>all subsequent layers</em> in the network. This maximizes gradient reuse and ensures no subtle details
                  get lost in transit.
                </p>
                <p className="text-slate-500 dark:text-slate-400 text-sm leading-[1.6] m-0">
                  <strong>Clinical Role:</strong> Benign keratoses and subtle vascular spots are differentiated by tiny cellular
                  pigment granules. DenseNet excels at recognizing these fine dermatoscopic texture patterns.
                </p>
              </div>
            </div>

            {/* Model 3: EfficientNet-B3 */}
            <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_0.85fr] gap-6 lg:gap-10 items-center pb-10 border-b border-slate-200 dark:border-[#222222]">
              <div className="flex flex-col gap-3">
                <div className="flex items-center flex-wrap gap-3">
                  <h3 className="text-[1.35rem] font-extrabold m-0 text-slate-900 dark:text-slate-100">EfficientNet-B3</h3>
                  <span className="inline-block px-2.5 py-0.5 bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/25 rounded-full text-xs font-bold">
                    12.2M Params • Scale Specialist
                  </span>
                </div>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  Compound Scaling across Depth, Width & Image Resolution
                </span>
                <p className="text-slate-600 dark:text-slate-300 text-[0.925rem] leading-[1.65] m-0">
                  <strong>How it works:</strong> Rather than arbitrarily making the network deeper or wider, EfficientNet scales
                  <em>Depth (d)</em>, <em>Width (w)</em>, and <em>Resolution (r)</em> simultaneously using a mathematically principled
                  compound scaling coefficient.
                </p>
                <p className="text-slate-500 dark:text-slate-400 text-sm leading-[1.6] m-0">
                  <strong>Clinical Role:</strong> Dermatological images vary widely in zoom and lesion size. EfficientNet balances
                  both macro lesion overview and micro zoom details with supreme computational speed.
                </p>
              </div>

              <div className="w-full max-w-[440px] mx-auto flex items-center justify-center">
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto max-h-[140px] block">
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
              </div>
            </div>

            {/* Model 4: Meta-Classifier */}
            <div className="grid grid-cols-1 lg:grid-cols-[0.85fr_1.15fr] gap-6 lg:gap-10 items-center">
              <div className="order-2 lg:order-1 w-full max-w-[440px] mx-auto flex items-center justify-center">
                <svg viewBox="0 0 360 110" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto max-h-[140px] block">
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
              </div>

              <div className="order-1 lg:order-2 flex flex-col gap-3">
                <div className="flex items-center flex-wrap gap-3">
                  <h3 className="text-[1.35rem] font-extrabold m-0 text-slate-900 dark:text-slate-100">Logistic Meta-Learner</h3>
                  <span className="inline-block px-2.5 py-0.5 bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/25 rounded-full text-xs font-bold">
                    Stacking Arbiter • Softmax Consensus
                  </span>
                </div>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  Learned Probability Weighting for Calibrated Diagnoses
                </span>
                <p className="text-slate-600 dark:text-slate-300 text-[0.925rem] leading-[1.65] m-0">
                  <strong>How it works:</strong> Rather than taking a simple unweighted average, a second-stage Logistic Regression
                  meta-model learns the individual reliability and accuracy weights ($w_1, w_2, w_3$) of each CNN on validation data.
                </p>
                <p className="text-slate-500 dark:text-slate-400 text-sm leading-[1.6] m-0">
                  <strong>Clinical Role:</strong> In healthcare, confidence calibration is essential. The meta-learner ensures that
                  when a high risk prediction like Melanoma is flagged, it is backed by calibrated multi-model consensus.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 4: DATASET & TRAINING METHODOLOGY */}
      <section id="dataset" className="py-12 sm:py-16 px-4 sm:px-6 bg-white dark:bg-[#121212]">
        <div className="max-w-[1140px] mx-auto">
          <div className="text-center max-w-[780px] mx-auto mb-10">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 bg-slate-100 dark:bg-[#1e1e1e] border border-slate-200 dark:border-[#2e2e2e] rounded-full text-slate-600 dark:text-slate-400 text-[0.8125rem] font-semibold mb-3">
              <FiDatabase size={14} />
              <span>Training Corpus & Preprocessing</span>
            </div>
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              Trained on Kaggle's HAM10000 Dataset
            </h2>
            <p className="text-[1.05rem] text-slate-600 dark:text-slate-400 leading-[1.65]">
              10,015 multi-source dermatoscopic images processed through patient-isolated partitioning, dynamic balancing, and CLAHE normalization.
            </p>
          </div>

          {/* Minimalist Floating Stats Strip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6 py-6 border-y border-slate-200 dark:border-[#222222] mb-8">
            <div className="flex flex-col gap-1">
              <div className="text-[2rem] font-extrabold text-green-600 dark:text-green-400 tracking-tight leading-none">10,015</div>
              <div className="text-[0.8125rem] text-slate-500 dark:text-slate-400 font-semibold">Curated Dermoscopy Scans</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="text-[2rem] font-extrabold text-green-600 dark:text-green-400 tracking-tight leading-none">7 Classes</div>
              <div className="text-[0.8125rem] text-slate-500 dark:text-slate-400 font-semibold">Full Diagnostic Scope</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="text-[2rem] font-extrabold text-green-600 dark:text-green-400 tracking-tight leading-none">224 × 224</div>
              <div className="text-[0.8125rem] text-slate-500 dark:text-slate-400 font-semibold">Normalized RGB Matrix</div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="text-[2rem] font-extrabold text-green-600 dark:text-green-400 tracking-tight leading-none">70 / 15 / 15</div>
              <div className="text-[0.8125rem] text-slate-500 dark:text-slate-400 font-semibold">Patient-Isolated Split</div>
            </div>
          </div>

          {/* Minimalist Full-Width Class Distribution Spectrum */}
          <div className="flex flex-col gap-3 mb-10">
            <div className="flex justify-between items-center text-[0.825rem] font-semibold text-slate-500 dark:text-slate-400">
              <span>Dataset Class Distribution (10,015 Images)</span>
              <span className="hidden sm:inline">Balanced via Dynamic Class Weighting & Augmentation</span>
            </div>
            <div className="flex h-3 rounded-md overflow-hidden bg-slate-200 dark:bg-[#262626]">
              <div style={{ width: '67%', background: '#22c55e' }} title="NV: 6,705 (67%)" />
              <div style={{ width: '11%', background: '#ef4444' }} title="MEL: 1,113 (11%)" />
              <div style={{ width: '11%', background: '#10b981' }} title="BKL: 1,099 (11%)" />
              <div style={{ width: '5%', background: '#dc2626' }} title="BCC: 514 (5%)" />
              <div style={{ width: '3%', background: '#f59e0b' }} title="AKIEC: 327 (3%)" />
              <div style={{ width: '1.5%', background: '#3b82f6' }} title="VASC: 142 (1.5%)" />
              <div style={{ width: '1.5%', background: '#8b5cf6' }} title="DF: 115 (1.5%)" />
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-[0.775rem] text-slate-500 dark:text-slate-400">
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#22c55e' }} /><span>NV: 67% (Nevi)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#ef4444' }} /><span>MEL: 11% (Melanoma)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#10b981' }} /><span>BKL: 11% (Keratosis)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#dc2626' }} /><span>BCC: 5% (Carcinoma)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#f59e0b' }} /><span>AKIEC: 3% (Actinic)</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#3b82f6' }} /><span>VASC: 1.5%</span></div>
              <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#8b5cf6' }} /><span>DF: 1.5%</span></div>
            </div>
          </div>

          {/* Connected 3-Step Milestone Pipeline */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 relative">
            <div className="flex flex-col gap-2">
              <div className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-green-50 dark:bg-green-500/15 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/30 text-[0.8rem] font-extrabold mb-1">
                01
              </div>
              <h4 className="text-[1.05rem] font-bold m-0 text-slate-900 dark:text-slate-100">Patient-Isolated Partitioning</h4>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                Images are split by patient ID so scans from the same individual never overlap across train and test folds, guaranteeing authentic generalization.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-green-50 dark:bg-green-500/15 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/30 text-[0.8rem] font-extrabold mb-1">
                02
              </div>
              <h4 className="text-[1.05rem] font-bold m-0 text-slate-900 dark:text-slate-100">Dynamic Rebalancing & Augmentation</h4>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                On-the-fly random rotations (0°–360°), flips, and loss class-weighting prevent majority bias toward common benign moles.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-green-50 dark:bg-green-500/15 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-500/30 text-[0.8rem] font-extrabold mb-1">
                03
              </div>
              <h4 className="text-[1.05rem] font-bold m-0 text-slate-900 dark:text-slate-100">CLAHE Contrast & Normalization</h4>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-[1.6] m-0">
                Adaptive histogram equalization enhances subtle pigment network boundaries, followed by RGB tensor scaling to [0, 1] for stable gradient descent.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 5: 7 SUPPORTED SKIN CONDITIONS */}
      <section id="conditions" className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-[1140px] mx-auto">
          <div className="text-center max-w-[780px] mx-auto mb-10">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 bg-slate-100 dark:bg-[#1e1e1e] border border-slate-200 dark:border-[#2e2e2e] rounded-full text-slate-600 dark:text-slate-400 text-[0.8125rem] font-semibold mb-3">
              <FiCheckCircle size={14} />
              <span>Diagnostic Scope</span>
            </div>
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              7 Supported Skin Disease Conditions
            </h2>
            <p className="text-[1.05rem] text-slate-600 dark:text-slate-400 leading-[1.65]">
              Classified across 3 clinical risk tiers from high-priority malignant tumors to harmless benign growths.
            </p>
          </div>

          {/* Minimalist Filter Tabs */}
          <div className="flex items-center gap-2 mb-6 flex-wrap">
            <button
              onClick={() => setSelectedFilter('all')}
              className={`px-4 py-2 rounded-full text-[0.825rem] font-semibold transition-colors flex items-center gap-1.5 border ${
                selectedFilter === 'all'
                  ? 'bg-green-600 text-white border-green-600'
                  : 'bg-transparent text-slate-600 dark:text-slate-400 border-slate-200 dark:border-[#262626] hover:bg-slate-100 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              All Conditions (7)
            </button>
            <button
              onClick={() => setSelectedFilter('malignant')}
              className={`px-4 py-2 rounded-full text-[0.825rem] font-semibold transition-colors flex items-center gap-1.5 border ${
                selectedFilter === 'malignant'
                  ? 'bg-green-600 text-white border-green-600'
                  : 'bg-transparent text-slate-600 dark:text-slate-400 border-slate-200 dark:border-[#262626] hover:bg-slate-100 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              <span className="text-red-500">●</span> Malignant (2)
            </button>
            <button
              onClick={() => setSelectedFilter('precancer')}
              className={`px-4 py-2 rounded-full text-[0.825rem] font-semibold transition-colors flex items-center gap-1.5 border ${
                selectedFilter === 'precancer'
                  ? 'bg-green-600 text-white border-green-600'
                  : 'bg-transparent text-slate-600 dark:text-slate-400 border-slate-200 dark:border-[#262626] hover:bg-slate-100 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              <span className="text-amber-500">●</span> Pre-Cancerous (1)
            </button>
            <button
              onClick={() => setSelectedFilter('benign')}
              className={`px-4 py-2 rounded-full text-[0.825rem] font-semibold transition-colors flex items-center gap-1.5 border ${
                selectedFilter === 'benign'
                  ? 'bg-green-600 text-white border-green-600'
                  : 'bg-transparent text-slate-600 dark:text-slate-400 border-slate-200 dark:border-[#262626] hover:bg-slate-100 dark:hover:bg-white/5 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
            >
              <span className="text-green-500">●</span> Benign (4)
            </button>
          </div>

          {/* Condition Rows */}
          <div className="flex flex-col">
            {filteredConditions.map((cond) => (
              <div
                key={cond.code}
                className="grid grid-cols-1 md:grid-cols-[80px_1fr_auto] gap-2 md:gap-6 items-center py-4 border-b border-slate-200 dark:border-[#222222] first:border-t"
              >
                <div className="flex items-center justify-start">
                  <span
                    className="text-[0.775rem] font-extrabold px-2.5 py-1 rounded-full uppercase"
                    style={{
                      backgroundColor: `${cond.riskColor}15`,
                      color: cond.riskColor,
                      border: `1px solid ${cond.riskColor}35`,
                    }}
                  >
                    {cond.code}
                  </span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <h4 className="text-base font-bold m-0 text-slate-900 dark:text-slate-100">{cond.name}</h4>
                  <p className="text-[0.85rem] text-slate-600 dark:text-slate-400 leading-[1.5] m-0">{cond.description}</p>
                </div>
                <div className="flex items-center justify-start md:justify-end">
                  <span
                    className="text-[0.75rem] font-bold px-3 py-1 rounded-full whitespace-nowrap"
                    style={{
                      backgroundColor: `${cond.riskColor}12`,
                      color: cond.riskColor,
                      border: `1px solid ${cond.riskColor}30`,
                    }}
                  >
                    {cond.riskText}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* SECTION 6: PROJECT SCOPE & AI LIMITATIONS */}
      <section id="transparency" className="py-12 sm:py-16 px-4 sm:px-6 bg-white dark:bg-[#121212]">
        <div className="max-w-[1140px] mx-auto">
          <div className="text-center max-w-[780px] mx-auto mb-10">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 bg-slate-100 dark:bg-[#1e1e1e] border border-slate-200 dark:border-[#2e2e2e] rounded-full text-slate-600 dark:text-slate-400 text-[0.8125rem] font-semibold mb-3">
              <FiShield size={14} />
              <span>Project Transparency</span>
            </div>
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold tracking-tight text-slate-900 dark:text-slate-100 mb-3">
              Project Scope & AI Limitations
            </h2>
            <p className="text-[1.05rem] text-slate-600 dark:text-slate-400 leading-[1.65]">
              Technical considerations regarding how this deep learning tool was engineered and how its outputs should be interpreted.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 py-6 border-t border-slate-200 dark:border-[#222222]">
            <div className="flex flex-col gap-2">
              <div className="text-green-600 dark:text-green-500 flex items-center gap-2">
                <FiInfo size={18} />
                <h4 className="text-base font-bold m-0 text-slate-900 dark:text-slate-100">
                  Academic Research Project
                </h4>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-[1.6] m-0">
                This tool is an engineering project developed to evaluate ensemble deep learning on dermatoscopic images.
                It is not an FDA-approved clinical diagnostic device.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="text-amber-500 flex items-center gap-2">
                <FiShield size={18} />
                <h4 className="text-base font-bold m-0 text-slate-900 dark:text-slate-100">
                  Sensitivity to Photo Quality
                </h4>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-[1.6] m-0">
                Deep learning models can make errors, especially on blurry photos, non-standard smartphone lighting,
                hair occlusions, or lesions outside the HAM10000 distribution.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="text-green-600 dark:text-green-500 flex items-center gap-2">
                <FiCheckCircle size={18} />
                <h4 className="text-base font-bold m-0 text-slate-900 dark:text-slate-100">
                  Consult Certified Doctors
                </h4>
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 leading-[1.6] m-0">
                Always consult a certified dermatologist for professional clinical examination, dermoscopy,
                or biopsy confirmation of any concerning skin spot.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 7: LIVE DEMO CTA */}
      <section className="py-12 sm:py-16 px-4 sm:px-6">
        <div className="max-w-[1140px] mx-auto">
          <div className="bg-[#171717] dark:bg-[#181818] border border-neutral-800 dark:border-green-500/30 rounded-2xl px-6 sm:px-8 py-12 text-white text-center flex flex-col items-center">
            <h2 className="text-2xl sm:text-[2.25rem] font-extrabold m-0 mb-2.5 tracking-tight text-white">
              Ready to test the Ensemble Model?
            </h2>
            <p className="text-[1.05rem] text-neutral-400 max-w-[600px] m-0 mb-6 leading-[1.6]">
              Upload any skin lesion photo to view the predicted condition and probability breakdown across all 7 categories.
            </p>
            <Button asChild variant="brand" size="lg">
              <Link to={ctaRoute}>
                {ctaText}
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Minimalist Clean Footer */}
      <footer className="border-t border-slate-200 dark:border-[#1f1f1f] bg-white dark:bg-[#121212] px-4 sm:px-6 pt-10 pb-8">
        <div className="max-w-[1140px] mx-auto flex flex-col sm:flex-row justify-between items-center flex-wrap gap-4 text-center sm:text-left">
          <div>
            <div className="font-bold text-base text-slate-900 dark:text-slate-100">
              Skin Disease Classification Project
            </div>
            <div className="text-[0.825rem] text-slate-500 dark:text-slate-400 mt-1">
              Deep Learning Ensemble (ResNet-101 + DenseNet-121 + EfficientNet-B3) on HAM10000.
            </div>
          </div>

          <div className="flex items-center gap-4 text-[0.825rem] text-slate-500 dark:text-slate-400">
            <span>Educational ML Project</span>
            <span>•</span>
            <a
              href="https://github.com/Purvesh-PJ/skin_disease_detection"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-green-600 dark:text-green-500 font-semibold hover:underline"
            >
              GitHub Repository
              <FiExternalLink size={12} />
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
