import sampleMel from '../assets/samples/sample_mel.jpg';
import sampleNv from '../assets/samples/sample_nv.jpg';
import sampleBcc from '../assets/samples/sample_bcc.jpg';
import sampleBkl from '../assets/samples/sample_bkl.jpg';
import sampleAkiec from '../assets/samples/sample_akiec.jpg';
import sampleDf from '../assets/samples/sample_df.jpg';
import sampleVasc from '../assets/samples/sample_vasc.jpg';

/**
 * Curated sample dermoscopy images representing all 7 ISIC/HAM10000 disease classes.
 * Imported directly so Webpack bundles them safely without relying on static URL rewrites.
 */
export const SAMPLE_IMAGES = [
  {
    id: 'mel',
    code: 'MEL',
    name: 'Melanoma',
    shortName: 'Melanoma',
    type: 'malignant',
    typeLabel: 'Malignant',
    riskLevel: 'High Risk',
    badgeVariant: 'danger',
    fileName: 'sample_mel.jpg',
    imagePath: sampleMel,
    description: 'Aggressive malignant skin tumor with atypical pigment network.',
  },
  {
    id: 'nv',
    code: 'NV',
    name: 'Melanocytic Nevus',
    shortName: 'Nevus / Mole',
    type: 'benign',
    typeLabel: 'Benign',
    riskLevel: 'Low Risk',
    badgeVariant: 'success',
    fileName: 'sample_nv.jpg',
    imagePath: sampleNv,
    description: 'Common benign melanocytic proliferation (ordinary mole).',
  },
  {
    id: 'bcc',
    code: 'BCC',
    name: 'Basal Cell Carcinoma',
    shortName: 'Basal Cell Ca.',
    type: 'malignant',
    typeLabel: 'Malignant',
    riskLevel: 'High Risk',
    badgeVariant: 'danger',
    fileName: 'sample_bcc.jpg',
    imagePath: sampleBcc,
    description: 'Most common non-melanoma skin cancer with telangiectasias.',
  },
  {
    id: 'bkl',
    code: 'BKL',
    name: 'Benign Keratosis',
    shortName: 'Benign Keratosis',
    type: 'benign',
    typeLabel: 'Benign',
    riskLevel: 'Low Risk',
    badgeVariant: 'success',
    fileName: 'sample_bkl.jpg',
    imagePath: sampleBkl,
    description: 'Seborrheic keratosis / solar lentigo with pseudonetwork.',
  },
  {
    id: 'akiec',
    code: 'AKIEC',
    name: 'Actinic Keratosis',
    shortName: 'Actinic Keratosis',
    type: 'precancerous',
    typeLabel: 'Pre-Cancerous',
    riskLevel: 'Medium Risk',
    badgeVariant: 'warning',
    fileName: 'sample_akiec.jpg',
    imagePath: sampleAkiec,
    description: 'Pre-malignant sun-induced intraepithelial carcinoma.',
  },
  {
    id: 'df',
    code: 'DF',
    name: 'Dermatofibroma',
    shortName: 'Dermatofibroma',
    type: 'benign',
    typeLabel: 'Benign',
    riskLevel: 'Low Risk',
    badgeVariant: 'success',
    fileName: 'sample_df.jpg',
    imagePath: sampleDf,
    description: 'Benign cutaneous fibrous histiocytoma with central white patch.',
  },
  {
    id: 'vasc',
    code: 'VASC',
    name: 'Vascular Lesion',
    shortName: 'Vascular Lesion',
    type: 'benign',
    typeLabel: 'Benign',
    riskLevel: 'Low Risk',
    badgeVariant: 'success',
    fileName: 'sample_vasc.jpg',
    imagePath: sampleVasc,
    description: 'Benign vascular proliferation (cherry angioma / angiokeratoma).',
  },
];

