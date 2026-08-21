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
    confidence: '96',
    fileName: 'sample_mel.jpg',
    imagePath: sampleMel,
    description: 'Melanoma is a serious, aggressive form of skin cancer that can be life-threatening if not detected early. Immediate consultation with a dermatologist and surgical excision are recommended.',
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
    confidence: '98',
    fileName: 'sample_nv.jpg',
    imagePath: sampleNv,
    description: 'Melanocytic nevi (ordinary moles) are benign proliferations of melanocytes. They are typically harmless, though any noticeable change in size, shape, or color should be monitored.',
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
    confidence: '95',
    fileName: 'sample_bcc.jpg',
    imagePath: sampleBcc,
    description: 'Basal cell carcinoma is the most common form of skin cancer. While slow-growing and rarely metastasizing, early dermatological intervention is required to prevent local tissue destruction.',
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
    confidence: '93',
    fileName: 'sample_bkl.jpg',
    imagePath: sampleBkl,
    description: 'Benign keratoses (including seborrheic keratosis and solar lentigo) are non-cancerous skin lesions commonly associated with aging and sun exposure.',
  },
  {
    id: 'akiec',
    code: 'AKIEC',
    name: 'Actinic Keratoses',
    shortName: 'Actinic Keratosis',
    type: 'precancerous',
    typeLabel: 'Pre-Cancerous',
    riskLevel: 'Medium Risk',
    badgeVariant: 'warning',
    confidence: '92',
    fileName: 'sample_akiec.jpg',
    imagePath: sampleAkiec,
    description: 'Actinic keratoses are rough, scaly intraepidermal lesions resulting from cumulative ultraviolet radiation damage. They carry pre-malignant potential to evolve into squamous cell carcinoma.',
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
    confidence: '94',
    fileName: 'sample_df.jpg',
    imagePath: sampleDf,
    description: 'Dermatofibromas are common, benign cutaneous fibrohistiocytic nodules that typically remain stable and rarely require medical intervention.',
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
    confidence: '97',
    fileName: 'sample_vasc.jpg',
    imagePath: sampleVasc,
    description: 'Vascular lesions (such as cherry angiomas or angiokeratomas) are benign vascular anomalies of the dermal blood vessels that are harmless.',
  },
];


