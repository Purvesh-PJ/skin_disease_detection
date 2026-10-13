"""
ML Engine Constants & Hyperparameters
-------------------------------------
Single source of truth for model architectures, input dimensions,
and dataset configurations within the ML training engine.
"""

TARGET_IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*TARGET_IMAGE_SIZE, 3)
DEFAULT_BATCH_SIZE = 64
NUM_CLASSES = 7

# Diagnostic Classes (HAM10000)
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_INDICES = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
