# 0002. Transfer Learning over Custom CNN

## Status
Accepted

## Context
Designing and training a deep CNN from scratch within a 1-month college project deadline resulted in slow convergence, high overfitting, and poor generalization on minority disease classes.

## Decision
Use **Transfer Learning** with ImageNet-pretrained backbones (ResNet-101, DenseNet-121, EfficientNet-B3) and fine-tune their top layers on HAM10000 images.

## Consequences
- **Positive**: Fast convergence and high accuracy without weeks of training.
- **Positive**: Pre-trained low-level feature extractors prevent overfitting.
- **Negative**: Requires resizing and preprocessing input images to match model input dimensions ($224 \times 224$).
