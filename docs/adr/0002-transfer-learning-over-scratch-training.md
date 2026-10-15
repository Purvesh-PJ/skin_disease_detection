# 0002. Adopting Transfer Learning over Scratch Neural Network Design

Date: 2026-10-11  
Status: Accepted  

## Context

Designing, optimizing, and training a novel deep convolutional neural network from scratch for dermatoscopic classification requires months of iterative hyperparameter tuning, extensive dataset curation, and massive GPU compute clusters.

With a strict one-month project deadline, solo development responsibility, and limited initial deep learning experience, training a model from random initialization on 10,015 HAM10000 images led to severe overfitting, slow convergence, and poor generalization on underrepresented disease classes.

## Decision

We chose to adopt **Transfer Learning** using proven deep convolutional architectures pre-trained on ImageNet:

1. **Pre-trained Backbones**: Leveraged feature extractors from **ResNet-101**, **DenseNet-121**, and **EfficientNet-B3** to utilize robust low-level visual filters (edges, textures, gradients) learned from millions of images.
2. **Fine-Tuning on HAM10000**: Froze early convolutional blocks and fine-tuned top classification layers on preprocessed $224 \times 224$ dermatoscopic images.
3. **OpenCV Normalization**: Built a dedicated preprocessing pipeline using OpenCV to standardize varied input images into normalized floating-point RGB tensors matching the exact distribution expected by the pre-trained backbones.

## Consequences

### Positive
- **Rapid Convergence**: Achieved high classification accuracy within days rather than months.
- **Overfitting Mitigation**: Pre-trained representations prevented the models from memorizing the limited dataset size.
- **Feasible on Constrained Compute**: Training became computationally viable within standard GPU limits.

### Trade-offs
- Fixed input resolution requirements ($224 \times 224 \times 3$), requiring bilinear/bicubic resizing of high-resolution dermatoscopic scans.
