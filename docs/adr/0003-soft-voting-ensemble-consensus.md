# 0003. Multi-Model Soft-Voting Ensemble over Single-Model Prediction

Date: 2026-10-12  
Status: Accepted  

## Context

During validation, individual neural network architectures exhibited noticeable variance and class-specific blind spots:
- ResNet-101 performed well on distinct textured lesions but misclassified border-ambiguous melanoma cases.
- DenseNet-121 captured subtle color transitions but occasionally produced overconfident predictions on common benign nevi.
- EfficientNet-B3 excelled on scaled features but struggled with minority classes like vascular lesions.

Relying on a single model in a medical diagnostic context carries high risk due to false-positive or false-negative outliers on ambiguous edge cases.

## Decision

We designed a **Three-Model Soft-Voting Ensemble** combining predictions from ResNet-101, DenseNet-121, and EfficientNet-B3:

1. Rather than hard majority voting (which ignores probability distributions), the engine extracts full 7-class softmax probability vectors from each network.
2. The consensus engine averages the probabilities across all three architectures:
   $$\bar{P}(c) = \frac{1}{3} \sum_{m=1}^{3} P_m(c)$$
3. The condition with the highest average score becomes the consensus prediction, effectively penalizing single-model outliers.

## Consequences

### Positive
- **Reduced Diagnostic Variance**: Outlier predictions from one model are stabilized and overridden by the remaining two networks.
- **Calibrated Confidence**: Confidence scores reflect consensus agreement rather than a single model's potentially overconfident output.
- **Transparency**: The API returns individual model contributions alongside the consensus verdict.

### Trade-offs
- Triples the forward-pass computation requirement per image compared to running a single neural network.
