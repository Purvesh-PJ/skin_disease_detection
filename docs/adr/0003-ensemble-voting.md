# 0003. Multi-Model Soft-Voting Ensemble

## Status
Accepted

## Context
Individual CNN models showed class-specific biases and overconfidence on ambiguous lesion borders during testing, creating high diagnostic risk for single-model predictions.

## Decision
Combine all three architectures using **Soft-Voting Ensemble**, averaging their 7-class softmax probability vectors to produce a consensus diagnosis.

## Consequences
- **Positive**: Lowers diagnostic variance and suppresses single-model false positives.
- **Positive**: Provides a calibrated confidence score.
- **Negative**: Inference requires three forward passes per image instead of one.
