# 0004. TensorFlow Lite & Sequential Inference

## Status
Accepted

## Context
Deploying three full Keras `.h5` models consumed ~650 MB RAM, exceeding Render free tier's 512 MB memory limit and causing `SIGKILL 137` OOM crashes.

## Decision
Convert all models to quantized **TensorFlow Lite (`.tflite`)** flatbuffers and run inference sequentially with explicit garbage collection (`gc.collect()`).

## Consequences
- **Positive**: Peak RAM dropped from ~650 MB to ~120 MB (zero OOM crashes).
- **Positive**: Total model checkpoint size reduced from ~400 MB to ~64 MB (~84% reduction).
- **Negative**: Sequential execution adds ~150-200ms latency compared to parallel GPU execution.
