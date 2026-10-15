# 0004. Quantized TensorFlow Lite Flatbuffers & Sequential Inference for 512MB RAM Ceiling

Date: 2026-10-14  
Status: Accepted  

## Context

When deploying the Flask backend to Render's free tier (which enforces a strict **512 MB cgroup memory ceiling**), loading all three trained Keras `.h5` model graphs concurrently consumed over **650 MB RAM**. 

During incoming inference requests, the container ran out of memory, prompting the Linux kernel to terminate the worker process via `SIGKILL 137` and returning `HTTP 502 Bad Gateway` errors to users. Upgrading to a paid instance was ruled out as the system was designed for zero-cost public hosting.

## Decision

We restructured the production inference pipeline for ultra-low memory consumption:

1. **Quantized TFLite Conversion**: Converted all three Keras `.h5` models to 8-bit/float16 quantized TensorFlow Lite (`.tflite`) flatbuffers using `convert_to_tflite.py`, shrinking total weight size from ~400 MB to **64.26 MB** (~84% reduction).
2. **Sequential Forward Passes**: Structured inference to load, execute, and de-allocate model interpreters sequentially rather than concurrently:
   - ResNet-101 forward pass -> `del interpreter` -> `gc.collect()`
   - DenseNet-121 forward pass -> `del interpreter` -> `gc.collect()`
   - EfficientNet-B3 forward pass -> `del interpreter` -> `gc.collect()`
3. **On-Demand Checkpoint Streaming**: Checkpoints are hosted on Hugging Face Hub and streamed on-demand on first container startup to avoid Git repository bloat.

## Consequences

### Positive
- **Drastic RAM Reduction**: Peak inference memory dropped from **~650 MB to ~120 MB**, comfortably beneath Render's 512 MB limit.
- **Zero OOM Crashes**: Completely eliminated `SIGKILL 137` and 502 Bad Gateway errors under load.
- **Fast Cold Starts**: Flatbuffers map memory directly without building heavy computational graphs.

### Trade-offs
- Sequential execution adds ~150–250 ms latency compared to parallel GPU execution, which is an acceptable trade-off for an asynchronous clinical evaluation platform.
