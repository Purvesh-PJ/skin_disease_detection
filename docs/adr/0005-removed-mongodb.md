# 0005. Decouple MongoDB and Use Browser Storage for Scan History

## Status
Accepted

## Context
The legacy application coupled machine learning model inference with user authentication (JWT, bcrypt), a MongoDB cluster, and server-side scan history management. On cloud free tiers (Render, Vercel), this created significant friction:
- Reviewers and evaluators faced an unnecessary login/signup wall just to test the AI models.
- An active MongoDB cluster connection introduced cold-start connection timeouts, network latency, and external points of failure.
- Container memory usage on Render Free Tier (<512 MB) was burdened by database connection pools and auth middleware.

## Decision
1. **Decouple Persistence & Auth**: Completely remove MongoDB, user authentication, password hashing, and server-side history routes from the backend.
2. **Stateless AI Inference Microservice**: Restructure the Python backend into a dedicated, low-latency AI inference microservice (<200 MB RAM footprint).
3. **Client-Side History**: Persist diagnostic scan history purely in browser `localStorage`, providing instant zero-latency saving and offline resilience.
4. **Hugging Face Hub Model Pre-caching**: Stream quantized `.tflite` model checkpoints directly from Hugging Face Hub (`purvesh-pj/skin-disease-ensemble`) during Docker build to avoid committing large binaries to Git.
5. **Zero-Friction UI**: Grant instant 1-click access to the Diagnostic Studio (`/dashboard`) directly from the research overview landing page.

## Consequences
- **Positive**: Zero external database dependency; zero database hosting costs.
- **Positive**: Eliminates cold-start database connection timeouts and drops process RAM to ~100-150 MB.
- **Positive**: Zero friction for recruiters, doctors, and evaluators evaluating the diagnostic ensemble.
- **Positive**: Diagnostic scan history persists reliably and instantaneously in browser `localStorage`.
- **Negative**: History is scoped per browser client and does not sync across separate devices.

