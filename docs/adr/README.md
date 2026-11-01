# Architecture Decision Records (ADRs)

Key architectural decisions made during development based on real project constraints:

| ADR | Title | Decision | Status |
| :---: | :--- | :--- | :---: |
| [0001](0001-web-platform.md) | Web Platform over Desktop App | React + Flask instead of Tkinter `.exe` | Accepted |
| [0002](0002-transfer-learning.md) | Transfer Learning over Custom CNN | Pre-trained ImageNet backbones on HAM10000 | Accepted |
| [0003](0003-ensemble-voting.md) | Multi-Model Soft-Voting Ensemble | Average probability vectors across 3 models | Accepted |
| [0004](0004-tflite-low-memory.md) | TensorFlow Lite & Sequential Inference | TFLite flatbuffers + sequential execution for 512MB RAM | Accepted |
| [0005](0005-removed-mongodb.md) | Decouple MongoDB & Use Browser Storage | Remove MongoDB/auth, localStorage history, Hugging Face caching | Accepted |
| [0006](0006-testing-infrastructure-and-ci-cd.md) | Testing Infrastructure & CI/CD Gate | pytest + Jest, unified `npm test`, GitHub Actions CI pipeline | Accepted |
