# 0006. Full-Stack Automated Testing Infrastructure, Framework Selection, and CI/CD Quality Gate

## Status
Accepted

## Context
Prior to this architectural phase, the application relied on manual ad-hoc testing and an obsolete, non-executable legacy test script (`backend/app/tests/test_prediction.py`). In a polyglot AI architecture comprising a Python Flask ML microservice and a React 18 single-page application:
- Changes to core computer vision logic, model downloading, or ensemble voting carried high regression risks.
- Frontend component rendering, user interactions, and `localStorage` scan history persistence lacked verification.
- Developers and contributors had to remember fragmented, OS-dependent terminal commands across separate directories.
- Pull requests lacked automated quality gates to prevent broken builds or untested code from merging into `main`.

## Decision

### 1. Backend Testing Framework: `pytest` over standard library `unittest`
We selected **`pytest`** along with `pytest-cov` and `pytest-mock` rather than Python's standard library `unittest`:
- **Session & Fixture Architecture**: `conftest.py` provides session-scoped Flask test clients and in-memory mock image fixtures without repetitive boilerplate classes.
- **Native Exception & Warning Assertions**: Intuitive `pytest.raises` for custom domain exceptions (e.g., `ModelNotFoundError` 530, `InvalidImageError` 400).
- **Comprehensive Coverage Measurement**: Direct integration with `pytest-cov` to enforce statement and branch coverage metrics (achieved **91% backend coverage**).
- **Separation of Dev Dependencies**: Placed strictly in `backend/requirements-dev.txt` to keep the production container image lightweight and secure.

### 2. Frontend Testing Framework: `Jest` + `React Testing Library (RTL)`
We selected **`Jest`** paired with **`@testing-library/react`** and `@testing-library/jest-dom`:
- **User-Centric Behavioral Testing**: Tests simulate actual clinical user actions (uploading files, selecting lesion samples, clearing history) rather than asserting internal React state.
- **Subpath Resolution & Radix UI Mocking**: Handled Radix UI ESM subpath resolution through targeted test setups in `src/setupTests.js` without altering production bundles.
- **Built-in Istanbul Code Coverage**: Zero-configuration frontend coverage reporting via `npm run test:cov`.

### 3. Unified Cross-Platform Orchestrator (`npm test`)
To eliminate developer friction and OS-specific path issues (Windows backslashes vs POSIX paths, virtual environment detection):
- Created a root-level `package.json` with universal commands (`npm test`, `npm run test:cov`, `npm run test:backend`, `npm run test:frontend`).
- Implemented `scripts/run_backend_tests.py` that dynamically resolves virtual environment binaries across Windows (`backend/venv/Scripts/python.exe`) and Linux/macOS (`backend/venv/bin/python`), injecting `PYTHONPATH=backend` automatically.

### 4. Automated CI/CD Quality Gate (GitHub Actions)
Implemented `.github/workflows/ci.yml` running parallel verification jobs on every push and pull request against `main`:
- **Backend Job**: Sets up Python 3.11, installs system OpenCV dependencies (`libgl1-mesa-glx`), runs unit/integration tests, and enforces an 80% coverage threshold (`--cov-fail-under=80`).
- **Frontend Job**: Sets up Node.js 18, performs clean `npm ci`, runs tests with coverage, and validates the production build (`npm run build`).

## Consequences
- **Positive**: High architectural confidence across 63 automated tests (30 backend, 33 frontend) with 0 failures.
- **Positive**: 91% backend code coverage protecting inference, preprocessing, and error boundaries.
- **Positive**: 1-click execution (`npm test` / `npm run test:cov`) from the root directory across all operating systems.
- **Positive**: Pull requests are automatically validated by GitHub Actions before merge.
- **Neutral**: Added dev testing packages (`pytest`, `pytest-cov`, `pytest-mock`) maintained separately in `backend/requirements-dev.txt`.
