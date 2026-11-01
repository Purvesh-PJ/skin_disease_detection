# Contributing Guidelines

Thank you for your interest in contributing to the Skin Disease Detection platform.

## Workflow & Branching

We follow GitHub Flow. Create focused branches from `main`:

- `feat/<feature-name>`: New features or UI components
- `fix/<bug-name>`: Bug fixes and patches
- `refactor/<module-name>`: Code cleanup and performance improvements
- `test/<test-scope>`: Test infrastructure, test suites, and CI/CD
- `docs/<topic>`: Documentation updates

## Commit Conventions

Use clear conventional commit messages:

- `feat`: Introduces a new feature
- `fix`: Patches a bug
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating test suites and testing infrastructure
- `docs`: Documentation updates
- `chore`: Dependency updates, tooling, and build tasks

## Local Verification

Ensure local checks pass before submitting a Pull Request:

```bash
# 1. Run all tests (Backend + Frontend) in one command:
npm test

# 2. Run all tests with code coverage reports:
npm run test:cov

# 3. Verify frontend production build:
npm --prefix frontend run build
```

Individual service test runners:
```bash
# Backend tests & 91%+ coverage (pytest):
npm run test:backend
npm run test:backend:cov

# Frontend tests & component coverage (Jest):
npm run test:frontend
npm run test:frontend:cov
```

Open a Pull Request against `main` with a clear summary of changes and test results.
