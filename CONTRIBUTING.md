# Contributing Guidelines

Thank you for your interest in contributing to the Skin Disease Detection platform.

## Workflow & Branching

We follow GitHub Flow. Create focused branches from `main`:

- `feat/<feature-name>`: New features or UI components
- `fix/<bug-name>`: Bug fixes and patches
- `refactor/<module-name>`: Code cleanup and performance improvements
- `docs/<topic>`: Documentation updates

## Commit Conventions

Use clear conventional commit messages:

- `feat`: Introduces a new feature
- `fix`: Patches a bug
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `docs`: Documentation updates
- `chore`: Dependency updates, tooling, and build tasks

## Local Verification

Ensure local checks pass before submitting a Pull Request:

```bash
# 1. Run backend unit tests
cd backend
python -m unittest discover -s app/tests

# 2. Verify frontend build
cd ../frontend
npm run build
```

Open a Pull Request against `main` with a clear summary of changes and test results.
