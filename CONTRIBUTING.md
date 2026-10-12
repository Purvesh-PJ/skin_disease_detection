# Contributing Guide

Guidelines for contributing to the Skin Disease Detection project.

---

## 🛠️ Branching Strategy

We follow a standard GitHub Flow model:
* `feature/<feature-name>`: New features
* `fix/<bug-name>`: Bug fixes
* `refactor/<module-name>`: Architectural and code restructuring
* `docs/<topic>`: Documentation updates

---

## 📝 Commit Convention

Follow conventional commit formats:
* `feat`: New feature
* `fix`: Bug fix
* `refactor`: Code change that neither fixes a bug nor adds a feature
* `docs`: Documentation changes
* `chore`: Build tasks, dependency updates, configs

---

## 🧪 Verification

Before submitting changes, make sure local checks pass:
```bash
# Backend tests
cd backend && python -m unittest discover -s app/tests

# Frontend production build
cd ../frontend && npm run build
```
