# 0001. Web Platform over Desktop App

## Status
Accepted

## Context
Initial plan was a Python desktop app (`.exe`). However, Tkinter/PyQt UI was outdated, and distributing an executable required manual environment setup and handling antivirus warnings on evaluators' machines.

## Decision
Build a web application using **React 18** for the UI and **Flask** for the Python backend.

## Consequences
- **Positive**: Zero setup for evaluators; accessible via browser link.
- **Positive**: Clean, modern clinical dashboard UI.
- **Negative**: Required setting up cloud hosting and CORS configuration.
