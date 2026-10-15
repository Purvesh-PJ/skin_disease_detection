# 0001. Web Platform over Desktop App

## Status
Accepted

## Context
The project was originally planned as a Python desktop application (executable / terminal script). However:
- A terminal or basic desktop GUI (Tkinter) made presenting to non-technical evaluators difficult, as it failed to clearly visualize lesion images, multi-class predictions, and confidence metrics.
- Distributing an executable (`.exe`) required end-users and recruiters to manually configure Python dependencies or run unverified binaries with antivirus warnings.
- The project needed instant public visibility on GitHub and resumes without requiring local installation.

## Decision
Build a web application using **React 18** for the UI and **Flask** for the Python backend.

## Consequences
- **Positive**: Zero setup for evaluators; accessible via browser link.
- **Positive**: Clean, modern clinical dashboard UI.
- **Negative**: Required setting up cloud hosting and CORS configuration.
