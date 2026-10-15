# 0001. Architecture Shift from Desktop Executable to Web Platform (React + Flask)

Date: 2026-10-10  
Status: Accepted  

## Context

The initial project plan was to build a local Python desktop application distributed as a standalone Windows executable (`.exe`). However, several practical constraints made this approach unfeasible:

1. **Poor UI Frameworks**: Python desktop GUI toolkits (Tkinter, PyQt) produced outdated, clunky interfaces with limited responsiveness and poor visualization for image analysis and confidence charts.
2. **Setup Friction for Evaluators & Recruiters**: Running a local executable required end-users, college evaluators, and recruiters to download large binaries, deal with Windows SmartScreen / antivirus false positives, and configure local environments.
3. **Visibility & Accessibility**: A local executable could not be easily demonstrated live via a browser URL or linked directly from a GitHub repository and resume.

## Decision

We pivoted the entire architecture from a local desktop executable to a decoupled, full-stack web platform:

- **Frontend**: Built with **React 18** to deliver a responsive clinical dashboard, real-time diagnostic visualizers, and zero-install browser access for any evaluator.
- **Backend**: Built with **Python Flask** to serve a lightweight REST API that seamlessly bridges web requests with Python's deep learning and computer vision ecosystem (TensorFlow, OpenCV).

## Consequences

### Positive
- **Instant Global Accessibility**: Anyone (evaluators, recruiters, peer developers) can access the live application via URL with zero setup or download steps.
- **Modern User Experience**: Enabled rich interactive features like drag-and-drop uploads, instant HAM10000 sample pickers, and scan history audit logs.
- **Clean Separation of Concerns**: Decoupled presentation logic (React on Vercel) from compute-intensive ML inference (Flask on Render).

### Trade-offs
- Requires managing cloud deployments, CORS configurations, and cross-origin authentication (JWT) instead of simple local function calls.
