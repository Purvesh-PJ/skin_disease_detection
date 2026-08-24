# 🔒 Security Policy

The **Skin Disease Detection Platform** team takes security vulnerabilities seriously. We appreciate the responsible disclosure of any potential security risks or data exposure vulnerabilities.

---

## 📋 Supported Versions

We actively maintain and provide security patches for the following versions of the platform:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| `1.0.x` | ✅ Yes | Current production release |
| `< 1.0.0` | ❌ No | Deprecated development prototypes |

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability (e.g. JWT token leakage, unauthorized MongoDB queries, input sanitization bypass, or dependency CVE), please follow these guidelines:

1. **Do NOT open a public GitHub issue** for sensitive vulnerabilities.
2. Send an email directly to the project maintainer with the subject `[SECURITY ISSUE] Skin Disease Detection Platform`:
   * 📧 **Security Contact**: [purveshpatel2005@gmail.com](mailto:purveshpatel2005@gmail.com)
3. Please include the following details in your report:
   * A detailed description of the vulnerability.
   * Step-by-step reproduction steps or proof-of-concept (PoC) payload.
   * Potential impact on user data, prediction models, or infrastructure.
   * Any suggested remediations or mitigations.

---

## ⏱️ Response & Disclosure Timeline

* **Initial Acknowledgment**: Within **48 hours** of receiving your report.
* **Triage & Assessment**: Within **5 business days**, confirming whether the vulnerability is valid.
* **Patch Release**: A security patch will be prepared, tested, and deployed promptly.
* **Public Credit**: Once resolved, we will gladly acknowledge your contribution in our [CHANGELOG.md](CHANGELOG.md) (unless you prefer to remain anonymous).

---

## 🛡️ Best Security Practices for Deployments

* Always set strong, cryptographically random values for `FLASK_SECRET_KEY` and `JWT_SECRET_KEY` in production.
* Never commit real `.env` files with database passwords or cloud secrets to public source control.
* In public healthcare deployments, ensure communication is strictly enforced over HTTPS/TLS.
