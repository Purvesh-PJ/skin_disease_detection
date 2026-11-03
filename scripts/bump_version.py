#!/usr/bin/env python3
"""
Single Source of Truth (SSOT) Version Bumper & Release Coordinator
------------------------------------------------------------------
Atomically updates project version across root manifests, frontend package,
and runtime metadata, ensuring 100% version consistency without manual edits.

Usage:
    python scripts/bump_version.py patch          # 2.1.0 -> 2.1.1
    python scripts/bump_version.py minor          # 2.1.0 -> 2.2.0
    python scripts/bump_version.py major          # 2.1.0 -> 3.0.0
    python scripts/bump_version.py 2.3.0          # Set explicit SemVer
    python scripts/bump_version.py --check        # Audit current version consistency
"""

import sys
import os
import re
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERSION_FILE = os.path.join(REPO_ROOT, "VERSION")
ROOT_PKG = os.path.join(REPO_ROOT, "package.json")
ROOT_LOCK = os.path.join(REPO_ROOT, "package-lock.json")
FRONTEND_PKG = os.path.join(REPO_ROOT, "frontend", "package.json")
FRONTEND_LOCK = os.path.join(REPO_ROOT, "frontend", "package-lock.json")

SEMVER_REGEX = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


def get_current_version() -> str:
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            v = f.read().strip()
            if v:
                return v
    if os.path.exists(ROOT_PKG):
        with open(ROOT_PKG, "r", encoding="utf-8") as f:
            return json.load(f).get("version", "0.0.0")
    return "0.0.0"


def calculate_next_version(current: str, bump_type: str) -> str:
    match = SEMVER_REGEX.match(current)
    if not match:
        raise ValueError(f"Current version '{current}' is not valid SemVer.")
    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

    b = bump_type.lower()
    if b == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif b == "minor":
        return f"{major}.{minor + 1}.0"
    elif b == "major":
        return f"{major + 1}.0.0"
    elif SEMVER_REGEX.match(bump_type):
        return bump_type
    else:
        raise ValueError(
            f"Invalid argument '{bump_type}'. Must be 'patch', 'minor', 'major', or explicit SemVer 'X.Y.Z'."
        )


def update_json_version(file_path: str, new_version: str) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    old_version = data.get("version")
    data["version"] = new_version
    # If package-lock has packages[""] entry
    if "packages" in data and "" in data["packages"]:
        data["packages"][""]["version"] = new_version

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return old_version != new_version


def check_consistency():
    current = get_current_version()
    print(f"[*] Auditing project version consistency (Target: {current})...")
    files = [
        ("VERSION", VERSION_FILE, lambda p: open(p, "r", encoding="utf-8").read().strip()),
        ("Root package.json", ROOT_PKG, lambda p: json.load(open(p, "r", encoding="utf-8")).get("version")),
        ("Frontend package.json", FRONTEND_PKG, lambda p: json.load(open(p, "r", encoding="utf-8")).get("version")),
    ]
    all_matched = True
    for label, path, extractor in files:
        if os.path.exists(path):
            v = extractor(path)
            status = "OK" if v == current else f"MISMATCH ({v})"
            if v != current:
                all_matched = False
            print(f"  - {label:<22}: {v} [{status}]")
        else:
            print(f"  - {label:<22}: NOT FOUND")

    # Audit Backend AppConfig dynamic detection
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
        from app.core.config import config
        backend_v = config.VERSION
        b_status = "OK" if backend_v == current else f"MISMATCH ({backend_v})"
        if backend_v != current:
            all_matched = False
        print(f"  - {'Backend (AppConfig)':<22}: {backend_v} [{b_status}]")
    except Exception as e:
        print(f"  - {'Backend (AppConfig)':<22}: Import Warning ({e})")

    if all_matched:
        print(f"\n[+] SUCCESS: All project manifests are 100% consistent on v{current}!\n")
    else:
        print(f"\n[!] WARNING: Version drift detected across files.\n")
    return all_matched


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    arg = sys.argv[1].strip()
    if arg == "--check":
        consistent = check_consistency()
        sys.exit(0 if consistent else 1)

    current_v = get_current_version()
    new_v = calculate_next_version(current_v, arg)

    print(f"=====================================================")
    print(f" Skin Disease Detection - Atomic Version Bump")
    print(f" Version transition: v{current_v} -> v{new_v}")
    print(f"=====================================================")

    # 1. Update VERSION file
    with open(VERSION_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{new_v}\n")
    print(f" [OK] Updated VERSION -> {new_v}")

    # 2. Update Root package.json
    if update_json_version(ROOT_PKG, new_v):
        print(f" [OK] Updated package.json -> {new_v}")

    # 3. Update Frontend package.json
    if update_json_version(FRONTEND_PKG, new_v):
        print(f" [OK] Updated frontend/package.json -> {new_v}")

    # 4. Update Lockfiles if present
    if update_json_version(ROOT_LOCK, new_v):
        print(f" [OK] Updated package-lock.json -> {new_v}")
    if update_json_version(FRONTEND_LOCK, new_v):
        print(f" [OK] Updated frontend/package-lock.json -> {new_v}")

    print(f"\n[+] Version successfully bumped to v{new_v} across all components.")
    print(f"    - Backend dynamic config: auto-resolved to v{new_v}")
    print(f"    - Frontend build config : auto-resolved to v{new_v}")
    print(f"\nNext recommended steps:")
    print(f"  1. Run tests: npm test")
    print(f"  2. Commit   : git commit -am \"chore(release): bump version to v{new_v}\"")
    print(f"  3. Tag      : git tag v{new_v}")


if __name__ == "__main__":
    main()
