#!/usr/bin/env python3
"""
Docker Build & Status Inspector for Skin Disease Platform
---------------------------------------------------------
Provides transparent feedback during container builds and generates a
clean status summary table with image sizes, ports, and operational states.

Usage:
    # Build all services (frontend, backend, ml-engine) with live visible progress:
    python scripts/docker_manager.py build

    # Build a specific target service:
    python scripts/docker_manager.py build --service frontend
    python scripts/docker_manager.py build --service backend
    python scripts/docker_manager.py build --service ml-engine

    # Inspect current status of all project images and containers:
    python scripts/docker_manager.py status
"""

import sys
import subprocess
import argparse
import shutil

SERVICES_META = {
    "frontend": {
        "display_name": "Frontend (React UI)",
        "image_tag": "skin_disease_frontend:latest",
        "port": "http://localhost:3000",
        "role": "Web UI Tier"
    },
    "backend": {
        "display_name": "Backend (Flask API)",
        "image_tag": "skin_disease_backend:latest",
        "port": "http://localhost:5000",
        "role": "Inference Microservice"
    },
    "ml-engine": {
        "display_name": "ML Engine (TensorFlow)",
        "image_tag": "skin_disease_ml_engine:latest",
        "port": "CLI / Interactive (No HTTP)",
        "role": "Training & Serialization"
    }
}


def print_banner(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def check_docker_available():
    if not shutil.which("docker"):
        print("[ERROR] Docker is not installed or not in system PATH.")
        sys.exit(1)


def get_image_size(image_name):
    """Fetches human-readable image size using docker image inspect."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image_name, "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode == 0 and proc.stdout.strip():
            size_bytes = int(proc.stdout.strip())
            if size_bytes >= 1024 ** 3:
                return f"{size_bytes / (1024 ** 3):.2f} GB"
            else:
                return f"{size_bytes / (1024 ** 2):.1f} MB"
    except Exception:
        pass
    return "Not Built"


def display_status_table():
    print_banner("[STATUS] Docker Images & Services Summary")
    header = f"{'Service':<12} | {'Role':<24} | {'Image Size':<12} | {'Access / Endpoint':<24}"
    print(header)
    print("-" * 75)

    for svc_key, meta in SERVICES_META.items():
        size = get_image_size(meta["image_tag"])
        print(f"{svc_key:<12} | {meta['role']:<24} | {size:<12} | {meta['port']:<24}")

    print("=" * 75)
    print("Workflow Shortcuts:")
    print("  * Start Web App (Frontend + Backend): docker compose up -d backend frontend")
    print("  * Start All Services:                docker compose up -d")
    print("  * Inspect ML Setup & Dataset:        docker compose run --rm ml-engine python ml_engine/scripts/setup_ml.py")
    print("  * Run Model Training:                docker compose run --rm ml-engine python ml_engine/run_pipeline.py --help")
    print("  * Stop All Containers:               docker compose down\n")


def build_services(target_service=None, plain_output=True):
    check_docker_available()

    if target_service and target_service in SERVICES_META:
        targets = [target_service]
    else:
        targets = list(SERVICES_META.keys())

    print_banner(f"[BUILD] Starting Docker Build for: {', '.join(targets)}")
    for t in targets:
        meta = SERVICES_META[t]
        print(f"  -> Target: {meta['display_name']} ({meta['role']})")

    print("-" * 75)
    print("ℹ️ Note: Live streaming output enabled. You will see real-time download & build progress.\n")

    cmd = ["docker", "compose", "build"]
    if plain_output:
        cmd.append("--progress=plain")
    if target_service:
        cmd.append(target_service)

    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Docker build failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

    print("\n[SUCCESS] Build process completed successfully!")
    display_status_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker Build & Status Manager")
    subparsers = parser.add_subparsers(dest="action", required=True)

    build_parser = subparsers.add_parser("build", help="Build Docker services and display status table")
    build_parser.add_argument(
        "--service",
        type=str,
        choices=list(SERVICES_META.keys()),
        default=None,
        help="Specify single service to build (frontend, backend, ml-engine). Default: all"
    )

    status_parser = subparsers.add_parser("status", help="Inspect current status and size of all images")

    args = parser.parse_args()

    if args.action == "build":
        build_services(args.service)
    elif args.action == "status":
        display_status_table()
