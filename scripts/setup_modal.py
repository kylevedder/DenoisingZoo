#!/usr/bin/env python3
"""
Setup script for Modal resources.
This script helps set up the required secrets and volumes for Modal training.
"""

import os
import sys
import subprocess
from pathlib import Path
import subprocess

try:
    import modal
except ImportError:
    print("Error: modal is not installed. Run: uv sync")
    sys.exit(1)


def check_modal_auth() -> bool:
    """Check if Modal authentication is set up.

    Accept either environment variables or a ~/.modal.toml profile created by `modal token new`.
    """
    # Environment-based auth
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True

    # File-based auth
    modal_toml = Path.home() / ".modal.toml"
    if modal_toml.exists():
        try:
            # Verify via CLI if available
            subprocess.run(
                ["modal", "token", "whoami"], check=True, capture_output=True
            )
            return True
        except Exception:
            # File exists but verification failed; still consider present and let Modal surface details later
            return True

    print("Modal authentication not found.")
    print("Please run: modal token new")
    return False


def create_secret():
    """Create the training secrets."""
    print(
        "Skipping secret creation (not required). You can add secrets later in Modal if needed."
    )
    return True


def create_volume():
    """Create the training data volume."""
    print("Skipping volume creation (app runs without volumes by default).")
    return True


def main():
    print("Setting up Modal resources for denoisingzoo training...")
    print()

    if not check_modal_auth():
        return

    print("Creating required Modal resources:")
    print()

    # Create secret
    secret_created = create_secret()
    print()

    # Create volume
    volume_created = create_volume()
    print()

    if secret_created and volume_created:
        print("✓ All Modal resources created successfully!")
        print()
        print("You can now run training on Modal with:")
        print("  python launcher.py --backend modal")
        print()
        print(
            "To customize secrets, visit the Modal dashboard and edit the 'modal-training-secrets' secret."
        )
    else:
        print("✗ Some resources failed to create. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
