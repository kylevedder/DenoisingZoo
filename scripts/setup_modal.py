#!/usr/bin/env python3
"""
Setup script for Modal resources.
This script helps set up the required secrets and volumes for Modal training.
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    import modal
except ImportError:
    print("Error: modal is not installed. Run: uv sync")
    sys.exit(1)


def check_modal_auth():
    """Check if Modal authentication is set up."""
    if not os.getenv("MODAL_TOKEN_ID") and not os.getenv("MODAL_TOKEN_SECRET"):
        print("Modal authentication not found.")
        print("Please run: modal token new")
        return False
    return True


def create_secret():
    """Create the training secrets."""
    print("Creating Modal secret: modal-training-secrets")
    
    # Create a simple secret for now - users can add their own secrets later
    secret_data = {
        "EXAMPLE_API_KEY": "your_api_key_here",
        "EXAMPLE_SECRET": "your_secret_here"
    }
    
    try:
        secret = modal.Secret.from_dict(secret_data)
        secret.put("modal-training-secrets")
        print("✓ Secret 'modal-training-secrets' created successfully")
        print("  You can add your own secrets by editing this secret in the Modal dashboard")
        return True
    except Exception as e:
        print(f"✗ Failed to create secret: {e}")
        return False


def create_volume():
    """Create the training data volume."""
    print("Creating Modal volume: training-data")
    
    try:
        volume = modal.Volume.create("training-data")
        print("✓ Volume 'training-data' created successfully")
        print("  This volume will persist training checkpoints and outputs")
        return True
    except Exception as e:
        print(f"✗ Failed to create volume: {e}")
        return False


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
        print("To customize secrets, visit the Modal dashboard and edit the 'modal-training-secrets' secret.")
    else:
        print("✗ Some resources failed to create. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()