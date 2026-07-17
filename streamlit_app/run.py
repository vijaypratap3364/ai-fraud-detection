#!/usr/bin/env python
"""
Entry point for Streamlit Community Cloud deployment.
Handles data download before launching the Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

# Ensure we're in the right directory
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Check if data exists
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

REQUIRED_FILES = [
    DATA_DIR / "synthetic" / "customers_enriched.csv",
    DATA_DIR / "synthetic" / "transaction_history_demo.csv",
    DATA_DIR / "synthetic" / "known_fraud_cases.json",
    MODELS_DIR / "detection" / "xgboost_model.pkl",
    MODELS_DIR / "detection" / "scaler.pkl",
    CHROMA_DIR / "chroma.sqlite3",
]

def check_data():
    """Check if all required data files exist."""
    missing = [f for f in REQUIRED_FILES if not f.exists()]
    return len(missing) == 0, missing

def download_data():
    """Run the download script."""
    print("🔄 Downloading required data files...")
    download_script = APP_DIR / "download_data.py"
    if download_script.exists():
        result = subprocess.run([sys.executable, str(download_script)], capture_output=False)
        return result.returncode == 0
    return False

def main():
    print("=" * 60)
    print("AI Fraud Detection - Streamlit Cloud Startup")
    print("=" * 60)

    # Check for data
    has_data, missing = check_data()

    if not has_data:
        print(f"📦 Missing {len(missing)} required files:")
        for f in missing:
            print(f"  - {f.relative_to(PROJECT_ROOT)}")

        if not download_data():
            print("❌ Data download failed. Check logs above.")
            sys.exit(1)

        # Verify again
        has_data, missing = check_data()
        if not has_data:
            print("❌ Data still missing after download:")
            for f in missing:
                print(f"  - {f.relative_to(PROJECT_ROOT)}")
            sys.exit(1)

    print("✅ All data files present")
    print("🚀 Starting Streamlit...")

    # Launch Streamlit
    app_path = APP_DIR / "app.py"
    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app_path)])

if __name__ == "__main__":
    import os
    main()