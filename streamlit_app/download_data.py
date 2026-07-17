"""
Data download script for Streamlit Community Cloud deployment.
Downloads required data files from GitHub Releases at startup.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# GitHub repo for releases
REPO = "vijaypratap3364/ai-fraud-detection"
RELEASE_TAG = "data-v1"  # Create this release with data files

# Files to download (uploaded_name, local_path, expected_size_mb)
FILES = [
    ("data_synthetic_customers_enriched.csv", "data/synthetic/customers_enriched.csv", 4),
    ("data_synthetic_transaction_history_demo.csv", "data/synthetic/transaction_history_demo.csv", 6),
    ("data_synthetic_known_fraud_cases.json", "data/synthetic/known_fraud_cases.json", 0.2),
    ("models_detection_xgboost_model.pkl", "models/detection/xgboost_model.pkl", 335),
    ("models_detection_scaler.pkl", "models/detection/scaler.pkl", 1),
    ("chroma_db_chrome_db_chroma.sqlite3", "chroma_db/chroma.sqlite3", 1.7),
    ("chroma_db_c7d34386-454a-469e-a124-6b8f156f1162_data_level0.bin", "chroma_db/c7d34386-454a-469e-a124-6b8f156f1162/data_level0.bin", 0.16),
    ("chroma_db_c7d34386-454a-469e-a124-6b8f156f1162_header.bin", "chroma_db/c7d34386-454a-469e-a124-6b8f156f1162/header.bin", 0.001),
    ("chroma_db_c7d34386-454a-469e-a124-6b8f156f1162_length.bin", "chroma_db/c7d34386-454a-469e-a124-6b8f156f1162/length.bin", 0.001),
    ("chroma_db_c7d34386-454a-469e-a124-6b8f156f1162_link_lists.bin", "chroma_db/c7d34386-454a-469e-a124-6b8f156f1162/link_lists.bin", 0.001),
]

# Chroma DB files (handled above in FILES list)
CHROMA_FILES = []

def download_file(url, dest_path, expected_mb=None):
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"✓ {dest_path.name} already exists, skipping")
        return True

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        if expected_mb and total_size > 0:
            actual_mb = total_size / (1024 * 1024)
            if abs(actual_mb - expected_mb) > expected_mb * 0.1:
                print(f"⚠ Size mismatch for {dest_path.name}: expected ~{expected_mb}MB, got {actual_mb:.1f}MB")

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded {dest_path.name} ({total_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dest_path.name}: {e}")
        return False

def download_chroma_db(release_url, dest_dir):
    """Download Chroma DB files."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for chroma_file in CHROMA_FILES:
        file_url = f"{release_url}/{chroma_file}"
        dest_path = dest_dir / chroma_file

        if dest_path.exists():
            print(f"✓ {chroma_file} already exists")
            continue

        try:
            response = requests.get(file_url, stream=True, timeout=60)
            response.raise_for_status()

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Downloaded {chroma_file}")
        except Exception as e:
            print(f"✗ Failed to download {chroma_file}: {e}")
            return False
    return True

def main():
    """Main download function."""
    print("=" * 60)
    print("AI Fraud Detection - Data Download for Streamlit Cloud")
    print("=" * 60)

    # Determine base URL
    # For GitHub Releases: https://github.com/{REPO}/releases/download/{TAG}/{FILE}
    base_url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

    # Local data directory
    data_dir = Path(__file__).parent.parent / "data"
    models_dir = Path(__file__).parent.parent / "models"
    chroma_dir = Path(__file__).parent.parent / "chroma_db"

    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Chroma directory: {chroma_dir}")
    print()

    # Download all files (FILES list contains all needed files)
    for filename, rel_path, expected_mb in FILES:
        url = f"{base_url}/{filename}"

        # Determine destination based on rel_path
        if rel_path.startswith("models/"):
            dest = models_dir / rel_path.replace("models/", "")
        elif rel_path.startswith("chroma_db/"):
            dest = chroma_dir / rel_path.replace("chroma_db/", "")
        else:
            dest = data_dir / rel_path.replace("data/", "")

        download_file(url, dest, expected_mb)

    print()
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()