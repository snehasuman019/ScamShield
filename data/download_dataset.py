"""
Dataset download and verification utility for ScamShield.
Downloads the standard Employment Scam Aegean Dataset (EMSCAD / fake_job_postings.csv)
from a reliable public repository into data/fake_job_postings.csv.
"""

import os
import sys
import requests
import pandas as pd

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CSV = os.path.join(DATASET_DIR, "fake_job_postings.csv")

# Public reliable sources for the standard EMSCAD 17,880-row dataset
PRIMARY_URL = "https://raw.githubusercontent.com/Cindyalifia/bangkit-project-1/master/fake_job_postings.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/38832/Fake-Job-Posting-Prediction/master/fake_job_postings.csv"


def download_dataset(target_path=TARGET_CSV, force=False):
    if os.path.exists(target_path) and not force:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[OK] Dataset already exists at {target_path} ({size_mb:.2f} MB).")
        return verify_dataset(target_path)

    urls = [PRIMARY_URL, FALLBACK_URL]
    download_success = False

    for url in urls:
        try:
            print(f"[INFO] Attempting download from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 64  # 64 KB

                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_bytes > 0:
                                pct = (downloaded / total_bytes) * 100
                                sys.stdout.write(f"\r[DOWNLOADING] {pct:.1f}% ({downloaded / (1024*1024):.2f} MB)")
                                sys.stdout.flush()
                print("\n[OK] Download finished successfully!")
                download_success = True
                break
            else:
                print(f"[WARN] HTTP status {response.status_code} from {url}")
        except Exception as e:
            print(f"[ERROR] Failed to download from {url}: {e}")

    if not download_success:
        print("[ERROR] Could not download dataset automatically.")
        print(f"Please manually download fake_job_postings.csv and place it at:\n{target_path}")
        return False

    return verify_dataset(target_path)


def verify_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"[VERIFY] Successfully loaded {csv_path}")
        print(f"[VERIFY] Total records: {len(df):,}")
        print(f"[VERIFY] Total columns: {len(df.columns)}")
        print(f"[VERIFY] Columns list: {list(df.columns)}")
        if "fraudulent" in df.columns:
            counts = df["fraudulent"].value_counts().to_dict()
            print(f"[VERIFY] Target class distribution: {counts}")
            return True
        else:
            print("[ERROR] Required target column 'fraudulent' not found in dataset!")
            return False
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
