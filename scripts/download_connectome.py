"""
Download Male Drosophila CNS Connectome v1.0 data files from Janelia / Google Research public GCS bucket.
"""
import os
import sys
import time
import urllib.request
import pyarrow.feather as feather

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "original"))

FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

def download_file(filename, dest_path):
    url = f"{BASE_URL}/{filename}"
    print(f"[*] Checking {filename}...")
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"    File already exists ({size_mb:.2f} MB). Verifying feather format...")
        try:
            tbl = feather.read_table(dest_path)
            print(f"    [+] Successfully verified {filename}: {tbl.num_rows:,} rows, {tbl.num_columns} cols.")
            return
        except Exception as e:
            print(f"    [!] Corrupt or incomplete file ({e}). Re-downloading...")
            os.remove(dest_path)

    print(f"[*] Downloading {url} -> {dest_path}...")
    start_time = time.time()
    last_print = 0

    def reporthook(block_num, block_size, total_size):
        nonlocal last_print
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            now = time.time()
            if now - last_print > 5 or downloaded >= total_size:
                elapsed = now - start_time if now > start_time else 1e-5
                speed = (downloaded / (1024 * 1024)) / elapsed
                print(f"    {percent:5.1f}% | {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB | {speed:.2f} MB/s")
                last_print = now

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    total_elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[+] Download complete: {filename} ({file_size_mb:.2f} MB in {total_elapsed:.1f}s)")
    
    # Verify feather
    tbl = feather.read_table(dest_path)
    print(f"[+] Verified {filename}: {tbl.num_rows:,} rows, {tbl.num_columns} columns.")

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Target directory: {DEST_DIR}")
    for fname in FILES:
        dest_path = os.path.join(DEST_DIR, fname)
        download_file(fname, dest_path)
    print("\n[+] All core connectome files ready!")

if __name__ == "__main__":
    main()
