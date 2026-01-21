import os
import hashlib
import argparse
from pyDataverse.api import NativeApi


DATAVERSE_URL = "https://dataverse.harvard.edu"      # base URL
PERSISTENT_ID = "doi:10.7910/DVN/VBZKDY"             # dataset DOI or handle


# ----------------------
# Helpers
# ----------------------
def compute_md5(file_path):
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_file_curl(download_url, output_path):
    """
    Download a file using curl.
    Creates directories as needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.isfile(output_path):
        print(f"Removing incomplete file: {output_path}")
        os.remove(output_path)

    cmd = f'curl -L "{download_url}" -o "{output_path}"'
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to download {download_url}")


# ----------------------
# Dataverse helpers
# ----------------------
def get_dataset_files():
    """Fetch dataset JSON and return list of files."""
    api = NativeApi(DATAVERSE_URL)
    dataset = api.get_dataset(PERSISTENT_ID).json()
    version_data = dataset["data"]["latestVersion"]
    return version_data["files"]


# ----------------------
# Download a single file from dataset by label
# ----------------------
def download_single_file(file_label, out_dir):
    """
    Download a single file from the dataset by label.
    Skips if file exists and MD5 matches.
    """
    files = get_dataset_files()
    file_matches = []
    for f in files:
        if file_label in f["label"]:
            file_matches.append(f)

    if not file_matches:
        print(f"❌ File '{file_label}' not found in dataset.")
        return None

    for f_match in file_matches:
        file_id = f_match["dataFile"]["id"]
        md5 = f_match["dataFile"]["md5"]
        directory_label = f_match.get("directoryLabel", "")

        output_path = os.path.join(out_dir, directory_label, file_label)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Skip if MD5 matches
        if os.path.isfile(output_path) and compute_md5(output_path) == md5:
            print(f"✔ Skipping {output_path}, already downloaded")
            return output_path

        download_url = f"{DATAVERSE_URL}/api/access/datafile/{file_id}"
        print(f"Downloading {file_label}...")
        download_file_curl(download_url, output_path)
        print(f"Downloaded {output_path}")

    return output_path


# ----------------------
# Download all files from dataset
# ----------------------
def download_full_dataset(out_dir):
    """
    Download all files from dataset.
    """
    files = get_dataset_files()
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        file_label = f["label"]

        file_id = f["dataFile"]["id"]
        md5 = f["dataFile"]["md5"]
        directory_label = f.get("directoryLabel", "")
        output_path = os.path.join(out_dir, directory_label, file_label)

        # Skip if already downloaded
        if os.path.isfile(output_path) and compute_md5(output_path) == md5:
            print(f"Skipping {output_path}, already downloaded")
            continue

        download_url = f"{DATAVERSE_URL}/api/access/datafile/{file_id}"
        print(f"Downloading {file_label}...")
        download_file_curl(download_url, output_path)
        print(f"Downloaded {output_path}")

    print("\n✔ All files downloaded successfully.")



# -----------------------------
# Command-line interface
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Dataverse dataset (full or single file)."
    )
    parser.add_argument("out_dir", help="Output directory for downloads")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Download full dataset")
    group.add_argument("--file", type=str, help="Download a specific file or chunk set (label)")
    return parser.parse_args()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.full:
        download_full_dataset(out_dir=args.out_dir)
    elif args.file:
        download_single_file(file_label=args.file, out_dir=args.out_dir)
