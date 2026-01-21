import os
import re
import tarfile
from tqdm import tqdm
import argparse


def find_all_chunk_groups(input_dir):
    """
    Returns a dict grouping chunks by base tar.gz name.

    Example return value:
    {
        "digital_twin.tar.gz": ["digital_twin.tar.gz.000", "digital_twin.tar.gz.001", ...]
        "DrivIng.tar.gz":   ["DrivIng.tar.gz.000",   "DrivIng.tar.gz.001", ...],
    }
    """
    pattern = re.compile(r"(.*\.tar\.gz)\.(\d{3})$")
    groups = {}

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            base, index = match.groups()
            groups.setdefault(base, []).append((int(index), fname))

    # Sort chunks inside each group
    for base in groups:
        groups[base] = [fname for _, fname in sorted(groups[base])]

    return groups


def reassemble_tar_gz(input_dir, base_name, chunks):
    """Reassemble split files into a single tar.gz with progress bars."""
    output_tar_path = os.path.join(input_dir, base_name)
    num_chunks = len(chunks)

    print(f"\n🔧 Reassembling: {base_name}")

    with open(output_tar_path, "wb") as outfile:

        for i, chunk_file in enumerate(chunks, start=1):
            chunk_path = os.path.join(input_dir, chunk_file)
            chunk_size = os.path.getsize(chunk_path)

            desc = f"Adding {i}/{num_chunks}: {chunk_file}"

            # Per-chunk progress bar
            with open(chunk_path, "rb") as infile, tqdm(
                total=chunk_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True
            ) as pbar:

                # copy chunk with progress
                while True:
                    buf = infile.read(1024 * 1024)  # 1 MB buffer
                    if not buf:
                        break
                    outfile.write(buf)
                    pbar.update(len(buf))

    print(f"✔ Created: {output_tar_path}")
    return output_tar_path


def extract_tar_gz(tar_path, output_dir):
    """Extract tar.gz into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Extracting {tar_path} → {output_dir}")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)

    print("✔ Extraction complete.")
    return output_dir


def maybe_delete(paths):
    """Delete given files."""
    for p in paths:
        try:
            os.remove(p)
            print(f"🗑 Deleted {p}")
        except Exception as e:
            print(f"⚠ Could not delete {p}: {e}")


# -----------------------------------------
# Main function: extract EVERYTHING
# -----------------------------------------
def extract_all_files(input_dir, output_root, delete_chunks=False, delete_tar=False):
    os.makedirs(output_root, exist_ok=True)

    # 1. Find all chunk groups
    groups = find_all_chunk_groups(input_dir)

    if not groups:
        print("❌ No split tar.gz.* files found.")
        return

    print("\n=== Found files ===")
    for base, chunks in groups.items():
        print(f"{base}: {chunks}")

    # 2. Process each file
    for base_name, chunk_list in groups.items():

        # Build path for this file output
        file_name = base_name.replace(".tar.gz", "")
        extract_dir = os.path.join(output_root, file_name)

        # Assemble
        tar_path = reassemble_tar_gz(input_dir, base_name, chunk_list)

        # Extract
        extract_tar_gz(tar_path, extract_dir)

        # Optionally delete chunks
        if delete_chunks:
            to_delete = [os.path.join(input_dir, c) for c in chunk_list]
            maybe_delete(to_delete)

        # Optionally delete reassembled tar.gz
        if delete_tar:
            maybe_delete([tar_path])

    print("\n✔ All files extracted successfully.")


# -----------------------------
# Command-line interface
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Reassemble split tar.gz files and extract them.")
    parser.add_argument("input_dir", help="Directory containing split tar.gz files (*.tar.gz.001, etc.)")
    parser.add_argument("output_root", help="Root directory where datasets will be extracted")
    parser.add_argument("--delete-chunks", action="store_true",
                        help="Delete original split chunks after reassembly")
    parser.add_argument("--delete-tar", action="store_true",
                        help="Delete reassembled tar.gz after extraction")
    return parser.parse_args()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    extract_all_files(
        input_dir=args.input_dir,
        output_root=args.output_root,
        delete_chunks=args.delete_chunks,
        delete_tar=args.delete_tar
    )
