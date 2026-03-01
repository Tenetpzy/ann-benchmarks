#!/usr/bin/env python3
"""
Script to rename algorithm names stored in HDF5 result files.

This script traverses all HDF5 files in a specified directory and modifies
the 'algo' attribute stored in each file to a new name.

Usage:
    # Rename all HDF5 files to a single name:
    python rename_algorithm_in_hdf5.py --dir /path/to/results/dataset --new-name hnsw_baseline

    # Rename each subdirectory's files to its subdirectory name:
    python rename_algorithm_in_hdf5.py --dir /path/to/results/dataset --use-dir-name

Examples:
    python rename_algorithm_in_hdf5.py \
        --dir results/mnist-784-euclidean/10 \
        --new-name hnsw_baseline

    python rename_algorithm_in_hdf5.py \
        --dir results/sift-128-euclidean/10 \
        --use-dir-name
"""

import argparse
import h5py
import os
from typing import List


def find_hdf5_files(directory: str) -> List[str]:
    """Find all HDF5 files in the directory and its subdirectories.

    Args:
        directory: Root directory to search for HDF5 files

    Returns:
        List of paths to HDF5 files
    """
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files


def find_hdf5_files_by_subdir(directory: str) -> dict:
    """Find HDF5 files grouped by immediate subdirectory.

    Args:
        directory: Root directory containing subdirectories with HDF5 files

    Returns:
        Dict mapping subdirectory name to list of HDF5 file paths
    """
    result = {}
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            hdf5_files = []
            for file in os.listdir(item_path):
                if file.endswith('.hdf5'):
                    hdf5_files.append(os.path.join(item_path, file))
            if hdf5_files:
                result[item] = hdf5_files
    return result


def get_current_algo_name(hdf5_file: str) -> str:
    """Read the current algorithm name from HDF5 file attributes.

    Args:
        hdf5_file: Path to the HDF5 file

    Returns:
        The algorithm name stored in the file, or None if not found
    """
    try:
        with h5py.File(hdf5_file, 'r') as f:
            return f.attrs.get('algo')
    except Exception as e:
        print(f"Error reading {hdf5_file}: {e}")
        return None


def update_algo_name(hdf5_file: str, new_name: str) -> bool:
    """Update the algorithm name in HDF5 file attributes.

    Args:
        hdf5_file: Path to the HDF5 file
        new_name: New algorithm name to set

    Returns:
        True if successful, False otherwise
    """
    try:
        with h5py.File(hdf5_file, 'r+') as f:
            old_name = f.attrs.get('algo')
            f.attrs['algo'] = new_name
            print(f"  Updated {os.path.basename(hdf5_file)}: {old_name} -> {new_name}")
            return True
    except Exception as e:
        print(f"  Error updating {hdf5_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Rename algorithm names in HDF5 result files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all HDF5 files to a single algorithm name
  python rename_algorithm_in_hdf5.py --dir results/mnist-784-euclidean/10 --new-name hnsw_baseline

  # Rename each subdirectory's files to use the subdirectory name
  python rename_algorithm_in_hdf5.py --dir results/sift-128-euclidean/10 --use-dir-name

  # Dry run to see what would be changed
  python rename_algorithm_in_hdf5.py --dir results/mnist-784-euclidean/10 --new-name hnsw_baseline --dry-run
  python rename_algorithm_in_hdf5.py --dir results/sift-128-euclidean/10 --use-dir-name --dry-run
        """
    )

    parser.add_argument(
        '--dir',
        required=True,
        help='Directory containing HDF5 result files (e.g., results/mnist-784-euclidean/10)'
    )

    parser.add_argument(
        '--new-name',
        help='New algorithm name (e.g., hnsw_baseline). Required unless --use-dir-name is specified.'
    )

    parser.add_argument(
        '--use-dir-name',
        action='store_true',
        help='Use the subdirectory name as the new algorithm name for each file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making actual changes'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.use_dir_name and not args.new_name:
        parser.error("Either --new-name or --use-dir-name must be specified")

    if args.use_dir_name and args.new_name:
        parser.error("Cannot specify both --new-name and --use-dir-name")

    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} does not exist")
        return 1

    if args.use_dir_name:
        # Mode: rename based on subdirectory name
        print(f"\nSearching for HDF5 files in subdirectories of {args.dir}...")
        files_by_subdir = find_hdf5_files_by_subdir(args.dir)

        if not files_by_subdir:
            print("No HDF5 files found in subdirectories.")
            return 0

        total_files = sum(len(files) for files in files_by_subdir.values())
        print(f"Found {total_files} HDF5 files in {len(files_by_subdir)} subdirectories\n")

        # Show current state
        print("Current algorithm names:")
        for subdir, files in sorted(files_by_subdir.items()):
            current_algo = get_current_algo_name(files[0]) if files else None
            print(f"  {subdir}/ ({len(files)} files): {current_algo} -> {subdir}")

        if args.dry_run:
            print("\n[Dry run mode - no changes were made]")
            return 0

        # Perform the updates
        print(f"\nUpdating algorithm names to subdirectory names...")
        success_count = 0
        for subdir, files in sorted(files_by_subdir.items()):
            print(f"\nProcessing {subdir}/ ({len(files)} files):")
            for hdf5_file in files:
                if update_algo_name(hdf5_file, subdir):
                    success_count += 1

        print(f"\nSuccessfully updated {success_count}/{total_files} files")
        print("\nDone!")
        return 0
    else:
        # Mode: rename all to a single new name (original behavior)
        print(f"\nSearching for HDF5 files in {args.dir}...")
        hdf5_files = find_hdf5_files(args.dir)

        if not hdf5_files:
            print("No HDF5 files found.")
            return 0

        print(f"Found {len(hdf5_files)} HDF5 files\n")

        # Show current state
        print("Current algorithm names in files:")
        algo_counts = {}
        for hdf5_file in hdf5_files:
            current_algo = get_current_algo_name(hdf5_file)
            if current_algo:
                algo_counts[current_algo] = algo_counts.get(current_algo, 0) + 1

        for algo, count in sorted(algo_counts.items()):
            print(f"  {algo}: {count} files")

        print(f"\n{len(hdf5_files)} files will be updated to '{args.new_name}'")

        # Show files that would be changed
        if args.dry_run:
            print("\nFiles that would be updated:")
            for hdf5_file in hdf5_files:
                current_algo = get_current_algo_name(hdf5_file)
                print(f"  {hdf5_file}: {current_algo} -> {args.new_name}")
            print("\n[Dry run mode - no changes were made]")
            return 0

        # Perform the updates
        print(f"\nUpdating all algorithm names to '{args.new_name}'...")

        success_count = 0
        for hdf5_file in hdf5_files:
            if update_algo_name(hdf5_file, args.new_name):
                success_count += 1

        print(f"\nSuccessfully updated {success_count}/{len(hdf5_files)} files")
        print("\nDone!")
        return 0


if __name__ == '__main__':
    exit(main())