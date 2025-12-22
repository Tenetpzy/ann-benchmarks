#!/usr/bin/env python3
"""
Script to rename algorithm names stored in HDF5 result files.

This script traverses all HDF5 files in a specified directory and modifies
the 'algo' attribute stored in each file. It can also optionally rename
the directory structure to match the new algorithm name.

Usage:
    python rename_algorithm_in_hdf5.py --dir /path/to/results/dataset --old-name hnswlib --new-name hnsw_baseline [--rename-dirs]

Example:
    python rename_algorithm_in_hdf5.py \
        --dir results/mnist-784-euclidean/10 \
        --old-name hnswlib \
        --new-name hnsw_baseline \
        --rename-dirs
"""

import argparse
import h5py
import os
import shutil
from pathlib import Path
from typing import Tuple, List


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


def rename_directory(old_path: str, new_path: str) -> bool:
    """Rename a directory.

    Args:
        old_path: Current directory path
        new_path: New directory path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        # Move the directory
        shutil.move(old_path, new_path)
        print(f"  Renamed directory: {old_path} -> {new_path}")
        return True
    except Exception as e:
        print(f"  Error renaming directory {old_path}: {e}")
        return False


def verify_changes(hdf5_files: List[str], new_name: str) -> Tuple[int, int]:
    """Verify that all files were updated correctly.

    Args:
        hdf5_files: List of HDF5 file paths
        new_name: Expected algorithm name

    Returns:
        Tuple of (success_count, failure_count)
    """
    success = 0
    failure = 0

    for hdf5_file in hdf5_files:
        current = get_current_algo_name(hdf5_file)
        if current == new_name:
            success += 1
        else:
            print(f"  WARNING: {hdf5_file} still has algo={current}")
            failure += 1

    return success, failure


def main():
    parser = argparse.ArgumentParser(
        description='Rename algorithm names in HDF5 result files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update algorithm name in HDF5 files without renaming directories
  python rename_algorithm_in_hdf5.py --dir results/mnist-784-euclidean/10 --old-name hnswlib --new-name hnsw_baseline

  # Update algorithm name and rename directories to match
  python rename_algorithm_in_hdf5.py --dir results/mnist-784-euclidean/10 --old-name hnswlib --new-name hnsw_baseline --rename-dirs

  # Dry run to see what would be changed
  python rename_algorithm_in_hdf5.py --dir results/mnist-784-euclidean/10 --old-name hnswlib --new-name hnsw_baseline --dry-run
        """
    )

    parser.add_argument(
        '--dir',
        required=True,
        help='Directory containing HDF5 result files (e.g., results/mnist-784-euclidean/10)'
    )

    parser.add_argument(
        '--old-name',
        required=True,
        help='Current algorithm name to replace (e.g., hnswlib)'
    )

    parser.add_argument(
        '--new-name',
        required=True,
        help='New algorithm name (e.g., hnsw_baseline)'
    )

    parser.add_argument(
        '--rename-dirs',
        action='store_true',
        help='Also rename algorithm directories to match the new name'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making actual changes'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify that all changes were applied correctly'
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} does not exist")
        return 1

    # Find all HDF5 files
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
        marker = " <-- TARGET" if algo == args.old_name else ""
        print(f"  {algo}: {count} files{marker}")

    # Filter files with the old algorithm name
    files_to_update = []
    for hdf5_file in hdf5_files:
        current_algo = get_current_algo_name(hdf5_file)
        if current_algo == args.old_name:
            files_to_update.append(hdf5_file)

    if not files_to_update:
        print(f"\nNo files found with algorithm name '{args.old_name}'")
        return 0

    print(f"\n{len(files_to_update)} files need to be updated")

    # Show files that would be changed
    if args.dry_run:
        print("\nFiles that would be updated:")
        for hdf5_file in files_to_update:
            print(f"  {hdf5_file}")

        if args.rename_dirs:
            # Find directories that would be renamed
            dirs_to_rename = set()
            for hdf5_file in files_to_update:
                dir_path = os.path.dirname(hdf5_file)
                if os.path.basename(dir_path) == args.old_name:
                    parent_dir = os.path.dirname(dir_path)
                    new_dir_path = os.path.join(parent_dir, args.new_name)
                    dirs_to_rename.add((dir_path, new_dir_path))

            if dirs_to_rename:
                print("\nDirectories that would be renamed:")
                for old_path, new_path in sorted(dirs_to_rename):
                    print(f"  {old_path} -> {new_path}")

        print("\n[Dry run mode - no changes were made]")
        return 0

    # Perform the updates
    print(f"\nUpdating algorithm name from '{args.old_name}' to '{args.new_name}'...")

    success_count = 0
    for hdf5_file in files_to_update:
        if update_algo_name(hdf5_file, args.new_name):
            success_count += 1

    print(f"\nSuccessfully updated {success_count}/{len(files_to_update)} files")

    # Rename directories if requested
    if args.rename_dirs:
        print(f"\nRenaming directories from '{args.old_name}' to '{args.new_name}'...")

        dirs_to_rename = set()
        for hdf5_file in files_to_update:
            dir_path = os.path.dirname(hdf5_file)
            if os.path.basename(dir_path) == args.old_name:
                parent_dir = os.path.dirname(dir_path)
                new_dir_path = os.path.join(parent_dir, args.new_name)
                dirs_to_rename.add((dir_path, new_dir_path))

        if dirs_to_rename:
            for old_path, new_path in sorted(dirs_to_rename):
                print(f"  {old_path} -> {new_path}")
                if not rename_directory(old_path, new_path):
                    print(f"  WARNING: Failed to rename {old_path}")
        else:
            print("  No matching directories found")

    # Verify changes if requested
    if args.verify:
        print(f"\nVerifying changes...")
        success, failure = verify_changes(files_to_update, args.new_name)
        print(f"Verification complete: {success} successful, {failure} failed")

        if failure > 0:
            print("\nSome files were not updated correctly. Please check the output above.")
            return 1

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
