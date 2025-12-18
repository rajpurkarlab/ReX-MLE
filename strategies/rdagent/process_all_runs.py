#!/usr/bin/env python3
"""
Process all rdagent runs from jsonl file.
For each run, unpickle all pickle files and save to a folder named after the challenge.
"""

import json
import pickle
import os
from pathlib import Path
from datetime import datetime

# Try to import dill for better compatibility with custom objects
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

def safe_unpickle(pkl_path):
    """
    Attempt to unpickle a file using dill (if available) or pickle.
    Returns (success, data_or_error)
    """
    try:
        with open(pkl_path, 'rb') as f:
            if HAS_DILL:
                data = dill.load(f)
            else:
                data = pickle.load(f)
        return True, data
    except Exception as e:
        return False, str(e)

def unpickle_all_files(source_dir, output_file):
    """
    Recursively find all .pkl files in source_dir, unpickle them,
    and write their contents to output_file.

    Args:
        source_dir: Directory to search for pickle files
        output_file: Path to output text file
    """
    pkl_files = list(Path(source_dir).rglob("*.pkl"))
    pkl_files.sort()

    if len(pkl_files) == 0:
        print(f"  WARNING: No pickle files found in {source_dir}")
        return False

    print(f"  Found {len(pkl_files)} pickle files")

    successful = 0
    failed = 0

    with open(output_file, 'w') as out_f:
        out_f.write(f"Unpickled data from {source_dir}\n")
        out_f.write(f"Generated on: {datetime.now()}\n")
        out_f.write(f"Total files: {len(pkl_files)}\n")
        out_f.write(f"Unpickler: {'dill' if HAS_DILL else 'pickle'}\n")
        out_f.write("=" * 80 + "\n\n")

        for idx, pkl_path in enumerate(pkl_files, 1):
            success, result = safe_unpickle(pkl_path)

            # Write file path and data
            out_f.write(f"\n{'='*80}\n")
            out_f.write(f"File {idx}/{len(pkl_files)}: {pkl_path.relative_to(source_dir)}\n")
            out_f.write(f"{'='*80}\n")

            if success:
                out_f.write(str(result))
                successful += 1
            else:
                out_f.write(f"[ERROR] Failed to unpickle: {result}\n")
                failed += 1

            out_f.write("\n")

            if idx % 100 == 0:
                print(f"    Processed {idx}/{len(pkl_files)} files... ({successful} successful, {failed} failed)")

    print(f"  ✓ Successfully unpickled: {successful}")
    print(f"  ✗ Failed to unpickle: {failed}")
    print(f"  📁 Output file: {output_file}")
    print(f"  📊 Output size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    return True

def process_all_runs(jsonl_file, output_base_dir):
    """
    Read jsonl file and process all runs.

    Args:
        jsonl_file: Path to jsonl file with challenge_id and submission_path
        output_base_dir: Base directory to create challenge-specific folders in
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    with open(jsonl_file, 'r') as f:
        runs = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(runs)} runs to process\n")

    successful_runs = 0
    failed_runs = 0

    for run_idx, run in enumerate(runs, 1):
        challenge_id = run.get('challenge_id')
        submission_path = run.get('submission_path')

        if not challenge_id or not submission_path:
            print(f"Run {run_idx}/{len(runs)}: SKIPPED - Missing challenge_id or submission_path")
            failed_runs += 1
            continue

        if not os.path.exists(submission_path):
            print(f"Run {run_idx}/{len(runs)}: {challenge_id} - SKIPPED - Path does not exist")
            print(f"  Path: {submission_path}")
            failed_runs += 1
            continue

        # Create challenge-specific output folder
        challenge_output_dir = os.path.join(output_base_dir, challenge_id)
        os.makedirs(challenge_output_dir, exist_ok=True)
        output_file = os.path.join(challenge_output_dir, "unpickled_data.txt")

        print(f"Run {run_idx}/{len(runs)}: {challenge_id}")
        print(f"  Source: {submission_path}")
        print(f"  Output dir: {challenge_output_dir}")

        try:
            if unpickle_all_files(submission_path, output_file):
                successful_runs += 1
            else:
                failed_runs += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_runs += 1

        print()

    print("=" * 80)
    print(f"Summary:")
    print(f"  Total runs: {len(runs)}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {failed_runs}")
    print(f"  Output directory: {output_base_dir}")

if __name__ == "__main__":
    jsonl_file = "./rdagent_runs.jsonl"
    output_base_dir = "./unpickled_runs"

    process_all_runs(jsonl_file, output_base_dir)
