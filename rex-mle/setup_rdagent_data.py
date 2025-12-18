#!/usr/bin/env python3
"""
Setup rdagent_data folder structure for rex-mle

This script creates a rdagent_data directory with symbolic links to the
challenge_data folder, following the same pattern as rex-mle.

The structure created is:
  rdagent_data/
    ├── eval/
    │   └── <challenge_name>/
    │       ├── grade.py (copied from rexmle package)
    │       ├── valid.py (generated format validation script)
    │       └── submission_test.csv -> challenge_data/<challenge>/prepared/private/test_labels.csv
    ├── zip_files/
    │   └── <challenge_name>.zip -> challenge_data/<challenge>/raw/*.zip (if exists)
    ├── <challenge_name>/
    │   ├── description.md -> challenge_data/<challenge_name>/prepared/public/description.md
    │   ├── sample_submission.csv -> challenge_data/<challenge_name>/prepared/public/sample_submission.csv
    │   ├── test -> challenge_data/<challenge_name>/prepared/public/test
    │   ├── train -> challenge_data/<challenge_name>/prepared/public/train
    │   └── ... (any other files in public/)
    └── ...

Usage:
    python setup_rdagent_data.py [--dry-run]
"""

import os
import sys
import shutil
from pathlib import Path
import argparse


def get_base_dirs():
    """Get the base directories for the project"""
    script_dir = Path(__file__).resolve().parent
    challenge_data_dir = script_dir / "challenge_data"
    rdagent_data_dir = script_dir / "rdagent_data"
    rexmle_root = script_dir

    return challenge_data_dir, rdagent_data_dir, rexmle_root


def get_all_challenges(challenge_data_dir):
    """Get list of all challenge directories"""
    if not challenge_data_dir.exists():
        print(f"❌ ERROR: challenge_data directory not found at {challenge_data_dir}")
        sys.exit(1)

    challenges = []
    for item in sorted(challenge_data_dir.iterdir()):
        if item.is_dir():
            # Check if prepared/public exists
            public_dir = item / "prepared" / "public"
            if public_dir.exists():
                challenges.append(item.name)

    return challenges


def create_symlinks_for_challenge(challenge_name, challenge_data_dir, rdagent_data_dir, dry_run=False):
    """Create symbolic links for a single challenge"""
    source_dir = challenge_data_dir / challenge_name / "prepared" / "public"
    target_dir = rdagent_data_dir / challenge_name

    if not source_dir.exists():
        print(f"  ⚠️  Skipping {challenge_name}: prepared/public directory not found")
        return False

    # Create target directory
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  [DRY RUN] Would create directory: {target_dir}")

    # Get all files and directories in source
    items = list(source_dir.iterdir())

    if not items:
        print(f"  ⚠️  Skipping {challenge_name}: no files in prepared/public")
        return False

    symlinks_created = 0

    for item in items:
        target_path = target_dir / item.name

        # Check if symlink already exists and points to correct location
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_symlink() and target_path.resolve() == item.resolve():
                continue  # Already correct
            elif not dry_run:
                # Remove incorrect symlink
                target_path.unlink()

        # Create symlink
        if not dry_run:
            target_path.symlink_to(item)
            symlinks_created += 1
        else:
            print(f"    [DRY RUN] Would create symlink: {target_path.name} -> {item}")
            symlinks_created += 1

    if symlinks_created > 0:
        print(f"  ✓ {challenge_name}: Created {symlinks_created} symlinks")
    else:
        print(f"  ✓ {challenge_name}: All symlinks already exist")

    return True


def create_eval_directory(rdagent_data_dir, dry_run=False):
    """Create the eval directory"""
    eval_dir = rdagent_data_dir / "eval"

    if not dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created eval directory: {eval_dir}")
    else:
        print(f"[DRY RUN] Would create eval directory: {eval_dir}")


def create_zip_files_directory(rdagent_data_dir, dry_run=False):
    """Create the zip_files directory"""
    zip_dir = rdagent_data_dir / "zip_files"

    if not dry_run:
        zip_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created zip_files directory: {zip_dir}")
    else:
        print(f"[DRY RUN] Would create zip_files directory: {zip_dir}")


def setup_eval_for_challenge(challenge_name, challenge_data_dir, rdagent_data_dir, rexmle_root, dry_run=False):
    """Setup eval directory for a challenge with grade.py, valid.py, and submission_test.csv symlink"""
    eval_dir = rdagent_data_dir / "eval" / challenge_name
    private_dir = challenge_data_dir / challenge_name / "prepared" / "private"

    # Check if private directory exists
    if not private_dir.exists():
        return False

    # Create eval challenge directory
    if not dry_run:
        eval_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"    [DRY RUN] Would create directory: {eval_dir}")

    files_created = 0

    # 1. Copy grade.py from rexmle package
    grade_source = rexmle_root / "rexmle" / "challenges" / challenge_name / "grade.py"
    grade_target = eval_dir / "grade.py"

    if grade_source.exists():
        if not dry_run:
            if grade_target.exists():
                grade_target.unlink()
            shutil.copy2(grade_source, grade_target)
            files_created += 1
        else:
            print(f"    [DRY RUN] Would copy grade.py from {grade_source}")
            files_created += 1

    # 2. Create valid.py (format validation script)
    valid_target = eval_dir / "valid.py"
    valid_content = f'''"""
Submission format validation for {challenge_name}
"""
from pathlib import Path
import pandas as pd

# Check if our submission file exists
assert Path("submission.csv").exists(), "Error: submission.csv not found"

submission_lines = Path("submission.csv").read_text().splitlines()
test_lines = Path("submission_test.csv").read_text().splitlines()

is_valid = len(submission_lines) == len(test_lines)

if is_valid:
    message = f"✓ submission.csv and submission_test.csv have the same number of lines ({{len(submission_lines)}})."
else:
    message = (
        f"✗ submission.csv has {{len(submission_lines)}} lines, "
        f"while submission_test.csv has {{len(test_lines)}} lines."
    )

print(message)

if not is_valid:
    raise AssertionError("Submission is invalid: line count mismatch")

# Additional validation: check if files are valid CSVs
try:
    sub_df = pd.read_csv("submission.csv")
    test_df = pd.read_csv("submission_test.csv")

    # Check if columns match
    if list(sub_df.columns) != list(test_df.columns):
        raise AssertionError(
            f"Column mismatch: submission has {{list(sub_df.columns)}}, "
            f"test has {{list(test_df.columns)}}"
        )

    print(f"✓ Columns match: {{list(sub_df.columns)}}")
    print("✓ Submission format is valid")

except Exception as e:
    raise AssertionError(f"Failed to parse CSV files: {{e}}")
'''

    if not dry_run:
        if valid_target.exists():
            valid_target.unlink()
        valid_target.write_text(valid_content)
        files_created += 1
    else:
        print(f"    [DRY RUN] Would create valid.py")
        files_created += 1

    # 3. Create symlink to test_labels.csv as submission_test.csv
    test_labels_source = private_dir / "test_labels.csv"
    submission_test_target = eval_dir / "submission_test.csv"

    if test_labels_source.exists():
        if not dry_run:
            if submission_test_target.exists() or submission_test_target.is_symlink():
                submission_test_target.unlink()
            submission_test_target.symlink_to(test_labels_source)
            files_created += 1
        else:
            print(f"    [DRY RUN] Would create symlink: submission_test.csv -> {test_labels_source}")
            files_created += 1

    return files_created


def setup_zip_for_challenge(challenge_name, challenge_data_dir, rdagent_data_dir, dry_run=False):
    """Create symlink to zip file if it exists"""
    raw_dir = challenge_data_dir / challenge_name / "raw"
    zip_dir = rdagent_data_dir / "zip_files"

    if not raw_dir.exists():
        return False

    # Find zip files in raw directory
    zip_files = list(raw_dir.glob("*.zip"))

    if not zip_files:
        return False

    # Use the first zip file (or could combine multiple)
    # For now, just pick the largest zip file
    largest_zip = max(zip_files, key=lambda p: p.stat().st_size)

    target_link = zip_dir / f"{challenge_name}.zip"

    if not dry_run:
        if target_link.exists() or target_link.is_symlink():
            if target_link.is_symlink() and target_link.resolve() == largest_zip.resolve():
                return False  # Already correct
            target_link.unlink()
        target_link.symlink_to(largest_zip)
        return True
    else:
        print(f"    [DRY RUN] Would create symlink: {challenge_name}.zip -> {largest_zip}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup rdagent_data folder structure with symbolic links to challenge_data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating files/symlinks"
    )
    parser.add_argument(
        "--challenge",
        type=str,
        help="Setup only a specific challenge (default: all challenges)"
    )

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║              ReX-MLE RDAgent Data Setup                              ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()

    # Get directories
    challenge_data_dir, rdagent_data_dir, rexmle_root = get_base_dirs()

    print(f"Challenge data directory: {challenge_data_dir}")
    print(f"RDAgent data directory:   {rdagent_data_dir}")
    print(f"ReX-MLE root:       {rexmle_root}")
    print()

    # Create rdagent_data directory
    if not args.dry_run:
        rdagent_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created rdagent_data directory")
    else:
        print(f"[DRY RUN] Would create rdagent_data directory")

    # Create eval directory
    create_eval_directory(rdagent_data_dir, args.dry_run)

    # Create zip_files directory
    create_zip_files_directory(rdagent_data_dir, args.dry_run)
    print()

    # Get challenges to process
    if args.challenge:
        challenges = [args.challenge]
        print(f"Processing single challenge: {args.challenge}")
    else:
        challenges = get_all_challenges(challenge_data_dir)
        print(f"Found {len(challenges)} challenges to process")

    print()
    print("Creating symbolic links for public data...")
    print()

    # Create symlinks for each challenge (public data)
    success_count = 0
    for challenge in challenges:
        if create_symlinks_for_challenge(challenge, challenge_data_dir, rdagent_data_dir, args.dry_run):
            success_count += 1

    print()
    print("Setting up eval directories...")
    print()

    # Setup eval for each challenge
    eval_count = 0
    for challenge in challenges:
        files = setup_eval_for_challenge(challenge, challenge_data_dir, rdagent_data_dir, rexmle_root, args.dry_run)
        if files:
            eval_count += 1
            if not args.dry_run:
                print(f"  ✓ {challenge}: Created {files} eval files")

    print()
    print("Setting up zip file links...")
    print()

    # Setup zip files for each challenge
    zip_count = 0
    for challenge in challenges:
        if setup_zip_for_challenge(challenge, challenge_data_dir, rdagent_data_dir, args.dry_run):
            zip_count += 1
            if not args.dry_run:
                print(f"  ✓ {challenge}: Created zip symlink")

    print()
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                              Summary                                       ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"✓ Successfully processed {success_count}/{len(challenges)} challenges")
    print(f"✓ Created eval directories for {eval_count}/{len(challenges)} challenges")
    print(f"✓ Created zip symlinks for {zip_count}/{len(challenges)} challenges")

    if args.dry_run:
        print()
        print("🔍 This was a dry run. Run without --dry-run to apply changes.")
    else:
        print()
        print(f"✓ RDAgent data directory ready at: {rdagent_data_dir}")
        print()
        print("You can now run RDAgent with these challenges!")

    print()


if __name__ == "__main__":
    main()
