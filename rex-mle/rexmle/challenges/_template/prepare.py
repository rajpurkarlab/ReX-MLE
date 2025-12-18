"""
Data preparation script for [CHALLENGE NAME]

This script:
1. Downloads data from Zenodo (if not present)
2. Organizes data into standard structure
3. Creates train/test splits
4. Generates sample submission
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def download_data(raw_dir: Path) -> None:
    """
    Download data from Zenodo.

    Args:
        raw_dir: Directory to save raw data
    """
    # Load config to get Zenodo info
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    zenodo_config = config['zenodo']

    print(f"Downloading data from Zenodo record: {zenodo_config['record_id']}")

    downloader = ZenodoDownloader()

    # Download files
    files = downloader.download_record(
        record_id=zenodo_config['record_id'],
        output_dir=raw_dir,
        files=zenodo_config.get('files')  # Download specific files if specified
    )

    # Extract archives
    for file in files:
        if file.suffix in ['.zip', '.tar', '.gz', '.tgz']:
            downloader.extract_archive(file, raw_dir)
            print(f"Extracted: {file.name}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare challenge data.

    This function splits raw data into public (for participants) and private (for grading).

    Args:
        raw: Path to raw data directory (downloaded from Zenodo)
        public: Path to public data directory (visible to participants)
        private: Path to private data directory (held out for grading)
    """
    print(f"Preparing challenge data...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if raw directory is empty or doesn't exist
    if not raw.exists() or not list(raw.glob('*')):
        print("\nDownloading data from Zenodo...")
        download_data(raw)

    # ===================================================================
    # CUSTOMIZE THIS SECTION FOR YOUR CHALLENGE
    # ===================================================================

    # Example: Load annotations
    # annotations_path = raw / "annotations.csv"
    # annotations = pd.read_csv(annotations_path)

    # Example: Create train/test split
    # train_df, test_df = train_test_split(
    #     annotations,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=annotations['label']  # For classification
    # )

    # Example: Create directories
    # (public / "train").mkdir(parents=True, exist_ok=True)
    # (public / "test").mkdir(parents=True, exist_ok=True)
    # (private).mkdir(parents=True, exist_ok=True)

    # Example: Copy/process images
    # for idx, row in train_df.iterrows():
    #     image_id = row['image_id']
    #     src = raw / "images" / f"{image_id}.dcm"
    #     dst = public / "train" / f"{image_id}.dcm"
    #     shutil.copy(src, dst)

    # Example: Save labels
    # train_df.to_csv(public / "train.csv", index=False)
    # test_df.to_csv(private / "test.csv", index=False)  # Ground truth (hidden)

    # Example: Create test CSV without labels (for public)
    # test_public = test_df.drop(columns=['label'])
    # test_public.to_csv(public / "test.csv", index=False)

    # Example: Create sample submission
    # sample_sub = test_public.copy()
    # sample_sub['prediction'] = 0.5  # Or appropriate default
    # sample_sub.to_csv(public / "sample_submission.csv", index=False)

    # ===================================================================
    # END CUSTOMIZATION
    # ===================================================================

    print("\n✓ Data preparation complete!")

    # Validation checks
    # assert public.exists(), "Public directory not created"
    # assert private.exists(), "Private directory not created"
    # assert (public / "sample_submission.csv").exists(), "Sample submission not created"

    print("\nFiles created:")
    print(f"  Public: {list(public.rglob('*'))[:5]}...")  # Show first 5 files
    print(f"  Private: {list(private.rglob('*'))[:5]}...")
