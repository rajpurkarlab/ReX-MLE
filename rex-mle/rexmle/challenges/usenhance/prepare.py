"""
Data preparation script for USenhance Challenge 2023

This script:
1. Downloads training data from Google Drive
2. Splits into 80/20 train/test sets
3. Organizes data into standard structure (low-quality input, high-quality target pairs)
4. Creates sample submission file
"""

from pathlib import Path
import pandas as pd
import shutil
import gdown
from sklearn.model_selection import train_test_split
from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def download_data(raw_dir: Path) -> None:
    """
    Download training data from Google Drive.

    Args:
        raw_dir: Directory to save raw data
    """
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    downloader = ZenodoDownloader()

    # Download training data from Google Drive
    google_train_config = config.get('google_train_set')
    if google_train_config:
        file_id = google_train_config['id']
        train_zip_path = raw_dir / "train_dataset.zip"
        print(f"Downloading training data from Google Drive (ID: {file_id})...")
        gdown.download(id=file_id, output=str(train_zip_path), quiet=False)

        # Extract training archive
        if train_zip_path.exists():
            print(f"Extracting training data...")
            downloader.extract_archive(train_zip_path, raw_dir)


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare USenhance dataset for image enhancement challenge.

    Dataset structure:
    - Downloads training data (1500 pairs of low/high quality images)
    - Splits 80/20 into train/test sets
    - Test set ground truth kept private for evaluation

    Goals:
    1. Download and extract training data
    2. Split into 80% train, 20% test
    3. Copy train pairs to public/train/
    4. Copy test low-quality images to public/test/
    5. Copy test high-quality images (ground truth) to private/
    6. Create sample_submission.csv in public/
    """
    print(f"Preparing USenhance challenge data...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if raw directory is empty or doesn't exist
    if not raw.exists() or not list(raw.glob('*')):
        print("\nDownloading data from Google Drive...")
        download_data(raw)

    # Explore raw directory structure to understand the data organization
    print("\nExploring raw directory structure...")

    # Find train_datasets directory
    train_datasets_dir = raw / "train_datasets"
    if not train_datasets_dir.exists():
        # Try alternative paths
        possible_dirs = list(raw.glob('*train*')) + list(raw.glob('*Train*'))
        if possible_dirs:
            train_datasets_dir = possible_dirs[0]
        else:
            raise ValueError(f"Could not find training data directory in {raw}")

    print(f"Using training data directory: {train_datasets_dir}")

    # Find organ directories (breast, carotid, kidney, liver, thyroid)
    organ_dirs = [d for d in train_datasets_dir.iterdir() if d.is_dir()]
    print(f"Found {len(organ_dirs)} organ directories: {[d.name for d in organ_dirs]}")

    # Create output directories
    (public / "train" / "low_quality").mkdir(parents=True, exist_ok=True)
    (public / "train" / "high_quality").mkdir(parents=True, exist_ok=True)
    (public / "test" / "low_quality").mkdir(parents=True, exist_ok=True)
    (private).mkdir(parents=True, exist_ok=True)
    (private / "test" / "high_quality").mkdir(parents=True, exist_ok=True)

    # Collect all image pairs from all organs
    image_pairs = []

    for organ_dir in organ_dirs:
        organ_name = organ_dir.name
        low_quality_dir = organ_dir / "low_quality"
        high_quality_dir = organ_dir / "high_quality"

        if not low_quality_dir.exists() or not high_quality_dir.exists():
            print(f"Warning: Skipping {organ_name} - missing quality directories")
            continue

        # Get all low-quality images for this organ
        low_quality_images = sorted([f for f in low_quality_dir.glob("*.png") if f.is_file()])

        print(f"\nProcessing {organ_name}: {len(low_quality_images)} images")

        # Match with high-quality images (assuming same filenames)
        for lq_img in low_quality_images:
            hq_img = high_quality_dir / lq_img.name
            if hq_img.exists():
                # Create unique image_id with organ prefix to avoid conflicts
                image_id = f"{organ_name}_{lq_img.stem}"
                image_pairs.append({
                    'image_id': image_id,
                    'organ': organ_name,
                    'low_quality_path': lq_img,
                    'high_quality_path': hq_img,
                    'filename': lq_img.name
                })
            else:
                print(f"Warning: No matching high-quality image for {organ_name}/{lq_img.name}")

    if not image_pairs:
        raise ValueError("No image pairs found! Check the data structure.")

    print(f"\nTotal image pairs found: {len(image_pairs)}")

    # Create train/test split (80/20)
    # Use image_id for splitting to ensure reproducibility
    image_ids = [pair['image_id'] for pair in image_pairs]
    train_ids, test_ids = train_test_split(
        image_ids,
        test_size=0.2,
        random_state=42
    )

    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    print(f"\nSplitting data:")
    print(f"  Training: {len(train_ids)} pairs (80%)")
    print(f"  Test: {len(test_ids)} pairs (20%)")

    # Copy training pairs to public
    print("\nCopying training images to public/train/...")
    for pair in image_pairs:
        if pair['image_id'] in train_ids_set:
            # Create unique filenames with organ prefix
            lq_filename = f"{pair['image_id']}.png"
            hq_filename = f"{pair['image_id']}.png"

            # Copy low-quality image
            shutil.copy(
                pair['low_quality_path'],
                public / "train" / "low_quality" / lq_filename
            )
            # Copy high-quality image
            shutil.copy(
                pair['high_quality_path'],
                public / "train" / "high_quality" / hq_filename
            )

    # Copy test pairs (low-quality to public, high-quality to private)
    print("Copying test images (low-quality to public, high-quality to private)...")
    test_labels = []
    for pair in image_pairs:
        if pair['image_id'] in test_ids_set:
            # Create unique filenames with organ prefix
            lq_filename = f"{pair['image_id']}.png"
            hq_filename = f"{pair['image_id']}.png"

            # Copy low-quality image to public
            shutil.copy(
                pair['low_quality_path'],
                public / "test" / "low_quality" / lq_filename
            )
            # Copy high-quality image to private (ground truth)
            shutil.copy(
                pair['high_quality_path'],
                private / "test" / "high_quality" / hq_filename
            )

            # Record ground truth path
            test_labels.append({
                'image_id': pair['image_id'],
                'ground_truth_path': f"test/high_quality/{hq_filename}"
            })

    # Save test labels to private directory
    test_df = pd.DataFrame(test_labels)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"\nCreated test_labels.csv with {len(test_labels)} entries")

    # Create sample submission
    sample = test_df[['image_id']].copy()
    sample['enhanced_image_path'] = sample['image_id'].apply(
        lambda x: f"enhanced/{x}.png"  # Expected submission format
    )
    sample.to_csv(public / "sample_submission.csv", index=False)
    print(f"Created sample_submission.csv")

    # Copy description.md to public if it exists
    description_path = Path(__file__).parent / "description.md"
    if description_path.exists():
        shutil.copy(description_path, public / "description.md")
        print("Copied description.md")

    # Copy grade.py to public
    grading_path = Path(__file__).parent / "grade.py"
    if grading_path.exists():
        shutil.copy(grading_path, public / "grade.py")
        print("Copied grade.py")

    print("\n✓ Data preparation complete!")

    # Print summary
    print("\nDataset summary:")
    print(f"  Training low-quality images: {len(list((public / 'train' / 'low_quality').glob('*')))}")
    print(f"  Training high-quality images: {len(list((public / 'train' / 'high_quality').glob('*')))}")
    print(f"  Test low-quality images: {len(list((public / 'test' / 'low_quality').glob('*')))}")
    print(f"  Test high-quality images (private): {len(list((private / 'test' / 'high_quality').glob('*')))}")
