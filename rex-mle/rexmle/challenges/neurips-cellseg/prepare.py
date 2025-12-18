"""
Data preparation script for NeurIPS 2022 Cell Segmentation Challenge

Dataset structure after extraction:
- Training-labeled/: Limited labeled patches for supervised training
- train-unlabeled-part1/ & part2/: Large unlabeled image collections
- Tuning/: Validation set for hyperparameter optimization
- Testing/: Test images for final evaluation
"""

from pathlib import Path
import pandas as pd
import shutil
from typing import List, Dict
import zipfile
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def extract_zip_file(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file to a specified directory."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted {zip_path.name}")


def download_data(raw_dir: Path) -> None:
    """Download and extract data from Zenodo."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    zenodo_record_id = "10719375"
    print(f"Downloading NeurIPS Cell Segmentation dataset from Zenodo: {zenodo_record_id}")

    downloader = ZenodoDownloader()
    files = downloader.download_record(
        record_id=zenodo_record_id,
        output_dir=raw_dir,
        files=None  # Download all files
    )

    # Extract archives
    for file in files:
        if file.suffix == '.zip':
            extract_zip_file(file, raw_dir)


def find_image_label_pairs(images_dir: Path, labels_dir: Path,
                            image_suffixes: List[str] = None) -> List[Dict]:
    """
    Find matching image-label pairs.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        image_suffixes: List of valid image file extensions

    Returns:
        List of dicts with image_path and label_path
    """
    if image_suffixes is None:
        image_suffixes = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']

    matched_pairs = []

    # Find all images
    image_files = []
    for suffix in image_suffixes:
        image_files.extend(images_dir.glob(f"*{suffix}"))
        image_files.extend(images_dir.glob(f"*{suffix.upper()}"))

    for img_path in sorted(image_files):
        # Look for corresponding label (typically .tif or .tiff)
        base_name = img_path.stem

        # Try different label naming conventions
        possible_label_names = [
            f"{base_name}.tif",
            f"{base_name}.tiff",
            f"{base_name}_label.tif",
            f"{base_name}_label.tiff",
        ]

        label_path = None
        for label_name in possible_label_names:
            potential_label = labels_dir / label_name
            if potential_label.exists():
                label_path = potential_label
                break

        if label_path:
            matched_pairs.append({
                'image_id': base_name,
                'image_path': img_path,
                'label_path': label_path
            })

    return matched_pairs


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare NeurIPS Cell Segmentation Challenge data.

    Expected structure in raw/ after download and extraction:
    - Training-labeled/: Labeled training patches
    - train-unlabeled-part1/: Unlabeled images (part 1)
    - train-unlabeled-part2/: Unlabeled images (part 2)
    - Tuning/: Validation/tuning set
    - Testing/: Test images
    """
    print(f"Preparing NeurIPS Cell Segmentation Challenge...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if not present
    if not raw.exists() or not list(raw.glob('*.zip')):
        download_data(raw)

    # Extract archives if needed
    required_dirs = ["Training-labeled", "Tuning", "Testing"]
    all_exist = all((raw / d).exists() for d in required_dirs)

    if not all_exist:
        print("\nExtracting dataset archives...")
        for zip_file in raw.glob("*.zip"):
            if not zip_file.stem in ["Training-labeled", "Tuning", "Testing",
                                      "train-unlabeled-part1", "train-unlabeled-part2"]:
                continue
            extract_to = raw / zip_file.stem
            if not extract_to.exists():
                extract_zip_file(zip_file, raw)

    # Verify directories exist
    labeled_train_dir = raw / "Training-labeled"
    tuning_dir = raw / "Tuning"
    test_dir = raw / "Testing"

    if not labeled_train_dir.exists():
        raise FileNotFoundError(f"Training-labeled directory not found in {raw}")
    if not tuning_dir.exists():
        raise FileNotFoundError(f"Tuning directory not found in {raw}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found in {raw}")

    # Find labeled training data
    # Typically structure: Training-labeled/images/ and Training-labeled/labels/
    train_images_dir = labeled_train_dir / "images" if (labeled_train_dir / "images").exists() else labeled_train_dir
    train_labels_dir = labeled_train_dir / "labels" if (labeled_train_dir / "labels").exists() else labeled_train_dir

    print(f"\nScanning for labeled training data...")
    train_pairs = find_image_label_pairs(train_images_dir, train_labels_dir)
    print(f"Found {len(train_pairs)} labeled training image-label pairs")

    # Find tuning/validation data
    print(f"\nScanning for tuning data...")
    tuning_images_dir = tuning_dir / "images" if (tuning_dir / "images").exists() else tuning_dir
    tuning_labels_dir = tuning_dir / "labels" if (tuning_dir / "labels").exists() else tuning_dir

    tuning_pairs = find_image_label_pairs(tuning_images_dir, tuning_labels_dir)
    print(f"Found {len(tuning_pairs)} tuning image-label pairs")

    # Combine all labeled data for train/test split
    print(f"\nCombining all labeled data...")
    all_pairs = train_pairs + tuning_pairs
    print(f"Total labeled pairs: {len(all_pairs)}")

    if len(all_pairs) == 0:
        raise ValueError("No labeled image-label pairs found!")

    # Sort by image_id to ensure consistent ordering across systems
    all_pairs = sorted(all_pairs, key=lambda x: x['image_id'])

    # Split train/test (80/20) with fixed random_state for reproducibility
    # All users will get the exact same split
    train_pairs_split, test_pairs_split = train_test_split(
        all_pairs,
        test_size=0.2,
        random_state=42,
        shuffle=True  # Explicit shuffle with fixed seed
    )

    print(f"Split: {len(train_pairs_split)} train, {len(test_pairs_split)} test")

    # Create output directories
    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)
    (private / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy training data (with labels)
    print("\nCopying training data...")
    for pair in train_pairs_split:
        shutil.copy(pair['image_path'], public / "train" / "images" / pair['image_path'].name)
        shutil.copy(pair['label_path'], public / "train" / "labels" / pair['label_path'].name)

    # Copy test data
    print("Copying test data...")
    test_metadata = []
    for pair in test_pairs_split:
        # Copy image to public (without label)
        shutil.copy(pair['image_path'], public / "test" / "images" / pair['image_path'].name)

        # Copy label to private
        shutil.copy(pair['label_path'], private / "test" / "labels" / pair['label_path'].name)

        test_metadata.append({
            'image_id': pair['image_id'],
            'image_path': f"test/images/{pair['image_path'].name}",
            'label_path': f"test/labels/{pair['label_path'].name}"
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test metadata: {len(test_df)} cases")

    # Create sample submission
    if len(test_df) > 0:
        sample_sub = test_df[['image_id']].copy()
        sample_sub['predicted_mask_path'] = sample_sub['image_id'].apply(
            lambda x: f"predictions/{x}_label.tiff"
        )
    else:
        # Create empty sample submission with proper columns
        sample_sub = pd.DataFrame(columns=['image_id', 'predicted_mask_path'])
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    # Copy description.md to public
    description_path = Path(__file__).parent / "description.md"
    if description_path.exists():
        shutil.copy(description_path, public / "description.md")
        print("Copied description.md")

    # Copy grade.py to public
    grading_path = Path(__file__).parent / "grade.py"
    if grading_path.exists():
        shutil.copy(grading_path, public / "grade.py")
        print("Copied grade.py")

    # Create README
    readme = f"""# NeurIPS 2022 Cell Segmentation Challenge

## Dataset

- **Task**: Weakly Supervised Cell Instance Segmentation
- **Modalities**: Brightfield, Fluorescent, Phase-contrast (PC), DIC microscopy
- **Format**: Various (TIFF, PNG, JPG, BMP)
- **Total annotated**: {len(all_pairs)} cases
- **Training cases**: {len(train_pairs_split)} (with images and labels)
- **Test cases**: {len(test_pairs_split)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_pairs_split)} labeled training images
└── labels/        # {len(train_pairs_split)} instance segmentation masks

test/
└── images/        # {len(test_pairs_split)} test images
```

## Task: Weakly Supervised Instance Segmentation

Segment individual cells across diverse microscopy modalities using limited labeled data and optional unlabeled images.

## Label Format

- **Instance segmentation masks** where:
  - 0 = background
  - 1, 2, 3, ... = individual cell instances
- Format: TIFF (recommended) or PNG

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Image identifier (filename without extension)
   - `predicted_mask_path`: Relative path to predicted segmentation mask

2. **predictions/** directory with instance masks:
   - Instance masks with individual cell labels
   - Same dimensions as input images
   - Format: TIFF or PNG

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── image_001_label.tiff
    ├── image_002_label.tiff
    └── ...
```

## Evaluation

- **Primary metric**: F1 Score (IoU threshold = 0.5)
- **Additional thresholds**: 0.6, 0.7, 0.8, 0.9
- **Matching**: Hungarian algorithm for optimal instance matching
- **Boundary handling**: Cells touching image borders (within 2 pixels) are excluded

## Weakly Supervised Setting

While only labeled patches are provided in this benchmark, the original challenge encouraged:
- Semi-supervised learning with unlabeled images
- Self-supervised pre-training
- Few-shot learning approaches

## Citation

```
@article{{neurips2022cellseg,
  title={{NeurIPS 2022 Cell Segmentation Challenge}},
  year={{2022}},
  note={{Zenodo record: 10719375}}
}}
```

## License

CC-BY-NC-ND

## Contact

neurips.cellseg@gmail.com
"""

    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Training: {len(train_pairs_split)} cases with images and labels")
    print(f"  Testing: {len(test_pairs_split)} cases (labels in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
