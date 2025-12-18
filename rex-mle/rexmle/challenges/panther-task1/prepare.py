"""
Data preparation script for PANTHER Task 1 - Diagnostic MRI

Actual structure after extraction:
- ImagesTr/: 92 arterial phase MHA files (*_0001_0000.mha)
- LabelsTr/: 92 tumor segmentation masks (*_0001.mha)
- Naming: 10000_0001_0000.mha -> 10000_0001.mha
"""

from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def download_data(raw_dir: Path) -> None:
    """Download and extract data from Zenodo."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)
    zenodo_config = config['zenodo']

    print(f"Downloading PANTHER dataset from Zenodo: {zenodo_config['record_id']}")
    
    downloader = ZenodoDownloader()
    files = downloader.download_record(
        record_id=zenodo_config['record_id'],
        output_dir=raw_dir,
        files=zenodo_config.get('files')
    )

    # Extract archives
    for file in files:
        if file.suffix == '.zip' or '.tar.gz' in file.name:
            print(f"Extracting: {file.name}")
            downloader.extract_archive(file, raw_dir)
            print(f"✓ Extracted: {file.stem}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare PANTHER Task 1 data.
    
    Actual structure:
    - ImagesTr/: 92 arterial phase MHA files (10000_0001_0000.mha)
    - LabelsTr/: 92 labels (10000_0001.mha)
    """
    print(f"Preparing PANTHER Task 1 data...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download if needed
    if not raw.exists() or not list(raw.glob('*')):
        print("\nDownloading from Zenodo...")
        download_data(raw)

    # Extract if needed
    images_dir = raw / "ImagesTr"
    labels_dir = raw / "LabelsTr"
    
    if not images_dir.exists() and (raw / "PANTHER_Task1.zip").exists():
        print("\nExtracting PANTHER_Task1.zip...")
        downloader = ZenodoDownloader()
        downloader.extract_archive(raw / "PANTHER_Task1.zip", raw)
        print("✓ Extracted")

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected ImagesTr/ and LabelsTr/ directories in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find arterial phase images
    arterial_images = sorted(images_dir.glob("*_0001_0000.mha"))
    print(f"\nFound {len(arterial_images)} arterial phase images")

    # Match images with labels
    matched_cases = []
    for img_path in arterial_images:
        # Extract ID: 10000_0001_0000.mha -> 10000_0001
        img_id = '_'.join(img_path.stem.split('_')[:2])  # "10000_0001"
        
        # Find corresponding label
        label_path = labels_dir / f"{img_id}.mha"
        
        if label_path.exists():
            matched_cases.append({
                'patient_id': img_id,
                'image_path': img_path,
                'label_path': label_path
            })
        else:
            print(f"⚠ Warning: No label found for {img_path.name}")

    print(f"Matched {len(matched_cases)} cases with labels")

    if len(matched_cases) == 0:
        raise ValueError("No matched cases found!")

    # Sort by patient_id to ensure consistent ordering across systems
    matched_cases = sorted(matched_cases, key=lambda x: x['patient_id'])

    # Split train/test (80/20) with fixed random_state for reproducibility
    # All users will get the exact same split
    train_cases, test_cases = train_test_split(
        matched_cases,
        test_size=0.2,
        random_state=42,
        shuffle=True  # Explicit shuffle with fixed seed
    )

    print(f"Split: {len(train_cases)} train, {len(test_cases)} test")

    # Create output directories
    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (public / "train_unlabeled" / "images").mkdir(parents=True, exist_ok=True)
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)
    (private / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy training data (with labels)
    print("\nCopying training data...")
    for case in train_cases:
        shutil.copy(case['image_path'], public / "train" / "images" / case['image_path'].name)
        shutil.copy(case['label_path'], public / "train" / "labels" / case['label_path'].name)

    # Copy unlabeled training data (for semi-supervised learning)
    # Note: These are different MRI sequences (not arterial phase)
    unlabeled_dir = raw / "ImagesTr_unlabeled"
    if unlabeled_dir.exists():
        print("\nCopying unlabeled training data...")
        unlabeled_images = sorted(unlabeled_dir.glob("*.mha"))
        for img_path in unlabeled_images:
            shutil.copy(img_path, public / "train_unlabeled" / "images" / img_path.name)
        print(f"✓ Copied {len(unlabeled_images)} unlabeled images (various MRI sequences)")

    # Copy test data
    print("Copying test data...")
    test_metadata = []
    for case in test_cases:
        # Copy image to public (without label)
        shutil.copy(case['image_path'], public / "test" / "images" / case['image_path'].name)
        
        # Copy label to private
        shutil.copy(case['label_path'], private / "test" / "labels" / case['label_path'].name)
        
        test_metadata.append({
            'image_id': case['patient_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{case['label_path'].name}"
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    sample_sub = test_df[['image_id']].copy()
    sample_sub['predicted_mask_path'] = sample_sub['image_id'].apply(
        lambda x: f"predictions/{x}.mha"
    )
    sample_sub.to_csv(public / "sample_submission.csv", index=False)

    # Copy leaderboard
    leaderboard_path = Path(__file__).parent / "leaderboard.csv"
    if leaderboard_path.exists():
        shutil.copy(leaderboard_path, public / "leaderboard.csv")
        print("Copied leaderboard.csv")

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

    # Count unlabeled images
    num_unlabeled = len(list((public / "train_unlabeled" / "images").glob("*.mha"))) if (public / "train_unlabeled" / "images").exists() else 0

    # Create README
    readme = f"""# PANTHER Task 1: Diagnostic MRI Segmentation

## Dataset

- **Modality**: T1-weighted contrast-enhanced arterial phase MRI
- **Scanner**: Siemens diagnostic scanners
- **Format**: MHA (.mha)
- **Total annotated**: 92 cases
- **Total unlabeled**: {num_unlabeled} cases
- **Training cases**: {len(train_cases)} (with images and labels)
- **Training unlabeled**: {num_unlabeled} (images only, for semi-supervised learning)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} arterial phase MRI scans
└── labels/        # {len(train_cases)} tumor segmentation masks

train_unlabeled/
└── images/        # {num_unlabeled} unlabeled MRI scans (for semi-supervised learning)

test/
└── images/        # {len(test_cases)} test images
```

## Using Unlabeled Data

The `train_unlabeled/` directory contains {num_unlabeled} additional MRI scans without labels.

**Important**: These are different MRI sequences (e.g., T2, diffusion-weighted, delayed phase)
from the same or different patients, NOT the arterial phase used for labeled training.

These can be used for:

- **Multi-sequence learning**: Training models that use multiple MRI sequences
- **Semi-supervised learning**: Methods like pseudo-labeling, consistency regularization
- **Self-supervised pre-training**: Contrastive learning, masked autoencoders
- **Transfer learning**: Pre-training on diverse MRI data
- **Domain adaptation**: Understanding MRI data distribution

Example usage:
```python
from pathlib import Path

# Load labeled data (arterial phase only)
labeled_dir = Path("train/images")
labeled_images = list(labeled_dir.glob("*_0001_0000.mha"))  # Arterial phase

# Load unlabeled data (various sequences)
unlabeled_dir = Path("train_unlabeled/images")
unlabeled_images = list(unlabeled_dir.glob("*.mha"))

print(f"Labeled (arterial): {{len(labeled_images)}}")
print(f"Unlabeled (various sequences): {{len(unlabeled_images)}}")

# Example: Pre-train on all MRI data
all_images = labeled_images + unlabeled_images
print(f"Total images for pre-training: {{len(all_images)}}")
```

## File Naming

- Images: `PATIENTID_0001_0000.mha` (e.g., `10000_0001_0000.mha`)
- Labels: `PATIENTID_0001.mha` (e.g., `10000_0001.mha`)

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID with sequence (e.g., "10000_0001")
   - `predicted_mask_path`: Relative path to predicted mask

2. **predictions/** directory with MHA masks:
   - Binary masks: 0 = background, 1 = tumor
   - Same dimensions as input images
   - Format: MHA (.mha)

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── 10000_0001.mha
    ├── 10001_0001.mha
    └── ...
```

## Evaluation

- **Primary metric**: Dice Similarity Coefficient (DSC)
- **Secondary metrics**: Surface Dice (5mm), MASD, HD95, RMSE on tumor burden

## Citation

Betancourt Tarifa, A. S., Mahmood, F., Bernchou, U., & Koopmans, P. J. (2025). 
PANTHER Challenge: Public Training Dataset (1.0) [Data set]. 
Zenodo. https://doi.org/10.5281/zenodo.15192302

## License

CC-BY-NC-SA 4.0
"""
    
    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Training: {len(train_cases)} cases with images and labels")
    print(f"  Testing: {len(test_cases)} cases (labels in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
