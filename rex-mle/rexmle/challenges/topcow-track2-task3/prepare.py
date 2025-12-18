"""
Data preparation script for TopCoW Track 2 Task 3 - MRA Graph Edge Classification

Dataset structure after extraction:
- imagesTr/: Training images (topcow_{ct|mr}_{pat_id}_0000.nii.gz)
- antpos_edges_labelsTr/: Graph edge classification labels
"""

from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def get_shared_raw_dir(raw: Path) -> Path:
    """Get shared raw directory for all TopCoW challenges."""
    return raw.parent.parent / "topcow-shared" / "raw"


def download_data(raw_dir: Path) -> None:
    """Download and extract data from Zenodo."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)
    zenodo_config = config['zenodo']

    print(f"Downloading TopCoW dataset from Zenodo: {zenodo_config['record_id']}")

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
    Prepare TopCoW Track 2 Task 3 data (MRA Classification).

    Expected structure in raw/:
    - imagesTr/: CTA and MRA training images
    - antpos_edges_labelsTr/: Graph edge classification labels
    """
    print(f"Preparing TopCoW Track 2 Task 3 (MRA Classification)...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Get shared raw directory
    shared_raw = get_shared_raw_dir(raw)

    # Check if data exists in shared location
    shared_images = shared_raw / "imagesTr"
    if shared_images.exists() and list(shared_images.glob("*.nii.gz")):
        print(f"\nUsing shared TopCoW data from: {shared_raw}")
        raw = shared_raw  # Use shared directory for all subsequent operations
    else:
        # Download to shared location (not challenge-specific raw)
        print(f"\nDownloading to shared location: {shared_raw}")
        if not shared_raw.exists() or not list(shared_raw.glob('*')):
            download_data(shared_raw)
        raw = shared_raw  # Use shared directory

    # Extract if needed
    images_dir = raw / "imagesTr"
    edge_labels_dir = raw / "antpos_edges_labelsTr"

    if not images_dir.exists() and (raw / "TopCoW2024_Data_Release.zip").exists():
        print("\nExtracting TopCoW2024_Data_Release.zip...")
        downloader = ZenodoDownloader()
        downloader.extract_archive(raw / "TopCoW2024_Data_Release.zip", raw)
        print("✓ Extracted")

    # Check if data is in nested subdirectory and move it up
    nested_dir = raw / "TopCoW2024_Data_Release"
    if nested_dir.exists() and not images_dir.exists():
        print(f"\nMoving data from nested directory to raw...")
        for item in nested_dir.iterdir():
            dest = raw / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        # Remove empty nested directory
        if not any(nested_dir.iterdir()):
            nested_dir.rmdir()
        print("✓ Moved data to raw directory")

    if not images_dir.exists() or not edge_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected imagesTr/ and antpos_edges_labelsTr/ directories in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find MRA images only (topcow_mr_*_0000.nii.gz)
    all_mra_images = sorted(images_dir.glob("topcow_mr_*_0000.nii.gz"))
    print(f"\nFound {len(all_mra_images)} MRA images")

    # Match images with edge classification labels
    matched_cases = []
    for img_path in all_mra_images:
        # Extract patient ID: topcow_mr_001_0000.nii.gz -> 001
        case_id = img_path.stem.replace('.nii', '').split('_')[2]

        # Label files don't have _0000 suffix and are .yml: topcow_mr_001.yml
        label_filename = f"topcow_mr_{case_id}.yml"
        edge_label_path = edge_labels_dir / label_filename

        if edge_label_path.exists():
            matched_cases.append({
                'case_id': case_id,
                'image_path': img_path,
                'edge_label_path': edge_label_path,
                'modality': 'MRA'
            })
        else:
            print(f"⚠ Warning: No edge classification label found for {img_path.name} (expected {label_filename})")

    print(f"Matched {len(matched_cases)} MRA cases with edge classification labels")

    if len(matched_cases) == 0:
        raise ValueError("No matched MRA cases found!")

    # Sort by case_id to ensure consistent ordering across systems
    matched_cases = sorted(matched_cases, key=lambda x: x['case_id'])

    # Split train/test (80/20) with fixed random_state for reproducibility
    train_cases, test_cases = train_test_split(
        matched_cases,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Split: {len(train_cases)} train, {len(test_cases)} test")

    # Create output directories
    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)
    (private / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy training data (with labels)
    print("\nCopying training data...")
    for case in train_cases:
        shutil.copy(case['image_path'], public / "train" / "images" / case['image_path'].name)
        shutil.copy(case['edge_label_path'], public / "train" / "labels" / case['edge_label_path'].name)

    # Copy test data
    print("Copying test data...")
    test_metadata = []
    for case in test_cases:
        # Copy image to public (without label)
        shutil.copy(case['image_path'], public / "test" / "images" / case['image_path'].name)

        # Copy label to private
        shutil.copy(case['edge_label_path'], private / "test" / "labels" / case['edge_label_path'].name)

        test_metadata.append({
            'image_id': case['case_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{case['edge_label_path'].name}",
            'modality': case['modality']
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    # Format matches Grand Challenge: JSON file with anterior/posterior edge classifications
    # {"anterior": {"L-A1": 0/1, "Acom": 0/1, "3rd-A2": 0/1, "R-A1": 0/1},
    #  "posterior": {"L-Pcom": 0/1, "L-P1": 0/1, "R-P1": 0/1, "R-Pcom": 0/1}}
    sample_sub = test_df[['image_id', 'modality']].copy()
    sample_sub['predicted_edges_path'] = sample_sub['image_id'].apply(
        lambda x: f"predictions/topcow_mr_{x}_edges.json"
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

    # Create README
    readme = f"""# TopCoW Track 2 Task 3: MRA Graph Edge Classification

## Dataset

- **Modality**: MRA (Magnetic Resonance Angiography)
- **Anatomy**: Brain - Circle of Willis vascular structure
- **Format**: NIfTI (.nii.gz)
- **Total MRA cases**: {len(matched_cases)}
- **Training cases**: {len(train_cases)} (with images and edge classification labels)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} MRA scans
└── labels/        # {len(train_cases)} graph edge classification masks

test/
└── images/        # {len(test_cases)} test images
```

## Task: Graph Edge Classification

Classify topological edges in the Circle of Willis graph structure.

## File Naming

- Images: `topcow_mr_{{pat_id}}_0000.nii.gz` (e.g., `topcow_mr_001_0000.nii.gz`)
- Labels: `topcow_mr_{{pat_id}}_0000.nii.gz` (edge classification mask)

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "MRA"
   - `predicted_edges_path`: Relative path to predicted edge classification mask

2. **predictions/** directory with NIfTI masks:
   - Edge classification masks
   - Same dimensions as input images
   - Format: NIfTI (.nii.gz)

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── topcow_mr_001_0000.nii.gz
    ├── topcow_mr_002_0000.nii.gz
    └── ...
```

## Evaluation

- **Primary metric**: Accuracy
- **Secondary metrics**: F1-score, Precision, Recall

## Citation

TopCoW 2024 Challenge
Zenodo: https://zenodo.org/records/15692630

## License

https://opendata.swiss/en/terms-of-use
"""

    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Training: {len(train_cases)} MRA cases with images and labels")
    print(f"  Testing: {len(test_cases)} MRA cases (labels in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
