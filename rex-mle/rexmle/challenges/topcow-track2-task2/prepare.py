"""
Data preparation script for TopCoW Track 2 Task 2 - MRA 3D Bounding Box Detection

Dataset structure after extraction:
- imagesTr/: Training images (topcow_{ct|mr}_{pat_id}_0000.nii.gz)
- roi_loc_labelsTr/: 3D bounding box annotations
"""

from pathlib import Path
import pandas as pd
import shutil
import json
import re
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def get_shared_raw_dir(raw: Path) -> Path:
    """Get shared raw directory for all TopCoW challenges."""
    return raw.parent.parent / "topcow-shared" / "raw"


def convert_bbox_txt_to_json(txt_path: Path, json_path: Path) -> None:
    """Convert bounding box from .txt format to .json format.

    Input .txt format (3 lines):
    ElementSpacing = 0.488281 0.488281 0.8
    Size = 179 143 85
    Origin = 98 158 104

    Output .json format:
    {"size": [179, 143, 85], "location": [98, 158, 104]}
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Parse size from line 2 (0-indexed line 1)
    size_line = lines[1].strip()
    size_values = re.findall(r"\b\d+\b", size_line)
    size = [int(x) for x in size_values]

    # Parse location from line 3 (0-indexed line 2)
    location_line = lines[2].strip()
    location_values = re.findall(r"\b\d+\b", location_line)
    location = [int(x) for x in location_values]

    # Create JSON object
    bbox_data = {
        "size": size,
        "location": location
    }

    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(bbox_data, f, indent=2)


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
    Prepare TopCoW Track 2 Task 2 data (MRA Detection).

    Expected structure in raw/:
    - imagesTr/: CTA and MRA training images
    - roi_loc_labelsTr/: 3D bounding box annotations
    """
    print(f"Preparing TopCoW Track 2 Task 2 (MRA Detection)...")
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
    bbox_labels_dir = raw / "roi_loc_labelsTr"

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

    if not images_dir.exists() or not bbox_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected imagesTr/ and roi_loc_labelsTr/ directories in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find MRA images only (topcow_mr_*_0000.nii.gz)
    all_mra_images = sorted(images_dir.glob("topcow_mr_*_0000.nii.gz"))
    print(f"\nFound {len(all_mra_images)} MRA images")

    # Match images with bounding box labels
    matched_cases = []
    for img_path in all_mra_images:
        # Extract patient ID: topcow_mr_001_0000.nii.gz -> 001
        case_id = img_path.stem.replace('.nii', '').split('_')[2]

        # Label files don't have _0000 suffix and are .txt: topcow_mr_001.txt
        label_filename = f"topcow_mr_{case_id}.txt"
        bbox_label_path = bbox_labels_dir / label_filename

        if bbox_label_path.exists():
            matched_cases.append({
                'case_id': case_id,
                'image_path': img_path,
                'bbox_label_path': bbox_label_path,
                'modality': 'MRA'
            })
        else:
            print(f"⚠ Warning: No bounding box label found for {img_path.name} (expected {label_filename})")

    print(f"Matched {len(matched_cases)} MRA cases with bounding box labels")

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
        shutil.copy(case['bbox_label_path'], public / "train" / "labels" / case['bbox_label_path'].name)

    # Copy test data
    print("Copying test data...")
    test_metadata = []
    for case in test_cases:
        # Copy image to public (without label)
        shutil.copy(case['image_path'], public / "test" / "images" / case['image_path'].name)

        # Convert bbox label from .txt to .json and save to private
        json_filename = f"topcow_mr_{case['case_id']}_bbox.json"
        json_label_path = private / "test" / "labels" / json_filename
        convert_bbox_txt_to_json(case['bbox_label_path'], json_label_path)

        test_metadata.append({
            'image_id': case['case_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{json_filename}",
            'modality': case['modality']
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    # Format matches Grand Challenge: JSON file with {"size": [x,y,z], "location": [x,y,z]}
    sample_sub = test_df[['image_id', 'modality']].copy()
    sample_sub['predicted_bbox_path'] = sample_sub['image_id'].apply(
        lambda x: f"predictions/topcow_mr_{x}_bbox.json"
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
    readme = f"""# TopCoW Track 2 Task 2: MRA 3D Bounding Box Detection

## Dataset

- **Modality**: MRA (Magnetic Resonance Angiography)
- **Anatomy**: Brain - Circle of Willis vascular structure
- **Format**: NIfTI (.nii.gz)
- **Total MRA cases**: {len(matched_cases)}
- **Training cases**: {len(train_cases)} (with images and bounding box labels)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} MRA scans
└── labels/        # {len(train_cases)} 3D bounding box masks

test/
└── images/        # {len(test_cases)} test images
```

## Task: 3D Bounding Box Detection

Detect the Circle of Willis region with a 3D bounding box.

## File Naming

- Images: `topcow_mr_{{pat_id}}_0000.nii.gz` (e.g., `topcow_mr_001_0000.nii.gz`)
- Labels: `topcow_mr_{{pat_id}}_0000.nii.gz` (binary mask defining bounding box region)

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "001")
   - `modality`: "MRA"
   - `predicted_bbox_path`: Relative path to predicted bounding box mask

2. **predictions/** directory with NIfTI masks:
   - Binary masks indicating detected CoW region
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

- **Primary metric**: 3D IoU (Intersection over Union)
- **Secondary metrics**: Precision, Recall

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
