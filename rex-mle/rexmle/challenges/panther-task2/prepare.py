"""
Data preparation script for PANTHER Task 2 - MR-Linac MRI

Actual structure after extraction:
- ImagesTr/: 50 T2-weighted MR-Linac MHA files (*_0000.mha)
- LabelsTr/: 50 tumor segmentation masks (*.mha)
- Naming: 10303_0000.mha -> 10303.mha
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
    Prepare PANTHER Task 2 data.
    
    Actual structure:
    - ImagesTr/: 50 T2-weighted MR-Linac MHA files (10303_0000.mha)
    - LabelsTr/: 50 labels (10303.mha)
    """
    print(f"Preparing PANTHER Task 2 data...")
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
    
    if not images_dir.exists() and (raw / "PANTHER_Task2.zip").exists():
        print("\nExtracting PANTHER_Task2.zip...")
        downloader = ZenodoDownloader()
        downloader.extract_archive(raw / "PANTHER_Task2.zip", raw)
        print("✓ Extracted")

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected ImagesTr/ and LabelsTr/ directories in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find MR-Linac images
    mrlinac_images = sorted(images_dir.glob("*_0000.mha"))
    print(f"\nFound {len(mrlinac_images)} MR-Linac images")

    # Match images with labels
    matched_cases = []
    for img_path in mrlinac_images:
        # Extract patient ID: 10303_0000.mha -> 10303
        patient_id = img_path.stem.split('_')[0]
        
        # Find corresponding label
        label_path = labels_dir / f"{patient_id}.mha"
        
        if label_path.exists():
            matched_cases.append({
                'patient_id': patient_id,
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
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)
    (private / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy training data (with labels)
    print("\nCopying training data...")
    for case in train_cases:
        shutil.copy(case['image_path'], public / "train" / "images" / case['image_path'].name)
        shutil.copy(case['label_path'], public / "train" / "labels" / case['label_path'].name)

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

    # Create README
    readme = f"""# PANTHER Task 2: MR-Linac Adaptive Radiotherapy

## Dataset

- **Modality**: T2-weighted MRI
- **Scanner**: Elekta Unity MR-Linac system
- **Format**: MHA (.mha)
- **Total annotated**: 50 cases
- **Training cases**: {len(train_cases)} (with images and labels)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Important Note

⚠️ **Inference time is critical for this task!** MR-Linac requires real-time or near real-time 
segmentation for adaptive radiotherapy planning.

## Directory Structure

```
train/
├── images/        # {len(train_cases)} T2W MR-Linac scans
└── labels/        # {len(train_cases)} tumor segmentation masks

test/
└── images/        # {len(test_cases)} test images
```

## File Naming

- Images: `PATIENTID_0000.mha` (e.g., `10303_0000.mha`)
- Labels: `PATIENTID.mha` (e.g., `10303.mha`)

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Patient ID (e.g., "10303")
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
    ├── 10303.mha
    ├── 10304.mha
    └── ...
```

## Evaluation

- **Primary metric**: Dice Similarity Coefficient (DSC)
- **Secondary metrics**: Surface Dice (5mm), MASD, HD95, RMSE on tumor burden
- **Additional consideration**: Inference time (critical for clinical use)

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
    print(f"\n⚠️  Note: Inference time is critical for this MR-Linac task!")
