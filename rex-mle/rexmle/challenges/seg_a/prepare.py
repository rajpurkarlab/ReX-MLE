"""
Data preparation script for SEG.A. 2023 - Segmentation of the Aortic Vessel Tree

Dataset structure after extraction:
- AVT dataset from Figshare with 56 CTA scans
- Images and segmentation masks in various formats (likely NIfTI or DICOM)
- Binary segmentation of aortic vessel tree

This script:
1. Downloads data from Figshare
2. Organizes CTA images and segmentation masks
3. Creates train/test splits
4. Generates sample submission
"""

from pathlib import Path
import pandas as pd
import shutil
import requests
import zipfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rexmle.utils import load_yaml


def download_data(raw_dir: Path) -> None:
    """
    Download AVT dataset from Figshare.

    Args:
        raw_dir: Directory to save raw data
    """
    # Figshare direct download URL
    figshare_url = "https://figshare.com/ndownloader/articles/14806362/versions/1"

    print(f"Downloading SEG.A. AVT dataset from Figshare...")
    print(f"URL: {figshare_url}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download the zip file
    zip_path = raw_dir / "avt_dataset.zip"

    if not zip_path.exists():
        print("Downloading dataset (this may take a while)...")
        response = requests.get(figshare_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✓ Downloaded: {zip_path.name}")
    else:
        print(f"Dataset already downloaded: {zip_path.name}")

    # Extract the zip file
    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print(f"✓ Extracted to: {raw_dir}")

        # Extract nested zip files (Dongyang.zip, KiTS.zip, Rider.zip)
        nested_zips = list(raw_dir.glob('*.zip'))
        for nested_zip in nested_zips:
            if nested_zip.name != zip_path.name:  # Don't re-extract the main archive
                print(f"Extracting nested archive: {nested_zip.name}...")
                extract_dir = raw_dir / nested_zip.stem
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"✓ Extracted: {nested_zip.name}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare SEG.A. challenge data.

    Expected structure in raw/:
    - CTA images and corresponding segmentation masks
    - Dataset contains 56 cases from multiple institutions

    Args:
        raw: Path to raw data directory (downloaded from Figshare)
        public: Path to public data directory (visible to participants)
        private: Path to private data directory (held out for grading)
    """
    print(f"Preparing SEG.A. - Segmentation of the Aortic Vessel Tree...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if raw directory is empty or doesn't exist
    if not raw.exists() or not list(raw.glob('*')):
        print("\nDownloading data from Figshare...")
        download_data(raw)

    # Find the extracted data directory
    # The dataset might be in various formats, look for common medical imaging formats
    potential_dirs = []
    for pattern in ['images*', 'data*', 'AVT*', 'aorta*', '*Tr*']:
        potential_dirs.extend(list(raw.glob(pattern)))

    # Look for .nrrd image files (excluding .seg.nrrd which are segmentations)
    all_nrrd_files = list(raw.rglob('*.nrrd'))

    # Separate images from segmentations
    image_files = [f for f in all_nrrd_files if '.seg.nrrd' not in f.name]
    seg_files = [f for f in all_nrrd_files if '.seg.nrrd' in f.name]

    print(f"\nFound {len(image_files)} image files (.nrrd)")
    print(f"Found {len(seg_files)} segmentation files (.seg.nrrd)")

    if len(image_files) == 0:
        print(f"\nSearching for directories in {raw}:")
        print(f"  Found: {list(raw.glob('*'))}")
        raise FileNotFoundError(
            f"No .nrrd image files found in {raw}\n"
            f"Please check if the nested archives were extracted correctly."
        )

    # Match images with their segmentation masks
    images_with_labels = []

    for img_path in image_files:
        # For each image (e.g., D1.nrrd), look for corresponding segmentation (D1.seg.nrrd)
        expected_seg_name = img_path.stem + '.seg.nrrd'
        seg_path = img_path.parent / expected_seg_name

        if seg_path.exists():
            # Extract case ID from path
            # Path structure: .../Dongyang/Dongyang/D1/D1.nrrd
            # We use the shortened case name (e.g., "D1") as the case ID
            case_id = img_path.stem  # e.g., "D1", "K13", "R17"
            institution = img_path.parent.parent.name  # e.g., "Dongyang"

            images_with_labels.append({
                'case_id': case_id,
                'image_path': img_path,
                'label_path': seg_path,
                'institution': institution
            })
        else:
            print(f"⚠ Warning: No segmentation found for {img_path.name} (expected: {expected_seg_name})")

    print(f"Matched {len(images_with_labels)} cases with labels")

    if len(images_with_labels) == 0:
        raise ValueError(
            "No image-label pairs found!\n"
            "Please check the dataset structure and ensure images have corresponding labels."
        )

    # Sort by case_id for consistent ordering
    images_with_labels = sorted(images_with_labels, key=lambda x: x['case_id'])

    # Split train/test (80/20) - test set represents the 4th institution
    train_cases, test_cases = train_test_split(
        images_with_labels,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Split: {len(train_cases)} train (3 institutions), {len(test_cases)} test (4th institution)")

    # Create output directories
    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)
    (private / "test" / "labels").mkdir(parents=True, exist_ok=True)

    # Copy training data (with labels)
    print("\nCopying training data...")
    for case in train_cases:
        img_dest = public / "train" / "images" / case['image_path'].name
        label_dest = public / "train" / "labels" / case['label_path'].name
        shutil.copy(case['image_path'], img_dest)
        shutil.copy(case['label_path'], label_dest)

    # Copy test data
    print("Copying test data...")
    test_metadata = []
    for case in test_cases:
        # Copy image to public (without label)
        img_dest = public / "test" / "images" / case['image_path'].name
        shutil.copy(case['image_path'], img_dest)

        # Copy label to private
        label_dest = private / "test" / "labels" / case['label_path'].name
        shutil.copy(case['label_path'], label_dest)

        test_metadata.append({
            'image_id': case['case_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{case['label_path'].name}",
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    sample_sub = test_df[['image_id']].copy()
    sample_sub['predicted_mask_path'] = sample_sub['image_id'].apply(
        lambda x: f"predictions/{x}.seg.nrrd"
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
    # Get institution breakdown
    institutions = {}
    for case in images_with_labels:
        inst = case['institution']
        institutions[inst] = institutions.get(inst, 0) + 1

    institution_info = ", ".join([f"{inst}: {count}" for inst, count in institutions.items()])

    readme = f"""# SEG.A. 2023 - Segmentation of the Aortic Vessel Tree

## Dataset

- **Challenge**: SEG.A. 2023 (MICCAI 2023)
- **Modality**: CTA (Computed Tomography Angiography)
- **Anatomy**: Aortic Vessel Tree (AVT)
- **Format**: NRRD (.nrrd)
- **Task Type**: Binary Segmentation
- **Total cases**: {len(images_with_labels)} ({institution_info})
- **Training cases**: {len(train_cases)} (with images and labels)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} CTA scans (.nrrd)
└── labels/        # {len(train_cases)} binary segmentation masks (.seg.nrrd)

test/
└── images/        # {len(test_cases)} test CTA scans (.nrrd)
```

## Task: Aortic Vessel Tree Segmentation

Automatically segment the complete aortic vessel tree (AVT) including:
- Main aorta (ascending, arch, descending, abdominal)
- Branch arteries
- Iliac arteries

The challenge evaluates algorithms across different:
- Scanning protocols
- Scanning devices
- Radiation doses
- Contrast agents
- Clinical institutions ({", ".join(institutions.keys())})

## Dataset Composition

- Multiple institutions: {", ".join(institutions.keys())}
- Mostly healthy aortas
- Some cases with abdominal aortic aneurysm (AAA)
- Some cases with aortic dissections (ADs)

## File Naming Convention

- Images: `{{case_id}}.nrrd` (e.g., `D1.nrrd`, `K13.nrrd`, `R17.nrrd`)
- Segmentations: `{{case_id}}.seg.nrrd` (e.g., `D1.seg.nrrd`, `K13.seg.nrrd`, `R17.seg.nrrd`)
- Case ID prefixes indicate institution: D* = Dongyang, K* = KiTS, R* = Rider

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Case identifier (e.g., "D1", "K13", "R17")
   - `predicted_mask_path`: Relative path to predicted segmentation mask

2. **predictions/** directory with segmentation masks:
   - Binary masks (0=background, 1=aortic vessel tree)
   - Same dimensions and spacing as input images
   - Format: NRRD (.seg.nrrd)

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── D1.seg.nrrd
    ├── K13.seg.nrrd
    └── ...
```

## Evaluation Metrics

- **Primary metric**: Dice Similarity Coefficient (DSC)
- **Secondary metric**: Hausdorff Distance (HD)

The reconstruction should be artifact-free for:
- Clinical visualization
- Blood flow simulation
- Computational fluid dynamics

## Special Recognition

SEG.A. 2023 received special mention in MICCAI challenge highlights for "New Analysis Method"

## Citation

AVT: Multicenter aortic vessel tree CTA dataset collection with ground truth segmentation masks
Radl, Lukas, et al. Data in Brief 40 (2022): 107801.
https://doi.org/10.1016/j.dib.2022.107801

Figshare: https://figshare.com/articles/dataset/14806362

## License

Creative Commons BY 4.0
"""

    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Total cases: {len(images_with_labels)} from {len(institutions)} institutions")
    print(f"  Institutions: {institution_info}")
    print(f"  Training: {len(train_cases)} cases with images and labels")
    print(f"  Testing: {len(test_cases)} cases (labels in private/)")
    print(f"  Image format: NRRD (.nrrd)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
