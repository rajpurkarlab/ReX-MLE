"""
Data preparation script for DENTEX Challenge

This script:
1. Downloads data from Zenodo (training_data.zip)
2. Extracts and organizes panoramic X-ray images
3. Splits training_data.zip into 80% train / 20% test
4. Processes COCO-format JSON annotations
5. Generates sample submission
"""

from pathlib import Path
import pandas as pd
import json
import shutil
import random

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

    print(f"Downloading DENTEX data from Zenodo record: {zenodo_config['record_id']}")

    downloader = ZenodoDownloader()

    # Download files (training_data.zip and validation_data.zip)
    files = downloader.download_record(
        record_id=zenodo_config['record_id'],
        output_dir=raw_dir,
        files=zenodo_config.get('files')  # Download specific files if specified
    )

    # Extract archives
    for file in files:
        if file.suffix in ['.zip', '.tar', '.gz', '.tgz']:
            print(f"Extracting: {file.name}")
            downloader.extract_archive(file, raw_dir)
            print(f"✓ Extracted: {file.name}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare DENTEX challenge data.

    This function organizes the DENTEX dataset:
    - training_data.zip contains 705 training images with annotations
    - Splits training data into 80% train / 20% test

    Args:
        raw: Path to raw data directory (downloaded from Zenodo)
        public: Path to public data directory (visible to participants)
        private: Path to private data directory (held out for grading)
    """
    print(f"Preparing DENTEX challenge data...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if raw directory is empty or doesn't exist
    if not raw.exists() or not list(raw.glob('*')):
        print("\nDownloading data from Zenodo...")
        download_data(raw)

    # Check if archives need extraction
    training_zip = raw / "training_data.zip"

    if training_zip.exists() and not (raw / "training_data").exists():
        print("\nExtracting training_data.zip...")
        downloader = ZenodoDownloader()
        downloader.extract_archive(training_zip, raw)

    # Define paths to extracted data
    training_dir = raw / "training_data"

    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {training_dir}\n"
            f"Available in raw: {list(raw.glob('*'))}"
        )

    print(f"\nFound training data: {training_dir}")

    # Create output directories
    (public / "train").mkdir(parents=True, exist_ok=True)
    (public / "test").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Load and Split Training Data (80/20)
    # ===================================================================

    print("\nLoading training data...")

    # Find training images and annotations
    # Expected structure: training_data/quadrant-enumeration-disease/
    train_subdir = training_dir / "quadrant-enumeration-disease"
    if not train_subdir.exists():
        # Try direct structure
        train_subdir = training_dir

    # Find all PNG/JPG images in training
    all_images = list(train_subdir.rglob("*.png")) + list(train_subdir.rglob("*.jpg"))

    # Find training annotation JSON file
    json_files = list(train_subdir.rglob("*.json"))

    print(f"Found {len(all_images)} total images")
    print(f"Found {len(json_files)} annotation files")

    # Load COCO annotations to split by image IDs
    if not json_files:
        raise FileNotFoundError("No annotation JSON files found")

    # Use the first JSON file (should be the main annotations file)
    annotations_file = json_files[0]
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Get all image IDs and shuffle for random split
    all_image_ids = [img['id'] for img in coco_data['images']]
    random.seed(42)  # For reproducibility
    random.shuffle(all_image_ids)

    # Split 80/20
    split_idx = int(len(all_image_ids) * 0.8)
    train_image_ids = set(all_image_ids[:split_idx])
    test_image_ids = set(all_image_ids[split_idx:])

    print(f"\nSplitting data:")
    print(f"  Train: {len(train_image_ids)} images (80%)")
    print(f"  Test: {len(test_image_ids)} images (20%)")

    # Create image ID to filename mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # ===================================================================
    # Process Training Split
    # ===================================================================

    print("\nProcessing training split...")

    # Create train annotations
    train_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in train_image_ids],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids],
    }

    # Copy category fields (may be 'categories' or 'categories_1', 'categories_2', 'categories_3')
    for key in coco_data.keys():
        if key.startswith('categories'):
            train_coco[key] = coco_data[key]

    # Copy training images to public
    train_img_dir = public / "train" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)

    train_images = []
    for img_id in train_image_ids:
        filename = image_id_to_filename[img_id]
        # Find the actual file
        img_files = [p for p in all_images if p.name == filename]
        if img_files:
            img_path = img_files[0]
            shutil.copy(img_path, train_img_dir / img_path.name)
            train_images.append(img_path)

    # Save training annotations to public
    train_json_path = public / "train" / annotations_file.name
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)

    print(f"  Copied {len(train_images)} training images")
    print(f"  Saved training annotations with {len(train_coco['annotations'])} annotations")

    # ===================================================================
    # Process Test Split
    # ===================================================================

    print("\nProcessing test split...")

    # Create test annotations
    test_coco = {
        'images': [img for img in coco_data['images'] if img['id'] in test_image_ids],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids],
    }

    # Copy category fields (may be 'categories' or 'categories_1', 'categories_2', 'categories_3')
    for key in coco_data.keys():
        if key.startswith('categories'):
            test_coco[key] = coco_data[key]

    # Copy test images to public (without annotations)
    test_img_dir = public / "test" / "images"
    test_img_dir.mkdir(parents=True, exist_ok=True)

    test_images = []
    for img_id in test_image_ids:
        filename = image_id_to_filename[img_id]
        # Find the actual file
        img_files = [p for p in all_images if p.name == filename]
        if img_files:
            img_path = img_files[0]
            shutil.copy(img_path, test_img_dir / img_path.name)
            test_images.append(img_path)

    # Save test annotations to private as individual JSON files (one per image)
    print(f"  Copied {len(test_images)} test images")
    print(f"  Creating individual JSON files for each test image...")

    # Create a directory for ground truth JSON files
    gt_json_dir = private / "ground_truth"
    gt_json_dir.mkdir(parents=True, exist_ok=True)

    # Create image_id to image mapping for easier lookup
    image_id_to_info = {img['id']: img for img in test_coco['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in test_coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Create individual JSON files for each test image
    test_metadata = []
    for img_path in test_images:
        image_id_str = img_path.stem  # Filename without extension

        # Find the numeric image_id
        numeric_img_id = None
        for img_id, img_info in image_id_to_info.items():
            if img_info['file_name'] == img_path.name:
                numeric_img_id = img_id
                break

        if numeric_img_id is None:
            print(f"Warning: Could not find image_id for {img_path.name}")
            continue

        # Create individual JSON file with annotations for this image
        image_gt = {
            'images': [image_id_to_info[numeric_img_id]],
            'annotations': annotations_by_image.get(numeric_img_id, []),
        }

        # Copy category fields
        for key in test_coco.keys():
            if key.startswith('categories'):
                image_gt[key] = test_coco[key]

        # Save individual JSON file
        json_filename = f"{image_id_str}.json"
        json_path = gt_json_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(image_gt, f, indent=2)

        # Add to metadata
        test_metadata.append({
            'image_id': image_id_str,
            'ground_truth_json': f"ground_truth/{json_filename}",
        })

    print(f"  Created {len(test_metadata)} individual ground truth JSON files")

    # ===================================================================
    # Create Test Labels CSV for Grading
    # ===================================================================

    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"\nSaved test labels CSV: {len(test_df)} images")

    # ===================================================================
    # Create Sample Submission
    # ===================================================================

    # Sample submission format for object detection
    # Format: image_id, predicted_annotations_path
    sample_sub = test_df[['image_id']].copy()
    sample_sub['predictions_json'] = sample_sub['image_id'].apply(
        lambda x: f"predictions/{x}.json"
    )
    sample_sub.to_csv(public / "sample_submission.csv", index=False)
    print(f"Created sample_submission.csv with {len(sample_sub)} entries")

    # ===================================================================
    # Copy Additional Files
    # ===================================================================

    # Copy description.md to public
    description_path = Path(__file__).parent / "description.md"
    if description_path.exists():
        shutil.copy(description_path, public / "description.md")
        print("Copied description.md")

    # Copy description.md to public
    grading_path = Path(__file__).parent / "grade.py"
    if grading_path.exists():
        shutil.copy(grading_path, public / "grade.py")
        print("Copied grade.py")

    # Create README
    readme = f"""# DENTEX Challenge - Dental Enumeration and Diagnosis on Panoramic X-rays

## Dataset

- **Challenge**: DENTEX (Dental Enumeration and Diagnosis on Panoramic X-rays)
- **Conference**: MICCAI 2023
- **Modality**: Panoramic X-ray
- **Anatomy**: Dental (Teeth and Jaw)
- **Annotation System**: FDI (Fédération Dentaire Internationale)

## Task

Detect and diagnose abnormal teeth on panoramic X-rays with:
1. **Quadrant identification** (Q: 1-4 using FDI system)
2. **Tooth enumeration** (N: 1-8 for each tooth in quadrant)
3. **Diagnosis classification** (D: caries, deep caries, periapical lesions, impacted teeth)

## Dataset Structure

- **Total images**: {len(all_images)} panoramic X-rays from original training set
- **Training split**: {len(train_images)} images (80%) with annotations
- **Test split**: {len(test_images)} images (20%) with annotations held out

Note: This dataset uses an 80/20 split of the original training data.

## Directory Structure

```
train/
├── images/           # {len(train_images)} training X-ray images
└── *.json           # COCO-format annotations

test/
└── images/           # {len(test_images)} test X-ray images
```

## Annotation Format

Annotations are in COCO JSON format with the following structure:
- **Bounding boxes**: For each abnormal tooth
- **Categories**: Quadrant (1-4), Enumeration (1-8), Diagnosis (4 classes)
- **FDI Numbering**: Each tooth labeled as QN (e.g., 48 = Quadrant 4, Tooth 8)

## Diagnosis Classes

1. **Caries**: Tooth decay
2. **Deep Caries**: Advanced tooth decay
3. **Periapical Lesions**: Infection at tooth root
4. **Impacted Teeth**: Teeth that haven't erupted properly

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `image_id`: Image identifier (filename without extension)
   - `predictions_json`: Path to JSON file with predictions (e.g., "predictions/image_001.json")

2. **predictions/** directory with COCO-format JSON files containing:
   - Bounding boxes for detected abnormal teeth
   - Quadrant, enumeration, and diagnosis labels for each detection

## Example Submission Structure:

```
submission/
├── submission.csv
└── predictions/
    ├── image_001.json
    ├── image_002.json
    └── ...
```

## Evaluation

- **Primary Metric**: Mean Average Precision (mAP)
- Detection must correctly identify:
  - Tooth location (bounding box)
  - Quadrant (Q)
  - Enumeration (N)
  - Diagnosis (D)

## Citation

DENTEX Challenge 2023
- Zenodo: https://zenodo.org/records/7812323
- Grand Challenge: https://dentex.grand-challenge.org/dentex/

## Additional Information

- **Hierarchical Data**: Challenge provides 3 levels of annotations
  - Level 1: 693 X-rays (quadrant only)
  - Level 2: 634 X-rays (quadrant + enumeration)
  - Level 3: 1005 X-rays (quadrant + enumeration + diagnosis)
- **Pre-training**: 1571 additional unlabeled X-rays available
- **Patient Age**: 12 and above
- **Sources**: Three different medical institutions
"""

    (public / "README.md").write_text(readme)
    print("Created README.md")

    print("\n✓ Data preparation complete!")

    # Validation checks
    assert public.exists(), "Public directory not created"
    assert private.exists(), "Private directory not created"
    assert (public / "sample_submission.csv").exists(), "Sample submission not created"
    assert (private / "test_labels.csv").exists(), "Test labels not created"

    print("\nSummary:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Training (80%): {len(train_images)} X-ray images with COCO annotations")
    print(f"  Testing (20%): {len(test_images)} X-ray images (labels in private/)")
    print(f"  Split: Random 80/20 split with seed=42")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
