"""
Data preparation script for PUMA Track 1 Task 1 - Semantic Tissue Segmentation

Dataset structure after extraction:
- 01_training_dataset_tif_ROIs/: Training images (TIF format, 1024x1024 pixels)
- 01_training_dataset_geojson_tissue/: Tissue segmentation annotations (GeoJSON format)
"""

from pathlib import Path
import pandas as pd
import shutil
import json
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def get_shared_raw_dir(raw: Path) -> Path:
    """Get shared raw directory for all PUMA challenges."""
    return raw.parent.parent / "puma-shared" / "raw"


def convert_geojson_to_mask(geojson_path: Path, output_path: Path, image_shape: tuple = (1024, 1024)) -> None:
    """
    Convert GeoJSON tissue annotation to segmentation mask.

    Args:
        geojson_path: Path to GeoJSON file with tissue annotations
        output_path: Path to save output mask (PNG format)
        image_shape: Output image shape (height, width)

    Class mapping:
    - Background: 0
    - Stroma: 1
    - Blood Vessel: 2
    - Tumor: 3
    - Epidermis: 4
    - Necrosis: 5
    """
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Create blank mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Class name to ID mapping
    class_mapping = {
        'stroma': 1,
        'blood vessel': 2,
        'tumor': 3,
        'epidermis': 4,
        'necrosis': 5,
        # Alternative names
        'blood_vessel': 2,
        'stromal': 1,
    }

    # Process each feature
    features = geojson_data.get('features', [])
    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        # Get class name
        class_name = properties.get('classification', {}).get('name', '').lower()
        if not class_name:
            class_name = properties.get('name', '').lower()

        # Get class ID
        class_id = class_mapping.get(class_name, 0)
        if class_id == 0:
            continue  # Skip unknown classes

        # Get coordinates
        if geometry.get('type') == 'Polygon':
            coordinates_list = [geometry.get('coordinates', [])]
        elif geometry.get('type') == 'MultiPolygon':
            coordinates_list = geometry.get('coordinates', [])
        else:
            continue

        # Draw each polygon
        for polygon_coords in coordinates_list:
            if not polygon_coords:
                continue

            # Get exterior ring (first element)
            exterior = polygon_coords[0] if polygon_coords else []
            if len(exterior) < 3:
                continue

            # Convert to PIL format [(x, y), ...]
            points = [(float(x), float(y)) for x, y in exterior]

            # Draw polygon on mask
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)
            draw.polygon(points, fill=class_id, outline=class_id)
            mask = np.array(img)

    # Save mask as PNG
    Image.fromarray(mask).save(output_path)


def download_data(raw_dir: Path) -> None:
    """Download and extract ALL PUMA data from Zenodo to shared location."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)
    zenodo_record_id = config['zenodo']['record_id']

    print(f"Downloading complete PUMA dataset from Zenodo: {zenodo_record_id}")
    print("Note: Downloading all PUMA files to shared location for use by all tasks")

    # Download ALL PUMA files (both tissue and nuclei annotations)
    all_puma_files = [
        "01_training_dataset_tif_ROIs.zip",
        "01_training_dataset_geojson_tissue.zip",
        "01_training_dataset_geojson_nuclei.zip"
    ]

    downloader = ZenodoDownloader()
    files = downloader.download_record(
        record_id=zenodo_record_id,
        output_dir=raw_dir,
        files=all_puma_files
    )

    # Extract archives
    for file in files:
        if file.suffix == '.zip' or '.tar.gz' in file.name:
            print(f"Extracting: {file.name}")
            downloader.extract_archive(file, raw_dir)
            print(f"✓ Extracted: {file.stem}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare PUMA Track 1 Task 1 data (Tissue Segmentation).

    Expected structure in raw/:
    - 01_training_dataset_tif_ROIs/: TIF images (1024x1024, H&E stained)
    - 01_training_dataset_geojson_tissue/: GeoJSON tissue annotations
    """
    print(f"Preparing PUMA Track 1 Task 1 (Tissue Segmentation)...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Get shared raw directory
    shared_raw = get_shared_raw_dir(raw)

    # Check if data exists in shared location
    shared_images = shared_raw / "01_training_dataset_tif_ROIs"
    shared_tissue = shared_raw / "01_training_dataset_geojson_tissue"

    # Download if images or tissue annotations don't exist
    if not (shared_images.exists() and shared_tissue.exists() and list(shared_images.glob("*.tif"))):
        print(f"\nDownloading to shared location: {shared_raw}")
        download_data(shared_raw)
    else:
        print(f"\nUsing shared PUMA data from: {shared_raw}")

    raw = shared_raw

    # Define data directories
    images_dir = raw / "01_training_dataset_tif_ROIs"
    tissue_labels_dir = raw / "01_training_dataset_geojson_tissue"

    # Handle case where tissue files are extracted directly to raw directory
    if not tissue_labels_dir.exists():
        tissue_files = list(raw.glob("*_tissue.geojson"))
        if tissue_files:
            print(f"Found {len(tissue_files)} tissue files in raw directory, organizing into subdirectory...")
            tissue_labels_dir.mkdir(exist_ok=True)
            for file in tissue_files:
                shutil.move(str(file), str(tissue_labels_dir / file.name))
            print(f"Moved tissue files to {tissue_labels_dir}")

    if not images_dir.exists() or not tissue_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected 01_training_dataset_tif_ROIs/ and 01_training_dataset_geojson_tissue/ in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find all TIF images
    all_images = sorted(images_dir.glob("*.tif"))
    print(f"\nFound {len(all_images)} TIF images")

    # Match images with tissue annotations
    matched_cases = []
    for img_path in all_images:
        # Extract case ID from filename
        # Expected format: training_set_primary_roi_001.tif or training_set_metastatic_roi_001.tif
        case_id = img_path.stem

        # Corresponding GeoJSON file (with _tissue suffix)
        label_path = tissue_labels_dir / f"{case_id}_tissue.geojson"

        if label_path.exists():
            matched_cases.append({
                'case_id': case_id,
                'image_path': img_path,
                'label_path': label_path
            })
        else:
            print(f"⚠ Warning: No tissue annotation found for {img_path.name}")

    print(f"Matched {len(matched_cases)} cases with tissue annotations")

    if len(matched_cases) == 0:
        raise ValueError("No matched cases found!")

    # Sort by case_id for reproducibility
    matched_cases = sorted(matched_cases, key=lambda x: x['case_id'])

    # Split train/test (80/20) with fixed random_state
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

    # Copy training data
    print("\nCopying training data...")
    for case in train_cases:
        shutil.copy(case['image_path'], public / "train" / "images" / case['image_path'].name)
        shutil.copy(case['label_path'], public / "train" / "labels" / case['label_path'].name)

    # Copy test data
    print("Copying test data and converting labels to TIF masks...")
    test_metadata = []
    for case in test_cases:
        # Copy image to public
        shutil.copy(case['image_path'], public / "test" / "images" / case['image_path'].name)

        # Convert GeoJSON label to TIF mask and save to private
        # This matches the real Grand Challenge format where ground truth is TIF
        geojson_path = case['label_path']
        mask_filename = case['case_id'] + '.tif'
        mask_path = private / "test" / "labels" / mask_filename

        print(f"  Converting {geojson_path.name} -> {mask_filename}")
        convert_geojson_to_mask(geojson_path, mask_path, image_shape=(1024, 1024))

        test_metadata.append({
            'case_id': case['case_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{mask_filename}"  # Now points to TIF, not GeoJSON
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    sample_sub = test_df[['case_id']].copy()
    sample_sub['predicted_mask_path'] = sample_sub['case_id'].apply(
        lambda x: f"predictions/{x}.tif"  # Use TIF to match Grand Challenge format
    )
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
    readme = f"""# PUMA Track 1 Task 1: Semantic Tissue Segmentation

## Dataset

- **Modality**: H&E Histopathology
- **Anatomy**: Skin - Melanoma
- **Format**: TIF (1024x1024 pixels at 40× magnification, 0.23 μm per pixel)
- **Total cases**: {len(matched_cases)} (103 primary + 102 metastatic melanoma)
- **Training cases**: {len(train_cases)} (with images and tissue annotations)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} H&E histopathology images
└── labels/        # {len(train_cases)} tissue segmentation annotations (GeoJSON)

test/
└── images/        # {len(test_cases)} test images
```

## Task: Semantic Tissue Segmentation

Segment tissue regions into the following classes:
1. **Tumor**: Melanoma tumor cells
2. **Stroma**: Connective tissue
3. **Epithelium**: Epithelial tissue
4. **Blood Vessel**: Vascular structures
5. **Necrosis**: Necrotic regions

## File Naming

- Images: `training_set_{{type}}_roi_{{id}}.tif`
  - Example: `training_set_primary_roi_001.tif`, `training_set_metastatic_roi_001.tif`
- Labels: `training_set_{{type}}_roi_{{id}}.geojson`

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `case_id`: Case identifier (e.g., "training_set_primary_roi_001")
   - `predicted_mask_path`: Relative path to predicted segmentation mask

2. **predictions/** directory with segmentation masks:
   - Format: PNG or TIF
   - Same dimensions as input images (1024x1024)
   - Pixel values represent tissue class labels

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── training_set_primary_roi_001.png
    ├── training_set_primary_roi_002.png
    └── ...
```

## Evaluation

- **Primary metric**: Dice Similarity Coefficient
- **Secondary metrics**: IoU, Precision, Recall

## Citation

If you use this dataset, please cite:

Schuiveling, Mark; Blokx, Willeke; Breimer, Gerben (2024):
Melanoma Histopathology Dataset with Tissue and Nuclei Annotations.
Zenodo. https://doi.org/10.5281/zenodo.14869398

## License

Please refer to the Zenodo dataset page for license information.
"""

    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Training: {len(train_cases)} cases with images and tissue annotations")
    print(f"  Testing: {len(test_cases)} cases (labels in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
