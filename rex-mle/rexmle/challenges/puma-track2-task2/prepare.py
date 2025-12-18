"""
Data preparation script for PUMA Track 2 Task 2 - Nuclei Detection (10 Classes)

Dataset structure after extraction:
- 01_training_dataset_tif_ROIs/: Training images (TIF format, 1024x1024 pixels)
- 01_training_dataset_geojson_nuclei/: Nuclei segmentation annotations (GeoJSON format)
"""

from pathlib import Path
import pandas as pd
import shutil
import json
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def get_shared_raw_dir(raw: Path) -> Path:
    """Get shared raw directory for all PUMA challenges."""
    return raw.parent.parent / "puma-shared" / "raw"


def convert_nuclei_geojson_to_json(geojson_path: Path, output_path: Path, class_mapping: dict) -> None:
    """
    Convert nuclei GeoJSON annotation to simplified "Multiple Polygons" JSON format.

    This matches the format expected by the real Grand Challenge evaluation.

    Args:
        geojson_path: Path to GeoJSON file with nuclei annotations
        output_path: Path to save output JSON
        class_mapping: Dict mapping full class names to simplified names
                      e.g., {'nuclei_tumor': 'tumor', 'nuclei_lymphocytes': 'lymphocytes'}

    Output format:
    {
        "polygons": [
            {
                "name": "tumor",
                "path_points": [[x1, y1], [x2, y2], ...],
                "score": 1.0
            },
            ...
        ]
    }
    """
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    polygons = []

    # Process each feature
    features = geojson_data.get('features', [])
    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        # Get class name
        classification = properties.get('classification', {})
        class_name = classification.get('name', '').lower()

        # Map to simplified class name
        mapped_class = class_mapping.get(class_name, class_name)
        if not mapped_class:
            continue

        # Get coordinates
        if geometry.get('type') == 'Polygon':
            coordinates = geometry.get('coordinates', [[]])[0]  # Get exterior ring
        else:
            continue

        if len(coordinates) < 3:
            continue

        # Convert to path_points format
        path_points = [[float(x), float(y)] for x, y in coordinates]

        polygons.append({
            'name': mapped_class,
            'path_points': path_points,
            'score': 1.0  # Default confidence score
        })

    # Save as JSON
    output_data = {'polygons': polygons}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def download_data(raw_dir: Path) -> None:
    """Download and extract ALL PUMA data from Zenodo to shared location."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)
    zenodo_record_id = config["zenodo"]["record_id"]

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
        if file.suffix == ".zip" or ".tar.gz" in file.name:
            print(f"Extracting: {file.name}")
            downloader.extract_archive(file, raw_dir)
            print(f"✓ Extracted: {file.stem}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare PUMA Track 2 Task 2 data (Nuclei Detection - 10 Classes).

    Expected structure in raw/:
    - 01_training_dataset_tif_ROIs/: TIF images (1024x1024, H&E stained)
    - 01_training_dataset_geojson_nuclei/: GeoJSON nuclei annotations
    """
    print(f"Preparing PUMA Track 2 Task 2 (Nuclei Detection - 10 Classes)...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Get shared raw directory
    shared_raw = get_shared_raw_dir(raw)

    # Check if data exists in shared location
    shared_images = shared_raw / "01_training_dataset_tif_ROIs"
    shared_nuclei = shared_raw / "01_training_dataset_geojson_nuclei"

    if shared_images.exists() and shared_nuclei.exists() and list(shared_images.glob("*.tif")):
        print(f"\nUsing shared PUMA data from: {shared_raw}")
        raw = shared_raw
    else:
        # Download to shared location
        print(f"\nDownloading to shared location: {shared_raw}")
        if not shared_nuclei.exists():
            download_data(shared_raw)
        raw = shared_raw

    # Define data directories
    images_dir = raw / "01_training_dataset_tif_ROIs"
    nuclei_labels_dir = raw / "01_training_dataset_geojson_nuclei"

    # Handle case where nuclei files are extracted directly to raw directory
    if not nuclei_labels_dir.exists():
        nuclei_files = list(raw.glob("*_nuclei.geojson"))
        if nuclei_files:
            print(f"Found {len(nuclei_files)} nuclei files in raw directory, organizing into subdirectory...")
            nuclei_labels_dir.mkdir(exist_ok=True)
            for file in nuclei_files:
                shutil.move(str(file), str(nuclei_labels_dir / file.name))
            print(f"Moved nuclei files to {nuclei_labels_dir}")

    if not images_dir.exists() or not nuclei_labels_dir.exists():
        raise FileNotFoundError(
            f"Expected 01_training_dataset_tif_ROIs/ and 01_training_dataset_geojson_nuclei/ in {raw}\n"
            f"Found: {list(raw.glob('*'))}"
        )

    # Find all TIF images
    all_images = sorted(images_dir.glob("*.tif"))
    print(f"\nFound {len(all_images)} TIF images")

    # Match images with nuclei annotations
    matched_cases = []
    for img_path in all_images:
        # Extract case ID from filename
        case_id = img_path.stem

        # Corresponding GeoJSON file (with _nuclei suffix based on tissue naming pattern)
        label_path = nuclei_labels_dir / f"{case_id}_nuclei.geojson"

        if label_path.exists():
            matched_cases.append({
                'case_id': case_id,
                'image_path': img_path,
                'label_path': label_path
            })
        else:
            print(f"⚠ Warning: No nuclei annotation found for {img_path.name}")

    print(f"Matched {len(matched_cases)} cases with nuclei annotations")

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
    print("Copying test data and converting labels to JSON...")

    # Track 2 has 10 classes (granular classification, no grouping)
    class_mapping_track2 = {
        'nuclei_tumor': 'tumor',
        'nuclei_lymphocytes': 'lymphocytes',
        'nuclei_plasma_cells': 'plasma_cells',
        'nuclei_histiocytes': 'histiocytes',
        'nuclei_melanophages': 'melanophages',
        'nuclei_neutrophils': 'neutrophils',
        'nuclei_stromal_cells': 'stromal_cells',
        'nuclei_epithelium': 'epithelium',
        'nuclei_endothelium': 'endothelium',
        'nuclei_apoptotic_cells': 'apoptotic_cells',
    }

    test_metadata = []
    for case in test_cases:
        # Copy image to public
        shutil.copy(case['image_path'], public / "test" / "images" / case['image_path'].name)

        # Convert GeoJSON label to simplified JSON and save to private
        # This matches the real Grand Challenge format where ground truth is JSON
        geojson_path = case['label_path']
        json_filename = case['case_id'] + '_nuclei.json'
        json_path = private / "test" / "labels" / json_filename

        print(f"  Converting {geojson_path.name} -> {json_filename}")
        convert_nuclei_geojson_to_json(geojson_path, json_path, class_mapping_track2)

        test_metadata.append({
            'case_id': case['case_id'],
            'image_path': f"test/images/{case['image_path'].name}",
            'label_path': f"test/labels/{json_filename}"  # Now points to JSON, not GeoJSON
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    sample_sub = test_df[['case_id']].copy()
    sample_sub['predicted_nuclei_path'] = sample_sub['case_id'].apply(
        lambda x: f"predictions/{x}.json"
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
    readme = f"""# PUMA Track 2 Task 2: Nuclei Detection (10 Classes)

## Dataset

- **Modality**: H&E Histopathology
- **Anatomy**: Skin - Melanoma
- **Format**: TIF (1024x1024 pixels at 40× magnification, 0.23 μm per pixel)
- **Total cases**: {len(matched_cases)} (103 primary + 102 metastatic melanoma)
- **Training cases**: {len(train_cases)} (with images and nuclei annotations)
- **Test cases**: {len(test_cases)} (images only, labels held out)

## Directory Structure

```
train/
├── images/        # {len(train_cases)} H&E histopathology images
└── labels/        # {len(train_cases)} nuclei detection annotations (GeoJSON)

test/
└── images/        # {len(test_cases)} test images
```

## Task: Nuclei Detection (10 Classes)

Detect and classify nuclei into ten granular categories:

1. **tumor**: Melanoma tumor cell nuclei
2. **lymphocytes**: Tumor-infiltrating lymphocytes
3. **plasma_cells**: Plasma cells
4. **histiocytes**: Histiocyte cells
5. **melanophages**: Melanophage cells
6. **neutrophils**: Neutrophil cells
7. **stromal_cells**: Stromal cells
8. **epithelium**: Epithelial cells
9. **endothelium**: Endothelial cells
10. **apoptotic_cells**: Apoptotic cells

## File Naming

- Images: `training_set_{{type}}_roi_{{id}}.tif`
  - Example: `training_set_primary_roi_001.tif`, `training_set_metastatic_roi_001.tif`
- Labels: `training_set_{{type}}_roi_{{id}}.geojson`

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `case_id`: Case identifier (e.g., "training_set_primary_roi_001")
   - `predicted_nuclei_path`: Relative path to predicted nuclei detection file

2. **predictions/** directory with nuclei detection results:
   - Format: JSON or GeoJSON
   - Each file contains list of detected nuclei with:
     - Centroid coordinates (x, y)
     - Class label (tumor, lymphocytes, plasma_cells, histiocytes, melanophages, neutrophils, stromal_cells, epithelium, endothelium, apoptotic_cells)
     - Optional: polygon coordinates, confidence score

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── training_set_primary_roi_001.json
    ├── training_set_primary_roi_002.json
    └── ...
```

## Evaluation

- **Primary metric**: F1 Score (detection accuracy with classification)
- **Secondary metrics**: Precision, Recall, Average Precision

## Quality Metrics

The dataset annotations have been validated with:
- **Intraobserver F1 score**: 85.66%
- **Intraobserver Precision**: 84.89%
- **Intraobserver Recall**: 86.45%
- **Interobserver F1 score**: 80.20%

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
    print(f"  Training: {len(train_cases)} cases with images and nuclei annotations")
    print(f"  Testing: {len(test_cases)} cases (labels in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
