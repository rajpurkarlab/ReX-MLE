"""
Data preparation script for ISLES'22 - Ischemic Stroke Lesion Segmentation Challenge

Dataset structure after extraction (BIDS convention):
- ISLES-2022/sub-strokecase####/ses-0001/anat/sub-strokecase####_ses-0001_FLAIR.nii.gz
- ISLES-2022/sub-strokecase####/ses-0001/dwi/sub-strokecase####_ses-0001_dwi.nii.gz
- ISLES-2022/sub-strokecase####/ses-0001/dwi/sub-strokecase####_ses-0001_adc.nii.gz
- ISLES-2022/derivatives/sub-strokecase####/ses-0001/sub-strokecase####_ses-0001_msk.nii.gz

Note: FLAIR uses uppercase, dwi/adc use lowercase in filenames
      FLAIR is in anat/ subdirectory, DWI and ADC are in dwi/ subdirectory

Alternative flat structure:
- sub-strokecase####_dwi.nii.gz
- sub-strokecase####_adc.nii.gz
- sub-strokecase####_flair.nii.gz
- sub-strokecase####_msk.nii.gz
"""

from pathlib import Path
import pandas as pd
import shutil
from typing import List, Dict
from sklearn.model_selection import train_test_split

from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def find_isles_cases(data_dir: Path) -> List[Dict]:
    """
    Find all ISLES'22 cases with their associated modalities.

    The ISLES'22 dataset follows BIDS convention:
    - FLAIR: sub-strokecase####/ses-0001/anat/sub-strokecase####_ses-0001_FLAIR.nii.gz (uppercase)
    - DWI/ADC: sub-strokecase####/ses-0001/dwi/sub-strokecase####_ses-0001_{dwi,adc}.nii.gz (lowercase)
    - Masks: derivatives/sub-strokecase####/ses-0001/sub-strokecase####_ses-0001_msk.nii.gz

    Alternatively, it may have a flat structure:
    - sub-strokecase####_dwi.nii.gz
    - sub-strokecase####_adc.nii.gz
    - sub-strokecase####_flair.nii.gz
    - sub-strokecase####_msk.nii.gz (ground truth)

    Args:
        data_dir: Directory containing the extracted ISLES data

    Returns:
        List of dicts with case information
    """
    cases = []

    # Find all mask files (ground truth)
    mask_files = list(data_dir.glob("**/sub-strokecase*_msk.nii.gz"))

    for mask_path in sorted(mask_files):
        # Extract case ID from mask filename
        # BIDS format: sub-strokecase####_ses-0001_msk.nii.gz
        # Flat format: sub-strokecase####_msk.nii.gz
        mask_name = mask_path.name

        if "_ses-" in mask_name:
            # BIDS format: extract case_id and session
            # Format: sub-strokecase0001_ses-0001_msk.nii.gz
            parts = mask_name.replace("_msk.nii.gz", "").split("_ses-")
            subject_id = parts[0]  # sub-strokecase0001
            session_id = parts[1] if len(parts) > 1 else "0001"  # 0001
            case_id = mask_name.replace("_msk.nii.gz", "")  # sub-strokecase0001_ses-0001

            # Look for modalities in BIDS structure
            # Check if we're in derivatives/, then look for rawdata/
            if "derivatives" in str(mask_path):
                # Navigate to find rawdata directory
                # derivatives/sub-strokecase####/ses-0001/ -> rawdata/
                rawdata_dir = None
                for parent in mask_path.parents:
                    if parent.name == "derivatives":
                        # Look for rawdata sibling directory
                        rawdata_candidate = parent.parent / "rawdata"
                        if rawdata_candidate.exists():
                            rawdata_dir = rawdata_candidate
                            break
                        # Or maybe the data is directly in parent (no rawdata folder)
                        elif (parent.parent / subject_id).exists():
                            rawdata_dir = parent.parent
                            break

                if rawdata_dir is None:
                    # Try looking in the same parent directory as derivatives
                    for parent in mask_path.parents:
                        if parent.name == "derivatives":
                            rawdata_dir = parent.parent
                            break

                if rawdata_dir:
                    # Look for modalities in BIDS structure
                    # FLAIR is in: rawdata/sub-strokecase####/ses-0001/anat/sub-strokecase####_ses-0001_FLAIR.nii.gz
                    # DWI/ADC are in: rawdata/sub-strokecase####/ses-0001/dwi/sub-strokecase####_ses-0001_{dwi,adc}.nii.gz
                    anat_dir = rawdata_dir / subject_id / f"ses-{session_id}" / "anat"
                    dwi_dir = rawdata_dir / subject_id / f"ses-{session_id}" / "dwi"

                    # FLAIR is in anat directory with uppercase FLAIR
                    flair_path = anat_dir / f"{case_id}_FLAIR.nii.gz"

                    # DWI and ADC are in dwi directory with lowercase names
                    dwi_path = dwi_dir / f"{case_id}_dwi.nii.gz"
                    adc_path = dwi_dir / f"{case_id}_adc.nii.gz"
                else:
                    # Can't find rawdata
                    print(f"⚠ Warning: Case {case_id} - cannot find rawdata directory")
                    continue
            else:
                # Masks not in derivatives, try BIDS structure relative to mask
                # Assume mask is at sub-strokecase####/ses-0001/sub-strokecase####_ses-0001_msk.nii.gz
                # Then modalities are in ../anat/ and ../dwi/
                session_dir = mask_path.parent
                anat_dir = session_dir / "anat"
                dwi_dir = session_dir / "dwi"

                if anat_dir.exists() and dwi_dir.exists():
                    flair_path = anat_dir / f"{case_id}_FLAIR.nii.gz"
                    dwi_path = dwi_dir / f"{case_id}_dwi.nii.gz"
                    adc_path = dwi_dir / f"{case_id}_adc.nii.gz"
                else:
                    # Fall back to flat structure in same directory
                    case_dir = mask_path.parent
                    dwi_path = case_dir / f"{case_id}_DWI.nii.gz"
                    adc_path = case_dir / f"{case_id}_ADC.nii.gz"
                    flair_path = case_dir / f"{case_id}_FLAIR.nii.gz"
        else:
            # Flat format
            case_id = mask_name.replace("_msk.nii.gz", "")
            case_dir = mask_path.parent
            dwi_path = case_dir / f"{case_id}_dwi.nii.gz"
            adc_path = case_dir / f"{case_id}_adc.nii.gz"
            flair_path = case_dir / f"{case_id}_flair.nii.gz"

        # Verify all modalities exist
        if dwi_path.exists() and adc_path.exists() and flair_path.exists():
            cases.append({
                'case_id': case_id,
                'dwi_path': dwi_path,
                'adc_path': adc_path,
                'flair_path': flair_path,
                'mask_path': mask_path
            })
        else:
            missing = []
            if not dwi_path.exists():
                missing.append("DWI")
            if not adc_path.exists():
                missing.append("ADC")
            if not flair_path.exists():
                missing.append("FLAIR")
            print(f"⚠ Warning: Case {case_id} missing modalities: {', '.join(missing)}")

    return cases


def download_data(raw_dir: Path) -> None:
    """Download and extract data from Zenodo."""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    zenodo_record_id = config['zenodo']['record_id']
    print(f"Downloading ISLES'22 dataset from Zenodo: {zenodo_record_id}")

    downloader = ZenodoDownloader()
    files = downloader.download_record(
        record_id=zenodo_record_id,
        output_dir=raw_dir,
        files=None  # Download all files
    )

    # Extract archives
    for file in files:
        if file.suffix == '.zip':
            print(f"Extracting {file.name}...")
            downloader.extract_archive(file, raw_dir)
            print(f"✓ Extracted {file.name}")


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare ISLES'22 data.

    Expected structure in raw/ after download and extraction:
    - ISLES-2022/ directory containing sub-strokecase#### subdirectories
    - Each case has dwi, adc, flair, and msk files
    """
    print(f"Preparing ISLES'22 Challenge...")
    print(f"  Raw: {raw}")
    print(f"  Public: {public}")
    print(f"  Private: {private}")

    # Download data if not present
    isles_zip = raw / "ISLES-2022.zip"
    isles_dir = raw / "ISLES-2022"

    if not isles_dir.exists() and not isles_zip.exists():
        download_data(raw)

    # Extract if needed
    if not isles_dir.exists() and isles_zip.exists():
        print("\nExtracting ISLES-2022.zip...")
        downloader = ZenodoDownloader()
        downloader.extract_archive(isles_zip, raw)
        print("✓ Extracted")

    # Look for the dataset directory
    # It might be directly in raw/ or in a subdirectory
    if not isles_dir.exists():
        # Check for alternative locations
        possible_dirs = [
            raw / "ISLES",
            raw / "isles22",
            raw / "dataset",
        ]
        for possible_dir in possible_dirs:
            if possible_dir.exists():
                isles_dir = possible_dir
                break

    if not isles_dir.exists():
        # Maybe the data is directly in raw/
        # Check if there are sub-strokecase directories
        stroke_cases = list(raw.glob("sub-strokecase*"))
        if stroke_cases:
            isles_dir = raw
        else:
            raise FileNotFoundError(
                f"ISLES-2022 directory not found in {raw}\n"
                f"Found: {list(raw.glob('*'))}"
            )

    print(f"\nUsing ISLES data from: {isles_dir}")

    # Find all cases
    print("\nScanning for stroke cases...")
    all_cases = find_isles_cases(isles_dir)
    print(f"Found {len(all_cases)} cases with all modalities and ground truth")

    if len(all_cases) == 0:
        raise ValueError("No complete cases found!")

    # Sort by case_id for reproducibility
    all_cases = sorted(all_cases, key=lambda x: x['case_id'])

    # Split train/test (80/20)
    train_cases, test_cases = train_test_split(
        all_cases,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Split: {len(train_cases)} train, {len(test_cases)} test")

    # Create output directories
    (public / "train").mkdir(parents=True, exist_ok=True)
    (public / "test").mkdir(parents=True, exist_ok=True)
    (private / "test").mkdir(parents=True, exist_ok=True)

    # Copy training data (with masks)
    print("\nCopying training data...")
    for case in train_cases:
        case_id = case['case_id']

        # Simplify case_id for output (remove session if present)
        # sub-strokecase0001_ses-0001 -> sub-strokecase0001
        output_id = case_id.split('_ses-')[0] if '_ses-' in case_id else case_id

        # Copy all modalities
        shutil.copy(case['dwi_path'], public / "train" / f"{output_id}_dwi.nii.gz")
        shutil.copy(case['adc_path'], public / "train" / f"{output_id}_adc.nii.gz")
        shutil.copy(case['flair_path'], public / "train" / f"{output_id}_flair.nii.gz")
        shutil.copy(case['mask_path'], public / "train" / f"{output_id}_msk.nii.gz")

    # Copy test data
    print("Copying test data...")
    test_metadata = []

    for case in test_cases:
        case_id = case['case_id']

        # Simplify case_id for output (remove session if present)
        output_id = case_id.split('_ses-')[0] if '_ses-' in case_id else case_id

        # Copy modalities to public (without masks)
        shutil.copy(case['dwi_path'], public / "test" / f"{output_id}_dwi.nii.gz")
        shutil.copy(case['adc_path'], public / "test" / f"{output_id}_adc.nii.gz")
        shutil.copy(case['flair_path'], public / "test" / f"{output_id}_flair.nii.gz")

        # Copy mask to private
        shutil.copy(case['mask_path'], private / "test" / f"{output_id}_msk.nii.gz")

        test_metadata.append({
            'case_id': output_id,
            'dwi_path': f"test/{output_id}_dwi.nii.gz",
            'adc_path': f"test/{output_id}_adc.nii.gz",
            'flair_path': f"test/{output_id}_flair.nii.gz",
            'mask_path': f"test/{output_id}_msk.nii.gz"
        })

    # Save test labels CSV
    test_df = pd.DataFrame(test_metadata)
    test_df.to_csv(private / "test_labels.csv", index=False)
    print(f"Saved test labels: {len(test_df)} cases")

    # Create sample submission
    sample_sub = test_df[['case_id']].copy()
    sample_sub['predicted_mask_path'] = sample_sub['case_id'].apply(
        lambda x: f"predictions/{x}_pred.nii.gz"
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
    readme = f"""# ISLES'22 - Ischemic Stroke Lesion Segmentation Challenge

## Dataset

- **Task**: Multimodal MRI Infarct Segmentation in Acute and Sub-Acute Stroke
- **Modalities**: DWI (Diffusion Weighted Imaging), ADC (Apparent Diffusion Coefficient), FLAIR
- **Format**: NIfTI (.nii.gz) following BIDS convention
- **Total cases**: {len(all_cases)}
- **Training cases**: {len(train_cases)} (with images and ground truth masks)
- **Test cases**: {len(test_cases)} (images only, masks held out)

## Directory Structure

```
train/
├── sub-strokecase####_dwi.nii.gz     # {len(train_cases)} DWI images
├── sub-strokecase####_adc.nii.gz     # {len(train_cases)} ADC maps
├── sub-strokecase####_flair.nii.gz   # {len(train_cases)} FLAIR images
└── sub-strokecase####_msk.nii.gz     # {len(train_cases)} Ground truth masks

test/
├── sub-strokecase####_dwi.nii.gz     # {len(test_cases)} DWI images
├── sub-strokecase####_adc.nii.gz     # {len(test_cases)} ADC maps
└── sub-strokecase####_flair.nii.gz   # {len(test_cases)} FLAIR images
```

## Task: Stroke Lesion Segmentation

Automatically segment ischemic stroke lesions using multimodal MRI (DWI, ADC, FLAIR).

## Clinical Context

- **Acute and sub-acute stroke lesions**
- **Multiple embolic and/or cortical infarcts**
- **Pre- and post-interventional MRI patterns**
- **Variable lesion size and burden**

## Image Modalities

1. **DWI (Diffusion Weighted Imaging, b=1000)**
   - Shows acute ischemic changes
   - Bright signal in infarcted tissue

2. **ADC (Apparent Diffusion Coefficient)**
   - Quantifies water diffusion
   - Reduced values in acute infarcts

3. **FLAIR (Fluid Attenuated Inversion Recovery)**
   - Suppresses CSF signal
   - Shows sub-acute and chronic lesions

## Submission Format

Your submission should include:

1. **submission.csv** with columns:
   - `case_id`: Case identifier (e.g., "sub-strokecase0001")
   - `predicted_mask_path`: Relative path to predicted segmentation mask

2. **predictions/** directory with NIfTI masks:
   - Binary masks (0 = background, 1 = stroke lesion)
   - Same dimensions as input images
   - Format: NIfTI (.nii.gz)

## Example Structure:
```
submission/
├── submission.csv
└── predictions/
    ├── sub-strokecase0001_pred.nii.gz
    ├── sub-strokecase0002_pred.nii.gz
    └── ...
```

## Evaluation

- **Primary metric**: Dice Similarity Coefficient (DSC)
- **Secondary metrics**: Hausdorff Distance, Precision, Recall, Lesion-wise F1

## Scanner Information

Images acquired using:
- 3T Philips (Achieva, Ingenia)
- 3T Siemens Verio
- 1.5T Siemens MAGNETOM (Avanto, Aera)

## Citation

```
@dataset{{isles22_2023,
  author       = {{ISLES Challenge Organizers}},
  title        = {{ISLES 2022 Challenge Dataset}},
  year         = {{2023}},
  publisher    = {{Zenodo}},
  doi          = {{10.5281/zenodo.7960856}},
  url          = {{https://doi.org/10.5281/zenodo.7960856}}
}}
```

## License

CC BY 4.0

## Links

- **Challenge Website**: https://isles22.grand-challenge.org/
- **Dataset**: https://zenodo.org/record/7960856
- **Challenge Document**: https://zenodo.org/record/6362388
"""

    (public / "README.md").write_text(readme)

    print("\n✓ Data preparation complete!")
    print(f"\nSummary:")
    print(f"  Training: {len(train_cases)} cases with all modalities and masks")
    print(f"  Testing: {len(test_cases)} cases (masks in private/)")
    print(f"  Files: sample_submission.csv, test_labels.csv, README.md")
