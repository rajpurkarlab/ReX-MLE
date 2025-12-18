from pathlib import Path
import pandas as pd
import shutil
import json
import gdown
from rexmle.zenodo_downloader import ZenodoDownloader
from rexmle.utils import load_yaml


def download_data(raw_dir: Path) -> None:
    """Download training data from Zenodo and test data from Google Drive"""
    config_path = Path(__file__).parent / "config.yaml"
    config = load_yaml(config_path)

    # Download training data from Zenodo
    zenodo_config = config['zenodo']
    downloader = ZenodoDownloader()
    files = downloader.download_record(
        record_id=zenodo_config['record_id'],
        output_dir=raw_dir,
        files=zenodo_config.get('files')
    )

    # Extract Zenodo archives
    for file in files:
        if file.suffix in ['.zip', '.tar', '.gz', '.tgz']:
            downloader.extract_archive(file, raw_dir)

    # Download test data from Google Drive
    google_config = config.get('google_test_set')
    if google_config:
        file_id = google_config['id']
        test_zip_path = raw_dir / "test_dataset.zip"
        gdown.download(id=file_id, output=str(test_zip_path), quiet=False)

        # Extract test archive
        if test_zip_path.exists():
            downloader.extract_archive(test_zip_path, raw_dir)


def prepare(raw: Path, public: Path, private: Path) -> None:
    """
    Prepare LDCT-IQA dataset for image quality assessment challenge.

    Dataset structure:
    - Training: Zenodo dataset with images and train.json
    - Test: Google Drive dataset with images and test.json

    Goals:
    1. Copy train images + labels to public/
    2. Copy test images (no labels) to public/
    3. Copy test labels to private/
    4. Create test_labels.csv in private/
    5. Create sample_submission.csv in public/
    """

    # Download if needed
    if not raw.exists() or not list(raw.glob('*')):
        download_data(raw)

    # Process training data
    train_json_path = raw / "LDCTIQAG2023_train" / "train.json"
    with open(train_json_path, 'r') as f:
        train_quality_scores = json.load(f)

    train_images_dir = raw / "LDCTIQAG2023_train" / "image"
    train_images = sorted(list(train_images_dir.glob("*.tif")))

    # Create directories
    (public / "train" / "images").mkdir(parents=True, exist_ok=True)
    (public / "test" / "images").mkdir(parents=True, exist_ok=True)

    # Copy training images and create train labels
    train_labels = []
    for img_path in train_images:
        img_name = img_path.name
        if img_name in train_quality_scores:
            # Copy image to public/train
            shutil.copy(img_path, public / "train" / "images" / img_name)

            train_labels.append({
                'image_id': img_name.replace('.tif', ''),
                'quality_score': train_quality_scores[img_name]
            })

    # Save training labels
    train_df = pd.DataFrame(train_labels)
    train_df.to_csv(public / "train_labels.csv", index=False)

    # Process test data
    test_json_path = raw / "LDCTIQAC_test" / "test.json"
    with open(test_json_path, 'r') as f:
        test_quality_scores = json.load(f)

    test_images_dir = raw / "LDCTIQAC_test" / "images"
    test_images = sorted(list(test_images_dir.glob("*.tiff")))

    # Copy test images and create test labels (for private)
    test_labels = []
    for img_path in test_images:
        img_name = img_path.name
        # Copy image to public/test (no labels visible)
        shutil.copy(img_path, public / "test" / "images" / img_name)

        # Save ground truth labels to private
        if img_name in test_quality_scores:
            test_labels.append({
                'image_id': img_name.replace('.tiff', ''),
                'quality_score': test_quality_scores[img_name]
            })

    # Save test labels to private directory
    test_df = pd.DataFrame(test_labels)
    test_df.to_csv(private / "test_labels.csv", index=False)

    # Create sample submission
    sample = test_df[['image_id']].copy()
    sample['quality_score'] = 0.0  # Placeholder prediction
    sample.to_csv(public / "sample_submission.csv", index=False)

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
