"""
Zenodo API client for downloading medical imaging datasets.
"""

import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import hashlib
import logging

from rexmle.utils import get_repo_dir

logger = logging.getLogger(__name__)


class ZenodoDownloader:
    """
    Client for downloading datasets from Zenodo.

    Zenodo is a general-purpose open-access repository where Grand Challenge
    datasets are often hosted.
    """

    BASE_URL = "https://zenodo.org/api/records"

    def __init__(self, api_token: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize Zenodo downloader.

        Args:
            api_token: Optional Zenodo API token for private records
            cache_dir: Directory to cache downloads (default: rex-mle/data/)
        """
        self.api_token = api_token
        if cache_dir is None:
            # Use rex-mle/data/ directory
            self.cache_dir = get_repo_dir() / "data"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_record_info(self, record_id: str) -> Dict[str, Any]:
        """
        Get metadata for a Zenodo record.

        Args:
            record_id: Zenodo record ID (e.g., "1234567")

        Returns:
            Record metadata dictionary

        Raises:
            requests.HTTPError: If record not found or access denied
        """
        url = f"{self.BASE_URL}/{record_id}"

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        logger.info(f"Fetching Zenodo record info: {record_id}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def download_file(
        self,
        url: str,
        destination: Path,
        checksum: Optional[str] = None,
        chunk_size: int = 8192
    ) -> None:
        """
        Download a file from Zenodo with progress bar and checksum verification.

        Args:
            url: URL to download from
            destination: Local path to save file
            checksum: Expected MD5 checksum (format: "md5:hash")
            chunk_size: Download chunk size in bytes

        Raises:
            ValueError: If checksum verification fails
            requests.HTTPError: If download fails
        """
        # Skip if already downloaded and valid
        if destination.exists():
            if checksum and self._verify_checksum(destination, checksum):
                logger.info(
                    f"File already downloaded and verified: {destination.name}")
                return
            else:
                logger.warning(
                    f"Re-downloading (checksum mismatch or no checksum): {destination.name}")

        # Download with progress bar
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        logger.info(f"Downloading: {destination.name}")
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify checksum
        if checksum:
            if self._verify_checksum(destination, checksum):
                logger.info(f"✓ Checksum verified: {destination.name}")
            else:
                raise ValueError(f"Checksum mismatch for {destination.name}")

    def download_record(
        self,
        record_id: str,
        output_dir: Path,
        files: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Download all files (or specific files) from a Zenodo record.

        Args:
            record_id: Zenodo record ID
            output_dir: Directory to save files
            files: Optional list of specific filenames to download

        Returns:
            List of downloaded file paths

        Raises:
            ValueError: If no files found or specified files not found
            requests.HTTPError: If record not found
        """
        logger.info(f"Fetching Zenodo record: {record_id}")

        record = self.get_record_info(record_id)

        # Get file list
        available_files = record.get('files', [])

        if not available_files:
            raise ValueError(f"No files found in Zenodo record {record_id}")

        # Filter files if specified
        if files:
            available_files = [
                f for f in available_files
                if f['key'] in files
            ]

            if not available_files:
                raise ValueError(
                    f"Specified files not found in record: {files}")

        logger.info(
            f"Downloading {len(available_files)} file(s) from Zenodo record {record_id}...")

        downloaded = []

        for file_info in available_files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            checksum = file_info.get('checksum', '')

            destination = output_dir / filename

            self.download_file(download_url, destination, checksum)
            downloaded.append(destination)

        return downloaded

    def extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """
        Extract a compressed archive (zip, tar, tar.gz).

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to

        Raises:
            ValueError: If archive format not supported
        """
        logger.info(f"Extracting {archive_path.name}...")

        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.tar':
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.suffix == '.tgz' or '.tar.gz' in archive_path.name:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(
                f"Unsupported archive format: {archive_path.suffix}")

        logger.info(f"✓ Extracted to: {extract_to}")

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """
        Verify checksum of a file.

        Args:
            file_path: Path to file
            expected_checksum: Expected checksum (format: "md5:hash" or just "hash")

        Returns:
            True if checksum matches
        """
        # Extract hash from format "md5:hash"
        if ':' in expected_checksum:
            _, expected_hash = expected_checksum.split(':', 1)
        else:
            expected_hash = expected_checksum

        md5 = hashlib.md5()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)

        return md5.hexdigest() == expected_hash
