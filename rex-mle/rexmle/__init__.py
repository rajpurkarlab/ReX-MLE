"""
Med-MLE-Bench: Medical Imaging Challenge Benchmark

A benchmark for evaluating AI agents on medical imaging challenges from Grand Challenge.
"""

from rexmle.registry import registry, Challenge
from rexmle.grade_helpers import Grader, InvalidSubmissionError
from rexmle.zenodo_downloader import ZenodoDownloader

__version__ = "0.1.0"

__all__ = [
    "registry",
    "Challenge",
    "Grader",
    "InvalidSubmissionError",
    "ZenodoDownloader",
]
