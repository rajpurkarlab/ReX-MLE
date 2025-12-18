"""
Registry for medical imaging challenges.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from appdirs import user_cache_dir

from rexmle.grade_helpers import Grader
from rexmle.utils import get_logger, get_module_dir, get_repo_dir, import_fn, load_yaml

logger = get_logger(__name__)


# Default data directory: data/ subdirectory in rex-mle repo root
def _get_default_data_dir() -> Path:
    """Get default data directory at rex-mle/data/"""
    data_dir = get_repo_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.resolve()


DEFAULT_DATA_DIR = _get_default_data_dir()


@dataclass(frozen=True)
class Challenge:
    """
    Medical imaging challenge definition.

    Similar to Competition in mle-bench, but adapted for medical challenges.
    """
    id: str
    name: str
    description: str
    grader: Grader
    answers: Path
    answers_dir: Path
    gold_submission: Path
    sample_submission: Path
    competition_type: str
    prepare_fn: Callable[[Path, Path, Path], Path]
    raw_dir: Path
    private_dir: Path
    public_dir: Path
    checksums: Path
    leaderboard: Path

    def __post_init__(self):
        assert isinstance(self.id, str), "Challenge id must be a string."
        assert isinstance(self.name, str), "Challenge name must be a string."
        assert isinstance(self.description,
                          str), "Challenge description must be a string."
        assert isinstance(
            self.grader, Grader), "Challenge grader must be of type Grader."
        assert isinstance(
            self.answers, Path), "Challenge answers must be a Path."
        assert isinstance(
            self.answers_dir, Path), "Challenge answers dir must be a Path."
        assert isinstance(self.gold_submission,
                          Path), "Gold submission must be a Path."
        assert isinstance(self.sample_submission,
                          Path), "Sample submission must be a Path."
        assert isinstance(self.competition_type,
                          str), "Challenge type must be a string."
        assert isinstance(self.checksums, Path), "Checksums must be a Path."
        assert isinstance(self.leaderboard,
                          Path), "Leaderboard must be a Path."
        assert len(self.id) > 0, "Challenge id cannot be empty."
        assert len(self.name) > 0, "Challenge name cannot be empty."
        assert len(self.description) > 0, "Challenge description cannot be empty."
        assert len(self.competition_type) > 0, "Challenge type cannot be empty."

    @staticmethod
    def from_dict(data: dict) -> "Challenge":
        grader = Grader.from_dict(data["grader"])

        try:
            return Challenge(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                grader=grader,
                answers=data["answers"],
                answers_dir=data["answers_dir"],
                sample_submission=data["sample_submission"],
                gold_submission=data["gold_submission"],
                competition_type=data["competition_type"],
                prepare_fn=data["prepare_fn"],
                raw_dir=data["raw_dir"],
                public_dir=data["public_dir"],
                private_dir=data["private_dir"],
                checksums=data["checksums"],
                leaderboard=data["leaderboard"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in challenge config!")


class Registry:
    """
    Registry for managing medical imaging challenges.

    Handles challenge discovery, loading, and data directory management.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._data_dir = data_dir.resolve()

    def get_challenge(self, challenge_id: str) -> Challenge:
        """
        Fetch a challenge from the registry.

        Args:
            challenge_id: Unique challenge identifier

        Returns:
            Challenge instance

        Raises:
            FileNotFoundError: If challenge config not found
            ValueError: If config is invalid
        """
        config_path = self.get_challenges_dir() / challenge_id / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Challenge '{challenge_id}' not found. "
                f"Expected config at: {config_path}"
            )

        config = load_yaml(config_path)

        checksums_path = self.get_challenges_dir() / challenge_id / "checksums.yaml"
        leaderboard_path = self.get_challenges_dir() / challenge_id / "leaderboard.csv"

        # Load description from challenge config or fall back to description file
        if "description" in config and isinstance(config["description"], str):
            if config["description"].endswith('.md'):
                # It's a file path - try to load it
                description_path = self.get_challenges_dir() / challenge_id / \
                    config["description"]
                if description_path.exists():
                    description = description_path.read_text()
                else:
                    # Try relative to repo root
                    try:
                        description_path = get_repo_dir() / \
                            config["description"]
                        description = description_path.read_text(
                        ) if description_path.exists() else config["description"]
                    except:
                        description = config["description"]
            else:
                # It's already a description string
                description = config["description"]
        else:
            description = ""

        preparer_fn = import_fn(config["preparer"])

        answers = self.get_data_dir() / config["dataset"]["answers"]
        # answers_dir is optional, default to parent directory of answers if not specified
        if "answers_dir" in config["dataset"]:
            answers_dir = self.get_data_dir() / config["dataset"]["answers_dir"]
        else:
            answers_dir = answers.parent
        gold_submission = answers
        if "gold_submission" in config["dataset"]:
            gold_submission = self.get_data_dir(
            ) / config["dataset"]["gold_submission"]
        sample_submission = self.get_data_dir(
        ) / config["dataset"]["sample_submission"]

        raw_dir = self.get_data_dir() / challenge_id / "raw"
        private_dir = self.get_data_dir() / challenge_id / "prepared" / "private"
        public_dir = self.get_data_dir() / challenge_id / "prepared" / "public"

        return Challenge.from_dict(
            {
                **config,
                "description": description,
                "answers": answers,
                "answers_dir": answers_dir,
                "sample_submission": sample_submission,
                "gold_submission": gold_submission,
                "prepare_fn": preparer_fn,
                "raw_dir": raw_dir,
                "private_dir": private_dir,
                "public_dir": public_dir,
                "checksums": checksums_path,
                "leaderboard": leaderboard_path,
            }
        )

    def get_challenges_dir(self) -> Path:
        """
        Get the challenges directory within the rexmle package.

        Returns:
            Path to challenges directory
        """
        return get_module_dir() / "challenges"

    def get_splits_dir(self) -> Path:
        """
        Get the splits directory (for challenge difficulty groupings).

        Returns:
            Path to splits directory
        """
        return get_repo_dir() / "experiments" / "splits"

    def get_lite_challenge_ids(self) -> list[str]:
        """
        List challenge IDs for the lite version (easy challenges).

        Returns:
            List of challenge IDs
        """
        lite_challenges_file = self.get_splits_dir() / "easy.txt"

        if not lite_challenges_file.exists():
            logger.warning(
                f"Lite challenges file not found: {lite_challenges_file}")
            return []

        with open(lite_challenges_file, "r") as f:
            challenge_ids = f.read().splitlines()

        return challenge_ids

    def get_data_dir(self) -> Path:
        """
        Get the data directory for cached/prepared challenge data.

        Returns:
            Path to data directory
        """
        return self._data_dir

    def set_data_dir(self, new_data_dir: Path) -> "Registry":
        """
        Set a new data directory.

        Args:
            new_data_dir: New data directory path

        Returns:
            New Registry instance with updated data directory
        """
        return Registry(new_data_dir)

    def list_challenge_ids(self) -> list[str]:
        """
        List all challenge IDs available in the registry.

        Returns:
            Sorted list of challenge IDs
        """
        challenge_configs = self.get_challenges_dir().rglob("config.yaml")
        challenge_ids = [f.parent.stem for f in sorted(challenge_configs)]

        return challenge_ids


# Global registry instance
registry = Registry()
