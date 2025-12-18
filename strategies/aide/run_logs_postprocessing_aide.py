import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_JSONL = Path("./aide_logs.jsonl")
DEFAULT_OUTPUT_DIR = Path("./aide_logs")
STRUCTURED_KEY = "analyses"


def extract_code_blocks(text: str) -> Tuple[str, List[str]]:
    """Extract code blocks from analysis text."""
    code_blocks: List[str] = []

    def _capture(match: re.Match) -> str:
        code_blocks.append(match.group(1).strip())
        return ""

    # Match ```python``` or `````` code blocks
    cleaned = re.sub(r"```(?:python)?\n(.*?)\n```", _capture, text, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, code_blocks


def extract_analyses_from_journal(journal_path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract analysis texts and code blocks from AIDE journal.json.
    AIDE journals have a "nodes" array where each node may have an "analysis" field.
    """
    try:
        journal_data = json.loads(journal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return [], []

    analyses: List[str] = []
    all_codes: List[str] = []

    nodes = journal_data.get("nodes", [])
    for node in nodes:
        analysis_text = node.get("analysis")
        if analysis_text and isinstance(analysis_text, str):
            # Extract code blocks from the analysis
            cleaned, codes = extract_code_blocks(analysis_text)
            if cleaned.strip():
                analyses.append(cleaned)
            all_codes.extend(codes)

    return analyses, all_codes


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL file and return list of records."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def save_code_blocks(challenge_id: str, code_blocks: List[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{challenge_id}_code.txt"
    content = "\n\n---\n\n".join(code_blocks)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def save_structured_payload(
    challenge_id: str,
    analyses: List[str],
    source_log: Path,
    output_dir: Path,
) -> Path:
    """Save structured analysis payload to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "challenge_id": challenge_id,
        STRUCTURED_KEY: analyses,
        "responses": [],  # AIDE doesn't have separate responses like ML-Master
        "source_log": str(source_log),
    }
    out_path = output_dir / f"{challenge_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract analysis sections from AIDE journals.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to aide_logs.jsonl manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write extracted analysis logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.jsonl)
    output_dir = args.output_dir
    written = []
    code_written = []
    skipped = []

    for entry in manifest:
        cid = entry.get("challenge_id")
        # Accept either "submission_path" or "logs_path" (for consistency with ML-Master)
        log_path = entry.get("submission_path") or entry.get("logs_path")
        if not cid or not log_path:
            skipped.append((cid or "unknown", "missing required keys"))
            continue

        log_path = Path(log_path)
        if not log_path.exists():
            skipped.append((cid, f"missing log file: {log_path}"))
            continue

        analyses, code_blocks = extract_analyses_from_journal(log_path)
        if not analyses:
            skipped.append((cid, "no analysis sections found"))
            continue

        # Save structured payload
        out_path = save_structured_payload(
            challenge_id=cid,
            analyses=analyses,
            source_log=log_path,
            output_dir=output_dir,
        )
        written.append(out_path)

        # Save code blocks if any
        if code_blocks:
            code_path = save_code_blocks(cid, code_blocks, output_dir)
            code_written.append(code_path)

    print(f"Wrote {len(written)} structured analysis files to {output_dir}")
    if code_written:
        print(f"Wrote {len(code_written)} code files to {output_dir}")
    if skipped:
        print("Skipped entries:")
        for cid, reason in skipped:
            print(f"- {cid}: {reason}")


if __name__ == "__main__":
    main()
