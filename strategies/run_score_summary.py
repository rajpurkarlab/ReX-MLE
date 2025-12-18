import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SCORES_DIR = Path("./mlmaster/mlmaster_strategies_scores")
DEFAULT_OUTPUT = DEFAULT_SCORES_DIR / "mlmaster_score_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize binary strategy scores per challenge.",
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=DEFAULT_SCORES_DIR,
        help="Directory containing *_scores.json files produced by the scorer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the aggregated summary JSON.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing to disk and only print the summary.",
    )
    return parser.parse_args()


def list_score_files(scores_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in scores_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".json"
            and not p.name.endswith("mlmaster_score_summary.json")
            and not p.name.endswith("mlmaster_scores.json")
        ]
    )


def normalize_challenge_name(path: Path) -> str:
    stem = path.stem
    for suffix in ("_scores_all", "_scores"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def collect_counts(results: List[Dict]) -> Dict[int, Dict[str, object]]:
    counts: Dict[int, Dict[str, object]] = {}

    for entry in results:
        try:
            sid = int(entry.get("id"))
        except (TypeError, ValueError):
            continue

        strategy = str(entry.get("strategy") or "").strip() or f"strategy-{sid}"
        raw_score = entry.get("score", 0)
        try:
            score = 1 if int(raw_score) == 1 else 0
        except (TypeError, ValueError):
            score = 0

        if sid not in counts:
            counts[sid] = {"id": sid, "strategy": strategy, "total": 0, "count_1": 0}

        counts[sid]["total"] += 1
        counts[sid]["count_1"] += score
        if strategy:
            counts[sid]["strategy"] = strategy

    return counts


def build_summary(counts: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for sid in sorted(counts.keys()):
        total = int(counts[sid]["total"])
        ones = int(counts[sid]["count_1"])
        zeros = total - ones
        ratio = ones / total if total else 0.0

        summary.append(
            {
                "id": sid,
                "strategy": counts[sid]["strategy"],
                "count": total,
                "count_0": zeros,
                "count_1": ones,
                "ratio_1": ratio,
            }
        )
    return summary


def summarize_challenge(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    results = payload.get("results") or []
    counts = collect_counts(results)

    return {
        "file": path.name,
        "num_results": len(results),
        "strategies": build_summary(counts),
    }


def main() -> None:
    args = parse_args()
    scores_dir = args.scores_dir.expanduser().resolve()

    if not scores_dir.exists():
        raise SystemExit(f"Scores directory not found: {scores_dir}")

    files = list_score_files(scores_dir)
    challenges = {}

    for path in files:
        name = normalize_challenge_name(path)
        summary = summarize_challenge(path)
        if summary:
            challenges[name] = summary

    payload = {
        "scores_dir": str(scores_dir),
        "num_files": len(challenges),
        "challenges": challenges,
    }

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote summary to {args.output}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
