import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT_DIR = Path("./mlmaster/mlmaster_strategies_scores")
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "mlmaster_scores.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-line strategy scores into per-strategy averages."
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_scores.json files (per-log outputs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write aggregated scores JSON.",
    )
    return parser.parse_args()


def collect_scores(scores_dir: Path) -> Dict[int, Dict]:
    agg: Dict[int, Dict] = {}
    files = sorted(
        [
            p
            for p in scores_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".json"
            and not p.name.endswith("mlmaster_scores.json")
        ]
    )

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results = payload.get("results") or []
        for item in results:
            try:
                sid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(item.get("strategy") or "").strip() or f"strategy-{sid}"
            score_raw = item.get("score", 0)
            try:
                score = 1 if int(score_raw) == 1 else 0
            except (TypeError, ValueError):
                score = 0

            if sid not in agg:
                agg[sid] = {"id": sid, "strategy": name, "scores": []}
            agg[sid]["scores"].append(score)
            # Keep the latest non-empty name.
            if name:
                agg[sid]["strategy"] = name

    return agg


def compute_summary(agg: Dict[int, Dict], num_logs: int) -> Dict:
    strategies: List[Dict] = []
    for sid in sorted(agg.keys()):
        scores = agg[sid]["scores"]
        count = len(scores)
        total = sum(scores)
        mean = total / count if count else 0.0
        strategies.append(
            {
                "id": sid,
                "strategy": agg[sid]["strategy"],
                "mean_score": mean,
                "count": count,
                "sum_score": total,
            }
        )

    return {"num_logs": num_logs, "strategies": strategies}


def main() -> None:
    args = parse_args()
    scores_dir = args.scores_dir.expanduser().resolve()
    if not scores_dir.exists():
        raise SystemExit(f"Scores directory not found: {scores_dir}")

    agg = collect_scores(scores_dir)
    num_logs = len(
        [
            p
            for p in scores_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".json"
            and not p.name.endswith("mlmaster_scores.json")
        ]
    )
    summary = compute_summary(agg, num_logs=num_logs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote aggregated scores to {args.output}")


if __name__ == "__main__":
    main()
