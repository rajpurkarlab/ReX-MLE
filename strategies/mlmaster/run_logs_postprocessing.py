import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_JSONL = Path("./mlmaster_logs.jsonl")
DEFAULT_OUTPUT_DIR = Path("./mlmaster_logs")
STRUCTURED_KEY = "analyses"


def strip_agent_prefix(line: str) -> str:
    """Remove leading timestamp/INFO/Agent prefix when present."""
    # Example prefix: [2025-11-20 16:53:07] [INFO] [Agent]
    match = re.match(r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+\[INFO\]\s+\[Agent\]\s+(.*)$", line)
    if match:
        return match.group(1)
    return line


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("INFO:ml-master:response"):
        return False
    if stripped.startswith("INFO:ml-master:"):
        return True
    if stripped.startswith("INFO:"):
        return True
    if stripped.startswith(("execution done", "ExecutionResult")):
        return True
    if stripped.startswith("ValueError:"):
        return True
    return False


def extract_analysis_sections(text: str) -> List[str]:
    lines = text.splitlines()
    sections: List[List[str]] = []
    current: List[str] = []
    capturing = False

    for line in lines:
        stripped = strip_agent_prefix(line)

        if "# Analysis" in stripped:
            if current:
                sections.append(current)
                current = []
            capturing = True
            current.append(stripped)
            continue
        if capturing:
            if is_noise_line(stripped):
                sections.append(current)
                current = []
                capturing = False
                continue
            current.append(stripped)

    if current:
        sections.append(current)

    cleaned = []
    for sec in sections:
        block = "\n".join(sec).strip()
        if block:
            cleaned.append(block)
    return cleaned


def split_out_code_blocks(section: str) -> Tuple[str, List[str]]:
    """Remove ```python``` blocks from a section and return code snippets separately."""
    code_blocks: List[str] = []

    def _capture(match: re.Match) -> str:
        # Save code without the surrounding fences; strip trailing/leading space for neatness.
        code_blocks.append(match.group(1).strip())
        return ""

    cleaned = re.sub(r"```python\n(.*?)\n```", _capture, section, flags=re.DOTALL)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, code_blocks


def split_analysis_and_responses(section: str) -> Tuple[str, List[str]]:
    """Separate the analysis text from ml-master responses inside a section."""
    analysis_lines: List[str] = []
    responses: List[str] = []
    current_response: List[str] = []
    in_response = False

    def _flush_response() -> None:
        nonlocal current_response
        if current_response:
            resp = "\n".join(current_response).strip()
            if resp:
                responses.append(resp)
        current_response = []

    for line in section.splitlines():
        if line.strip() == "# Analysis":
            continue

        if line.startswith("INFO:ml-master:response"):
            _flush_response()
            in_response = True
            content = re.sub(r"^INFO:ml-master:response(?: without think)?:\s*", "", line).strip()
            if content:
                current_response.append(content)
            continue

        if in_response:
            current_response.append(line)
        else:
            analysis_lines.append(line)

    _flush_response()

    analysis_text = "\n".join(analysis_lines).strip()
    return analysis_text, responses


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
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
    responses: List[str],
    source_log: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "challenge_id": challenge_id,
        STRUCTURED_KEY: analyses,
        "responses": responses,
        "source_log": str(source_log),
    }
    out_path = output_dir / f"{challenge_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract '# Analysis' sections from ml-master logs.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to mlmaster_logs.jsonl manifest.",
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
        # Accept either new key ("logs_path") or legacy key ("submission_path")
        log_path = entry.get("logs_path") or entry.get("submission_path")
        if not cid or not log_path:
            skipped.append((cid or "unknown", "missing required keys"))
            continue

        log_path = Path(log_path)
        if not log_path.exists():
            skipped.append((cid, f"missing log file: {log_path}"))
            continue

        sections = extract_analysis_sections(log_path.read_text(encoding="utf-8"))
        if not sections:
            skipped.append((cid, "no '# Analysis' sections found"))
            continue

        analyses: List[str] = []
        responses: List[str] = []
        code_blocks: List[str] = []

        for section in sections:
            cleaned, codes = split_out_code_blocks(section)
            analysis_text, response_blocks = split_analysis_and_responses(cleaned)
            if analysis_text:
                analyses.append(analysis_text)
            responses.extend(response_blocks)
            code_blocks.extend(codes)

        if analyses or responses:
            out_path = save_structured_payload(
                challenge_id=cid,
                analyses=analyses,
                responses=responses,
                source_log=log_path,
                output_dir=output_dir,
            )
            written.append(out_path)

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
