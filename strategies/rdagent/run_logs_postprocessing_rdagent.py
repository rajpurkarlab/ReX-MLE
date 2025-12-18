#!/usr/bin/env python3
"""
Extract agent reasoning, analysis, and feedback from unpickled rdagent output files.
Creates JSON files with structured analysis for each challenge.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


DEFAULT_INPUT_DIR = Path("./unpickled_runs")
DEFAULT_OUTPUT_DIR = Path("./rdagent_logs")


def remove_code_blocks_by_escaped_newlines(content: str) -> str:
    """
    Remove code blocks from content.

    Code blocks are marked by:
    - Pattern 1: Literal backticks with code (```...```)
    - Pattern 2: Diff sections (--- file, +++ file, @@ ... @@)
    - Pattern 3: "Modified code according to hypothesis:" followed by code
    - Pattern 4: Template syntax ({{ exp.experiment_workspace.all_codes }})
    - Pattern 5: "### Complete Code" sections
    """
    result = content

    # Remove literal triple backtick code blocks: ```...[code]...```
    result = re.sub(r'```[\s\S]*?```', '', result, flags=re.DOTALL)

    # Remove "Modified code according to hypothesis:" followed by code diff until next analysis keyword
    result = re.sub(r'Modified code according to hypothesis:[\s\S]*?(?=\n(?:###|Hypothesis|Results|Target|Chosen|Reason|Observations|New\s+Hypothesis|Decision|$))', '', result, flags=re.DOTALL)

    # Remove diff sections marked by --- and +++ followed by @@ diff markers
    result = re.sub(r'^---\s+\S+.*?(?=^[A-Z]|$)', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'^\+\+\+\s+\S+.*?(?=^[A-Z]|$)', '', result, flags=re.MULTILINE | re.DOTALL)
    result = re.sub(r'@@\s+[\-\+\d\s,]+@@[\s\S]*?(?=\n[A-Z\-\+]|\Z)', '', result, flags=re.DOTALL)

    # Remove Jinja template code references: {{ exp.experiment_workspace.all_codes }}
    result = re.sub(r'\{\{\s*exp\.experiment_workspace\.all_codes\s*\}\}', '', result)

    # Remove "### Complete Code" sections followed by actual code until we hit "### Hypothesis" or framework keywords
    result = re.sub(r'###\s*(?:Complete\s+)?Code[^\n]*\n(?:Here is the complete code[^\n]*\n)?[\s\S]*?(?=###\s*(?:Hypothesis|Results|Target|Chosen|Reason|Observations|New\s+Hypothesis|Decision)|\Z)', '', result)

    # Remove "Please refer to these..." sections that contain code references
    result = re.sub(r'Please refer to these[\s\S]*?(?=\n[A-Z]|\Z)', '', result)

    # Clean up multiple spaces and excessive newlines
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\n\n+', '\n', result)

    return result.strip()


def extract_feedback_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract structured feedback blocks from unpickled text.

    Looks for patterns like:
    - "Reason: [Experiment Analysis]..." or similar
    - "Observations: ..."
    - "Hypothesis Evaluation: ..."
    - "Code Change Summary: ..."

    Removes code blocks (identified by many \\n patterns) while preserving analysis.
    """
    blocks = []

    # Use regex to find feedback sections - look for lines that start with the keywords
    # followed by actual content (not JSON/template artifacts)
    patterns = [
        (r"Reason:\s*\[Experiment Analysis\](.*?)(?=\n(?:Observations|Hypothesis Evaluation|Code Change Summary|New Hypothesis|Reason:|$))", "Reason"),
        (r"Observations:\s*(.*?)(?=\n(?:Reason|Hypothesis Evaluation|Code Change Summary|New Hypothesis|$))", "Observations"),
        (r"Hypothesis Evaluation:\s*(.*?)(?=\n(?:Reason|Observations|Code Change Summary|New Hypothesis|$))", "Hypothesis Evaluation"),
        (r"Code Change Summary:\s*(.*?)(?=\n(?:Reason|Observations|Hypothesis Evaluation|New Hypothesis|$))", "Code Change Summary"),
    ]

    for pattern, label in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1).strip()

            # Remove code blocks from content (sequences with many \\n)
            content = remove_code_blocks_by_escaped_newlines(content)
            content = content.strip()

            # Skip if content is too short or is template/object representation
            if len(content) > 50 and not content.startswith("{{") and not content.startswith("{"):
                # Clean up the content
                content = re.sub(r'\\n', ' ', content)  # Replace escaped newlines
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                blocks.append({
                    'type': label,
                    'content': content[:1000]  # Limit to 1000 chars to avoid huge blocks
                })

    return blocks


def format_feedback_as_analysis(feedbacks: List[Dict[str, str]]) -> str:
    """Format extracted feedbacks as an analysis block."""
    if not feedbacks:
        return ""

    parts = []

    for fb in feedbacks:
        fb_type = fb.get('type', 'Unknown')
        content = fb.get('content', '').strip()

        if content:
            parts.append(f"{fb_type}: {content}")

    return "\n".join(parts)


def extract_all_analyses_from_txt(txt_path: Path) -> List[str]:
    """
    Extract all analysis sections from an unpickled txt file.
    Looks for feedback blocks and formats them as analyses.
    """
    if not txt_path.exists():
        return []

    try:
        text = txt_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return []

    analyses = []

    # Extract feedback blocks
    feedbacks = extract_feedback_blocks(text)

    if feedbacks:
        # Group consecutive feedbacks into analysis blocks
        current_block = []
        for fb in feedbacks:
            current_block.append(fb)
            # If we've collected a few feedbacks, format as an analysis
            if len(current_block) >= 3 or fb.get('type') == 'Hypothesis Evaluation':
                formatted = format_feedback_as_analysis(current_block)
                if formatted:
                    analyses.append(formatted)
                current_block = []

        # Add any remaining feedback
        if current_block:
            formatted = format_feedback_as_analysis(current_block)
            if formatted:
                analyses.append(formatted)

    # Also try to extract target problem / hypothesis patterns
    target_problem_pattern = r"Target Problem:\s*(.+?)(?=\n[A-Z]|\nCode|\Z)"
    for match in re.finditer(target_problem_pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Remove code blocks from target problem content
        content = remove_code_blocks_by_escaped_newlines(content)
        content = content.strip()
        if len(content) > 50:
            analysis = f"Target Problem: {content}"
            if analysis not in analyses:
                analyses.append(analysis)

    return analyses


def save_challenge_analysis(
    challenge_id: str,
    analyses: List[str],
    source_txt: Path,
    output_dir: Path,
) -> Path:
    """Save analysis as JSON in the format matching dentex.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean analyses to remove problematic characters and escape sequences
    clean_analyses = []
    for analysis in analyses:
        # Remove literal backslash-n sequences (from unpickled data)
        clean = analysis.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
        # Remove actual control characters (ASCII < 32 except whitespace)
        clean = ''.join(c if ord(c) >= 32 or c in ' \n\r\t' else '' for c in clean)
        # Collapse multiple spaces/newlines into single spaces
        clean = ' '.join(clean.split())
        if clean and len(clean) > 10:
            clean_analyses.append(clean)

    payload = {
        "challenge_id": challenge_id,
        "analyses": clean_analyses,
        "responses": [],
        "source_log": str(source_txt),
    }

    out_path = output_dir / f"{challenge_id}.json"
    # Use json.dumps which properly handles all escaping
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    out_path.write_text(json_str, encoding='utf-8')
    return out_path


def process_all_challenges(input_dir: Path, output_dir: Path) -> None:
    """Process all challenge unpickled data files."""
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    # Find all challenge directories
    challenge_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    written = []
    skipped = []

    for challenge_dir in challenge_dirs:
        challenge_id = challenge_dir.name
        txt_file = challenge_dir / "unpickled_data.txt"

        if not txt_file.exists():
            skipped.append((challenge_id, f"missing unpickled_data.txt"))
            print(f"⚠ {challenge_id}: No unpickled_data.txt found")
            continue

        # Extract analyses from the unpickled txt
        analyses = extract_all_analyses_from_txt(txt_file)

        if analyses:
            # Save as JSON
            out_path = save_challenge_analysis(
                challenge_id=challenge_id,
                analyses=analyses,
                source_txt=txt_file,
                output_dir=output_dir,
            )
            written.append(out_path)
            print(f"✓ {challenge_id}: Extracted {len(analyses)} analysis blocks")
        else:
            skipped.append((challenge_id, "no analysis blocks found"))
            print(f"⚠ {challenge_id}: No analysis blocks extracted")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Analysis files created: {len(written)}")
    if skipped:
        print(f"\nSkipped entries: {len(skipped)}")
        for cid, reason in skipped[:10]:
            print(f"  - {cid}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    print(f"Output directory: {output_dir}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract agent reasoning from unpickled rdagent output files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing unpickled_runs subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write analysis JSON files",
    )
    args = parser.parse_args()

    process_all_challenges(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
