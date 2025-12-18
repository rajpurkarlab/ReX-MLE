#!/usr/bin/env python3
"""
Convert rdagent unpickled runs to strategy analysis JSON format.

This script:
1. Reads unpickled_data.txt files from rdagent/unpickled_runs/<challenge>/
2. Extracts text sections (filtering out code sections with import statements)
3. Creates JSON in the same format as aide_logs (with challenge_id and analyses)
4. Saves to strategies/rdagent_logs/<challenge>.json
"""

import json
import os
import re
from pathlib import Path


def extract_analyses_from_text(text_content):
    """
    Extract analysis text from unpickled_data.txt file.

    The text file contains sections separated by "=" lines with file paths.
    Each section is unpickled data. We extract sections that are not code
    (identified by lack of import statements).

    Args:
        text_content: Full content of unpickled_data.txt

    Returns:
        List of analysis strings
    """
    analyses = []
    lines = text_content.split('\n')

    current_section = []
    in_section = False
    skip_header = True

    for line in lines:
        # Detect section separators
        if line.startswith('=' * 10):
            # End of previous section - save if we have content
            if current_section and in_section and not skip_header:
                section_text = '\n'.join(current_section).strip()

                # Only include if it's NOT code (check for import statements)
                if section_text and not is_code(section_text):
                    analyses.append(section_text)

            current_section = []
            in_section = True
            skip_header = True

        elif in_section:
            # Check if this is a header line with "File X/Y:"
            if line.strip().startswith('File ') and '/' in line:
                skip_header = False
            # Skip the header lines at start of section
            elif not skip_header or (line.strip() and not line.startswith('File ')):
                skip_header = False
                if line.strip():
                    current_section.append(line)

    # Don't forget the last section
    if current_section and in_section and not skip_header:
        section_text = '\n'.join(current_section).strip()
        if section_text and not is_code(section_text):
            analyses.append(section_text)

    return analyses


def is_code(text):
    """
    Detect if text is code by looking for import statements or Python object repr.

    Args:
        text: Text to check

    Returns:
        True if text contains import statements or looks like Python object repr
    """
    lines = text.split('\n')
    first_line = lines[0].strip() if lines else ""

    # Check for import statements
    for line in lines[:10]:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            return True

    # Check if it starts with a Python dict/list/object repr
    if first_line.startswith(('{', '[', '<')) and ':' in first_line:
        return True

    return False


def process_challenge(challenge_id, input_dir, output_dir):
    """
    Process unpickled_data.txt for a challenge and create JSON.

    Args:
        challenge_id: Challenge identifier (e.g., 'isles22')
        input_dir: Path to rdagent/unpickled_runs/<challenge>/
        output_dir: Path to strategies/rdagent_logs/

    Returns:
        True if successful, False otherwise
    """
    input_file = Path(input_dir) / challenge_id / "unpickled_data.txt"
    output_file = Path(output_dir) / f"{challenge_id}.json"

    if not input_file.exists():
        print(f"  ✗ Input file not found: {input_file}")
        return False

    try:
        # Read the unpickled text file
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract analyses (filtering out code)
        analyses = extract_analyses_from_text(content)

        if not analyses:
            print(f"  ✗ No analyses extracted from {challenge_id}")
            return False

        # Create the JSON structure matching aide_logs format
        output_data = {
            "challenge_id": challenge_id,
            "analyses": analyses,
        }

        # Write output JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file}")
        print(f"    - Extracted {len(analyses)} analyses")

        return True

    except Exception as e:
        print(f"  ✗ Error processing {challenge_id}: {e}")
        return False


def main():
    """Main entry point."""
    input_base_dir = "./unpickled_runs"
    output_dir = "./rdagent_logs"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all challenge directories
    input_base = Path(input_base_dir)
    if not input_base.exists():
        print(f"Error: Input directory not found: {input_base_dir}")
        return

    challenges = sorted([d.name for d in input_base.iterdir() if d.is_dir()])

    if not challenges:
        print(f"No challenge directories found in {input_base_dir}")
        return

    print(f"Found {len(challenges)} challenges to process:\n")

    successful = 0
    failed = 0

    for challenge_id in challenges:
        print(f"Processing {challenge_id}...")
        if process_challenge(challenge_id, input_base_dir, output_dir):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(challenges)}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
