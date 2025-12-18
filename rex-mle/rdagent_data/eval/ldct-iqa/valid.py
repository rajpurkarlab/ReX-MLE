"""
Submission format validation for ldct-iqa
"""
from pathlib import Path
import pandas as pd

# Check if our submission file exists
assert Path("submission.csv").exists(), "Error: submission.csv not found"

submission_lines = Path("submission.csv").read_text().splitlines()
test_lines = Path("submission_test.csv").read_text().splitlines()

is_valid = len(submission_lines) == len(test_lines)

if is_valid:
    message = f"✓ submission.csv and submission_test.csv have the same number of lines ({len(submission_lines)})."
else:
    message = (
        f"✗ submission.csv has {len(submission_lines)} lines, "
        f"while submission_test.csv has {len(test_lines)} lines."
    )

print(message)

if not is_valid:
    raise AssertionError("Submission is invalid: line count mismatch")

# Additional validation: check if files are valid CSVs
try:
    sub_df = pd.read_csv("submission.csv")
    test_df = pd.read_csv("submission_test.csv")

    # Check if columns match
    if list(sub_df.columns) != list(test_df.columns):
        raise AssertionError(
            f"Column mismatch: submission has {list(sub_df.columns)}, "
            f"test has {list(test_df.columns)}"
        )

    print(f"✓ Columns match: {list(sub_df.columns)}")
    print("✓ Submission format is valid")

except Exception as e:
    raise AssertionError(f"Failed to parse CSV files: {e}")
