"""
Command-line interface for rex-mle.
"""

import click
import pandas as pd
from pathlib import Path
from rexmle.registry import registry
from rexmle.grade import grade_jsonl


@click.group()
def cli():
    """Med-MLE-Bench: Medical Imaging Challenge Benchmark"""
    pass


@cli.command()
def list():
    """List all available challenges."""
    challenges = registry.list_challenge_ids()

    click.echo(f"\nAvailable challenges: {len(challenges)}\n")
    for challenge_id in challenges:
        click.echo(f"  • {challenge_id}")
    click.echo()


@cli.command()
@click.argument('challenge_id')
def info(challenge_id):
    """Show detailed information about a challenge."""
    try:
        challenge = registry.get_challenge(challenge_id)

        click.echo(f"\n{'='*70}")
        click.echo(f"Challenge: {challenge.name}")
        click.echo(f"{'='*70}")
        click.echo(f"ID:           {challenge.id}")
        click.echo(f"Type:         {challenge.competition_type}")
        click.echo(f"\nPaths:")
        click.echo(f"  Raw data:   {challenge.raw_dir}")
        click.echo(f"  Public:     {challenge.public_dir}")
        click.echo(f"  Private:    {challenge.private_dir}")
        click.echo(f"\nGrading:")
        click.echo(f"  Metric:     {challenge.grader.name}")
        click.echo(f"\nDescription:")
        desc_preview = challenge.description[:500] if challenge.description else "No description available"
        click.echo(f"{desc_preview}...")
        click.echo(f"{'='*70}\n")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"Error loading challenge: {e}", err=True)
        exit(1)


@cli.command()
@click.argument('challenge_id')
@click.option('--force', is_flag=True, help='Force re-download and re-preparation')
@click.option('--skip-download', is_flag=True, help='Skip download, only run prepare')
def prepare(challenge_id, force, skip_download):
    """
    Prepare data for a challenge.

    This downloads data from Zenodo and runs the challenge-specific prepare.py
    to create train/test splits.
    """
    try:
        challenge = registry.get_challenge(challenge_id)

        click.echo(f"\n{'='*70}")
        click.echo(f"Preparing challenge: {challenge.name}")
        click.echo(f"{'='*70}\n")

        # Check if already prepared
        if challenge.public_dir.exists() and not force:
            click.echo("✓ Challenge already prepared.")
            click.echo("  Use --force to re-prepare.\n")
            return

        # Create directories
        challenge.raw_dir.mkdir(parents=True, exist_ok=True)
        challenge.public_dir.mkdir(parents=True, exist_ok=True)
        challenge.private_dir.mkdir(parents=True, exist_ok=True)

        # Run prepare function
        click.echo("Running prepare function...\n")

        challenge.prepare_fn(
            challenge.raw_dir,
            challenge.public_dir,
            challenge.private_dir
        )

        click.echo(f"\n{'='*70}")
        click.echo("✓ Challenge prepared successfully!")
        click.echo(f"{'='*70}")
        click.echo(f"\nData locations:")
        click.echo(f"  Public (for training):  {challenge.public_dir}")
        click.echo(f"  Private (for grading):  {challenge.private_dir}")
        click.echo()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"Error preparing challenge: {e}", err=True)
        import traceback
        traceback.print_exc()
        exit(1)


@cli.command()
@click.argument('challenge_id')
@click.argument('submission_file')
@click.option('--submission-dir', type=click.Path(exists=True), help='Directory containing submission files (for segmentation tasks)')
@click.option('--show-ranking', is_flag=True, help='Show medal ranking based on leaderboard')
def grade(challenge_id, submission_file, submission_dir, show_ranking):
    """
    Grade a submission for a challenge.

    SUBMISSION_FILE: Path to submission CSV file

    For segmentation tasks, use --submission-dir to specify the directory containing mask files.
    """
    try:
        challenge = registry.get_challenge(challenge_id)

        click.echo(f"\n{'='*70}")
        click.echo(f"Grading submission for: {challenge.name}")
        click.echo(f"{'='*70}\n")

        # Load submission
        submission_path = Path(submission_file)
        if not submission_path.exists():
            click.echo(f"Error: Submission file not found: {submission_file}", err=True)
            exit(1)

        submission = pd.read_csv(submission_path)
        click.echo(f"Loaded submission: {len(submission)} rows")

        # Set submission directory
        if submission_dir:
            submission_dir = Path(submission_dir)
            click.echo(f"Submission directory: {submission_dir}")
        else:
            submission_dir = submission_path.parent

        # Load answers
        if not challenge.answers.exists():
            click.echo(f"Error: Ground truth not found: {challenge.answers}", err=True)
            click.echo("Did you run 'prepare' first?", err=True)
            exit(1)

        answers = pd.read_csv(challenge.answers)
        click.echo(f"Loaded ground truth: {len(answers)} rows\n")

        # Grade
        click.echo("Computing score...")
        result = challenge.grader(submission, answers, submission_dir=submission_dir, answers_dir=challenge.answers_dir)

        if result is None:
            click.echo("\n✗ Invalid submission or grading error", err=True)
            click.echo("Check logs above for details.", err=True)
            exit(1)

        # Show results
        click.echo(f"\n{'='*70}")

        # Handle dictionary results (multiple metrics)
        if isinstance(result, dict):
            click.echo(f"Scores ({challenge.grader.name}):")
            click.echo(f"{'='*70}")

            # Display all metrics
            for metric_name, metric_value in result.items():
                click.echo(f"  {metric_name:20s}: {metric_value:.6f}")

            # Use 'overall' metric for ranking if available, otherwise use first metric
            if 'overall' in result:
                score = result['overall']
            else:
                score = next(iter(result.values()))

        # Handle single score results (backward compatibility)
        else:
            score = result
            click.echo(f"Score ({challenge.grader.name}): {score:.6f}")
            click.echo(f"{'='*70}")

        # Show ranking if requested
        if show_ranking and challenge.leaderboard.exists():
            leaderboard = pd.read_csv(challenge.leaderboard)
            ranking = challenge.grader.rank_score(score, leaderboard)

            click.echo(f"\nRanking:")
            if ranking['gold_medal']:
                click.echo("  🥇 Gold Medal!")
            elif ranking['silver_medal']:
                click.echo("  🥈 Silver Medal!")
            elif ranking['bronze_medal']:
                click.echo("  🥉 Bronze Medal!")
            elif ranking['above_median']:
                click.echo("  ✓ Above Median")
            else:
                click.echo("  • Below Median")

            click.echo(f"\n  Thresholds:")
            click.echo(f"    Gold:   {ranking['gold_threshold']:.6f}")
            click.echo(f"    Silver: {ranking['silver_threshold']:.6f}")
            click.echo(f"    Bronze: {ranking['bronze_threshold']:.6f}")
            click.echo(f"    Median: {ranking['median_threshold']:.6f}")

        click.echo()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        exit(1)
    except Exception as e:
        click.echo(f"Error grading submission: {e}", err=True)
        import traceback
        traceback.print_exc()
        exit(1)


@cli.command('grade-batch')
@click.option('--submission', type=click.Path(exists=True), required=True, help='Path to JSONL file of submissions')
@click.option('--output-dir', type=click.Path(), required=True, help='Directory where grading report will be saved')
@click.option('--data-dir', type=click.Path(), help='Path to data directory (overrides default)')
@click.option('--suffix', type=str, default='', help='Optional text to append to the filename (before .json)')
def grade_batch(submission, output_dir, data_dir, suffix):
    """
    Grade multiple submissions from a JSONL file.

    JSONL format (one submission per line):
    {"challenge_id": "topcow-track1-task1", "submission_path": "/path/to/submission.csv"}

    For segmentation tasks, add "submission_dir":
    {"challenge_id": "topcow-track1-task1", "submission_path": "/path/to/submission.csv", "submission_dir": "/path/to/masks"}

    Similar to: mlebench grade --submission submissions.jsonl --output-dir metrics/
    """
    try:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        submission_path = Path(submission)
        output_path = Path(output_dir)

        # Use custom registry if data-dir is provided
        if data_dir:
            from rexmle.registry import Registry
            custom_registry = Registry(data_dir=Path(data_dir))
            grade_jsonl(submission_path, output_path, registry=custom_registry, suffix=suffix)
        else:
            grade_jsonl(submission_path, output_path, suffix=suffix)

        click.echo("\n✓ Batch grading completed successfully!")

    except Exception as e:
        click.echo(f"Error during batch grading: {e}", err=True)
        import traceback
        traceback.print_exc()
        exit(1)


@cli.command()
@click.option('--data-dir', type=click.Path(), help='Show or set custom data directory')
def config(data_dir):
    """Show or set configuration."""
    if data_dir:
        new_dir = Path(data_dir).resolve()
        click.echo(f"Setting data directory to: {new_dir}")
        # Note: This would require persistent config storage
        click.echo("(Feature not yet implemented - data dir is set per session)")
    else:
        click.echo(f"\nConfiguration:")
        click.echo(f"  Data directory: {registry.get_data_dir()}")
        click.echo(f"  Challenges directory: {registry.get_challenges_dir()}")
        click.echo()


if __name__ == '__main__':
    cli()
