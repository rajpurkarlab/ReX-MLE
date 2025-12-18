#!/usr/bin/env python3
"""
Script to apply no-docker patches to the installed rdagent package.

This script modifies the rdagent package installed in the conda environment
to support running without Docker when environment variables are set.
"""

import os
import sys
from pathlib import Path


def find_rdagent_dir():
    """Find the installed rdagent package directory."""
    try:
        import rdagent
        # Handle both regular packages (__file__) and namespace packages (__path__)
        if hasattr(rdagent, '__file__') and rdagent.__file__ is not None:
            rdagent_dir = Path(rdagent.__file__).parent
        elif hasattr(rdagent, '__path__'):
            rdagent_dir = Path(rdagent.__path__[0])
        else:
            raise ImportError("Cannot determine rdagent installation directory")
        return rdagent_dir
    except ImportError:
        print("ERROR: rdagent package not found. Please install it first:")
        print("  pip install rdagent")
        sys.exit(1)


def apply_env_patch(rdagent_dir):
    """Apply patches to rdagent/utils/env.py"""
    env_file = rdagent_dir / "utils" / "env.py"

    if not env_file.exists():
        print(f"ERROR: File not found: {env_file}")
        return False

    print(f"Patching {env_file}...")

    # Read the file
    content = env_file.read_text()

    # Patch 1: DockerEnv.prepare() method
    search_1 = '''    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image if it doesn't exist
        """
        client = docker.from_env()'''

    replace_1 = '''    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image if it doesn't exist
        """
        # Skip Docker if using conda/local mode (no-docker patch)
        import os
        if os.getenv('DS_Coder_CoSTEER_ENV_TYPE') in ['conda', 'local'] or os.getenv('DS_Runner_CoSTEER_ENV_TYPE') in ['conda', 'local']:
            logger.warning("Docker prepare() called but conda/local mode is enabled - skipping")
            return
        client = docker.from_env()'''

    if search_1 in content:
        content = content.replace(search_1, replace_1, 1)
        print("  ✓ Applied patch 1: DockerEnv.prepare()")
    else:
        print("  ⚠ Patch 1 already applied or pattern not found")

    # Patch 2: DockerEnv._run() method
    search_2 = '''        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        client = docker.from_env()'''

    replace_2 = '''        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        # Skip Docker if using conda/local mode (no-docker patch)
        import os
        if os.getenv('DS_Coder_CoSTEER_ENV_TYPE') in ['conda', 'local'] or os.getenv('DS_Runner_CoSTEER_ENV_TYPE') in ['conda', 'local']:
            raise RuntimeError("DockerEnv.run() called but conda/local mode is enabled. This should not happen!")
        client = docker.from_env()'''

    if search_2 in content:
        content = content.replace(search_2, replace_2, 1)
        print("  ✓ Applied patch 2: DockerEnv._run()")
    else:
        print("  ⚠ Patch 2 already applied or pattern not found")

    # Write the patched file
    env_file.write_text(content)
    print(f"  ✓ Saved changes to {env_file}")
    return True


def apply_kaggle_crawler_patch(rdagent_dir):
    """Apply patches to rdagent/scenarios/kaggle/kaggle_crawler.py"""
    crawler_file = rdagent_dir / "scenarios" / "kaggle" / "kaggle_crawler.py"

    if not crawler_file.exists():
        print(f"ERROR: File not found: {crawler_file}")
        return False

    print(f"Patching {crawler_file}...")

    # Read the file
    content = crawler_file.read_text()

    # Patch: download_data() function
    search = '''def download_data(competition: str, settings: ExtendedBaseSettings = KAGGLE_IMPLEMENT_SETTING) -> None:
    local_path = settings.local_data_path
    if settings.if_using_mle_data:
        zipfile_path = f"{local_path}/zip_files"
        zip_competition_path = Path(zipfile_path) / competition

        if not zip_competition_path.exists():
            mleb_env = MLEBDockerEnv()
            mleb_env.prepare()
            (Path(zipfile_path)).mkdir(parents=True, exist_ok=True)
            mleb_env.run(
                f"mlebench prepare -c {competition} --data-dir ./zip_files",
                local_path=local_path,
                running_extra_volume={str(Path("~/.kaggle").expanduser().absolute()): "/root/.kaggle"},
            )

        if not (Path(local_path) / competition).exists() or list((Path(local_path) / competition).iterdir()) == []:
            (Path(local_path) / competition).mkdir(parents=True, exist_ok=True)

            mleb_env = MLEBDockerEnv()
            mleb_env.prepare()
            mleb_env.run(f"cp -r ./zip_files/{competition}/prepared/public/* ./{competition}", local_path=local_path)'''

    replace = '''def download_data(competition: str, settings: ExtendedBaseSettings = KAGGLE_IMPLEMENT_SETTING) -> None:
    import os
    local_path = settings.local_data_path
    if settings.if_using_mle_data:
        # Check if running in conda/local mode
        is_no_docker = os.getenv('DS_Coder_CoSTEER_ENV_TYPE') in ['conda', 'local'] or \\
                      os.getenv('DS_Runner_CoSTEER_ENV_TYPE') in ['conda', 'local']

        zipfile_path = f"{local_path}/zip_files"
        zip_competition_path = Path(zipfile_path) / competition

        if not zip_competition_path.exists():
            if is_no_docker:
                # In no-docker mode, skip download and assume data is pre-prepared
                logger.warning(f"No-docker mode: Skipping zip file download for {competition}. Data should be pre-prepared.")
            else:
                mleb_env = MLEBDockerEnv()
                mleb_env.prepare()
                (Path(zipfile_path)).mkdir(parents=True, exist_ok=True)
                mleb_env.run(
                    f"mlebench prepare -c {competition} --data-dir ./zip_files",
                    local_path=local_path,
                    running_extra_volume={str(Path("~/.kaggle").expanduser().absolute()): "/root/.kaggle"},
                )

        if not (Path(local_path) / competition).exists() or list((Path(local_path) / competition).iterdir()) == []:
            if is_no_docker:
                # In no-docker mode, raise an error if data doesn't exist
                raise FileNotFoundError(
                    f"Competition data not found at {Path(local_path) / competition}. "
                    f"In no-docker mode, data must be pre-prepared. "
                    f"Please ensure the data is available at this path."
                )
            (Path(local_path) / competition).mkdir(parents=True, exist_ok=True)

            mleb_env = MLEBDockerEnv()
            mleb_env.prepare()
            mleb_env.run(f"cp -r ./zip_files/{competition}/prepared/public/* ./{competition}", local_path=local_path)'''

    if search in content:
        content = content.replace(search, replace, 1)
        print("  ✓ Applied patch: download_data()")
    else:
        print("  ⚠ Patch already applied or pattern not found")

    # Patch: crawl_descriptions() function - Add grade.py to competition description
    search_crawl = '''    if (fp := Path(f"{local_data_path}/{competition}/description.md")).exists() and not force:
        logger.info(f"Found {competition}/description.md, loading from it.")
        return fp.read_text()'''

    replace_crawl = '''    if (fp := Path(f"{local_data_path}/{competition}/description.md")).exists() and not force:
        logger.info(f"Found {competition}/description.md, loading from it.")
        description = fp.read_text()

        # Append grade.py if it exists in the same directory
        grade_py_path = Path(f"{local_data_path}/{competition}/grade.py")
        logger.info(f"Looking for grade.py at: {grade_py_path} (exists: {grade_py_path.exists()})")
        if grade_py_path.exists():
            logger.info(f"Found {competition}/grade.py, appending to description.")
            description += "\\n\\n## GRADING SCRIPT\\n\\n```python\\n"
            description += grade_py_path.read_text()
            description += "\\n```\\n"
        else:
            logger.warning(f"grade.py not found at {grade_py_path}")

        return description'''

    if search_crawl in content:
        content = content.replace(search_crawl, replace_crawl, 1)
        print("  ✓ Applied patch: crawl_descriptions() - Add grade.py")
    else:
        print("  ⚠ Patch for crawl_descriptions() already applied or pattern not found")

    # Write the patched file
    crawler_file.write_text(content)
    print(f"  ✓ Saved changes to {crawler_file}")
    return True


def apply_test_file_patch(rdagent_dir):
    """Copy the updated mle_submission_format_test.txt file."""
    print("Patching mle_submission_format_test.txt...")

    target_file = rdagent_dir / "scenarios" / "data_science" / "eval_tests" / "mle_submission_format_test.txt"
    source_file = Path(__file__).parent / "mle_submission_format_test.txt"

    if not source_file.exists():
        print(f"  ✗ Source file not found: {source_file}")
        return False

    # Create parent directories if they don't exist
    target_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy the updated content
    content = source_file.read_text()
    target_file.write_text(content)
    print(f"  ✓ Created and updated {target_file}")
    return True


def clear_rdagent_cache(rdagent_dir):
    """Clear all __pycache__ directories in the rdagent installation."""
    import shutil
    import glob

    print("Clearing all Python cache in rdagent installation...")
    try:
        # Find all __pycache__ directories in rdagent
        pycache_dirs = glob.glob(str(Path(rdagent_dir) / "**" / "__pycache__"), recursive=True)
        if pycache_dirs:
            for pycache_dir in pycache_dirs:
                try:
                    shutil.rmtree(pycache_dir)
                    print(f"  ✓ Cleared {pycache_dir}")
                except Exception as e:
                    print(f"  ⚠ Could not clear {pycache_dir}: {e}")
            print(f"Total: Cleared {len(pycache_dirs)} cache directories")
        else:
            print("  ℹ No __pycache__ directories found")
        return True
    except Exception as e:
        print(f"  ⚠ Error clearing cache: {e}")
        return False


def install_custom_backends(rdagent_dir):
    """Install custom Claude and Gemini backends with native schema support."""
    import shutil

    print("\nInstalling custom backends (Claude + Gemini with native schema support)...")

    backend_dir = rdagent_dir / "oai" / "backend"

    if not backend_dir.exists():
        print(f"  ✗ Backend directory not found: {backend_dir}")
        return False

    success = True

    # Copy Claude backend
    claude_backend_source = Path(__file__).parent / "backend_claude.py"
    claude_backend_target = backend_dir / "backend_claude.py"
    if claude_backend_source.exists():
        try:
            shutil.copy(claude_backend_source, claude_backend_target)
            print(f"  ✓ Installed Claude backend to {claude_backend_target}")
        except Exception as e:
            print(f"  ✗ Failed to install Claude backend: {e}")
            success = False
    else:
        print(f"  ✗ Claude backend source not found: {claude_backend_source}")
        success = False

    # Copy Gemini backend
    gemini_backend_source = Path(__file__).parent / "backend_gemini.py"
    gemini_backend_target = backend_dir / "backend_gemini.py"
    if gemini_backend_source.exists():
        try:
            shutil.copy(gemini_backend_source, gemini_backend_target)
            print(f"  ✓ Installed Gemini backend to {gemini_backend_target}")
        except Exception as e:
            print(f"  ✗ Failed to install Gemini backend: {e}")
            success = False
    else:
        print(f"  ✗ Gemini backend source not found: {gemini_backend_source}")
        success = False

    return success


def apply_reformatter_patch(rdagent_dir):
    """Patch response_reformatter.py to skip reformatting for native backends."""
    import shutil

    print("\nPatching response_reformatter.py to skip GPT-5 for native backends...")

    reformatter_file = rdagent_dir / "oai" / "backend" / "response_reformatter.py"

    if not reformatter_file.exists():
        print(f"  ⚠ response_reformatter.py not found at {reformatter_file}")
        return False

    # Read the file
    content = reformatter_file.read_text()

    # Patch 1: Add missing import os
    import_search = '''import json
from typing import Optional, Union, Type, Any

from pydantic import BaseModel'''

    import_replace = '''import os
import json
from typing import Optional, Union, Type, Any

from pydantic import BaseModel'''

    if import_search in content and "import os" not in content[:100]:
        content = content.replace(import_search, import_replace, 1)
        print("  ✓ Added 'import os' to response_reformatter.py")
    else:
        print("  ℹ 'import os' already present or skipped")

    # Patch 2: Check for native backends and skip reformatting
    search = '''def is_claude_or_gemini_model(chat_model: str) -> bool:
    """Check if the model is Claude or Gemini."""
    if not isinstance(chat_model, str):
        return False
    model_lower = chat_model.lower()
    return 'claude' in model_lower or 'gemini' in model_lower'''

    replace = '''def is_claude_or_gemini_model(chat_model: str) -> bool:
    """Check if the model is Claude or Gemini."""
    if not isinstance(chat_model, str):
        return False
    model_lower = chat_model.lower()

    # Check if using native backends - if so, skip reformatting
    # Native backends handle JSON schema natively without GPT-5
    backend = os.getenv("BACKEND", "")
    if "backend_claude" in backend or "backend_gemini" in backend:
        return False  # Skip reformatting for native backends

    return 'claude' in model_lower or 'gemini' in model_lower'''

    # Check if already patched
    if "backend_claude" in content and "backend_gemini" in content and "Skip reformatting for native backends" in content:
        print("  ℹ is_claude_or_gemini_model() already patched to skip reformatting for native backends")
    elif search in content:
        content = content.replace(search, replace, 1)
        print("  ✓ Patched is_claude_or_gemini_model() to skip reformatting for native backends")
    else:
        print("  ⚠ Could not find is_claude_or_gemini_model() function to patch")
        return False

    # Write the patched file
    reformatter_file.write_text(content)
    print(f"  ✓ Saved changes to {reformatter_file}")
    return True


def verify_patches(rdagent_dir):
    """Verify that patches were applied successfully."""
    print("\nVerifying patches...")

    env_file = rdagent_dir / "utils" / "env.py"
    crawler_file = rdagent_dir / "scenarios" / "kaggle" / "kaggle_crawler.py"
    test_file = rdagent_dir / "scenarios" / "data_science" / "eval_tests" / "mle_submission_format_test.txt"
    backend_dir = rdagent_dir / "oai" / "backend"
    reformatter_file = rdagent_dir / "oai" / "backend" / "response_reformatter.py"

    env_content = env_file.read_text()
    crawler_content = crawler_file.read_text()
    test_content = test_file.read_text()
    reformatter_content = reformatter_file.read_text() if reformatter_file.exists() else ""

    # Check for custom backends
    claude_backend_exists = (backend_dir / "backend_claude.py").exists()
    gemini_backend_exists = (backend_dir / "backend_gemini.py").exists()

    checks = [
        ("DockerEnv.prepare() no-docker check", "DS_Coder_CoSTEER_ENV_TYPE" in env_content and "logger.warning" in env_content),
        ("DockerEnv._run() no-docker check", "DockerEnv.run() called but conda/local mode is enabled" in env_content),
        ("download_data() no-docker check", "is_no_docker" in crawler_content),
        ("mle_submission_format_test.txt updated", "from mlebench" not in test_content and "validate_submission.sh" in test_content),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False

    return all_passed


def main():
    print("=" * 70)
    print("RD-Agent No-Docker Patch Installer")
    print("=" * 70)
    print()

    # Find rdagent installation
    rdagent_dir = find_rdagent_dir()
    print(f"Found rdagent at: {rdagent_dir}")
    print()

    # Clear cache first to ensure fresh imports
    clear_rdagent_cache(rdagent_dir)
    print()

    # Apply patches
    success = True
    success &= apply_env_patch(rdagent_dir)
    success &= apply_kaggle_crawler_patch(rdagent_dir)
    success &= apply_test_file_patch(rdagent_dir)

    # NOTE: Custom backends removed - using LiteLLM instead which has native support
    # for Claude and Gemini structured outputs (no need for separate backends)

    if not success:
        print("\n✗ Some patches failed to apply")
        sys.exit(1)

    # Verify patches
    if verify_patches(rdagent_dir):
        print("\n" + "=" * 70)
        print("✓ All patches applied successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Set environment variables:")
        print("   export DS_Coder_CoSTEER_ENV_TYPE=conda")
        print("   export DS_Runner_CoSTEER_ENV_TYPE=conda")
        print()
        print("2. Configure backend in .env to use LiteLLM:")
        print("   BACKEND='rdagent.oai.backend.litellm.LiteLLMAPIBackend'")
        print()
        print("   Option A - Use Claude (LiteLLM with native structured outputs):")
        print("      CHAT_MODEL='anthropic/claude-sonnet-4-5-20250929'")
        print("      ANTHROPIC_API_KEY='<your-key>'")
        print()
        print("   Option B - Use Gemini (LiteLLM with native structured outputs):")
        print("      CHAT_MODEL='gemini/gemini-1.5-pro'")
        print("      GEMINI_API_KEY='<your-key>'")
        print()
        print("3. Prepare competition data:")
        print("   rexmlebench prepare -c <competition_id>")
        print()
        print("4. Run rdagent in no-docker mode")
        print()
        print("BENEFITS OF LITELLM:")
        print("✓ Native support for Claude and Gemini structured outputs")
        print("✓ No custom backend maintenance needed")
        print("✓ Eliminates GPT-5 reformatting completely")
        print("✓ 50% faster - one API call instead of two")
        print("✓ 50% cheaper - no GPT-5 reformatting costs")
        print("✓ More reliable - uses official SDK implementations")
    else:
        print("\n✗ Patch verification failed")
        print("Please check the files manually")
        sys.exit(1)


if __name__ == "__main__":
    main()
