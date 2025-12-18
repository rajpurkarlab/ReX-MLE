import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from openai import AzureOpenAI


BASE_SYSTEM_PROMPT = (
    "You are an expert adjudicator evaluating the execution logs of an autonomous AI agent "
    "on a medical imaging task. Your objective is to determine if the agent implemented specific "
    "technical strategies based on the evidence in the logs.\n\n"
    "INSTRUCTIONS:\n"
    "1. Review the provided Log Content deeply.\n"
    "2. For EACH Strategy Definition, decide if there is explicit evidence of execution.\n"
    "3. Score 1 ONLY if there is explicit evidence of execution (code execution, specific library "
    "calls, distinct file outputs).\n"
    "4. Score 0 if the strategy is ambiguous, merely planned but not executed, or absent.\n"
    "5. Return a raw JSON object (no Markdown) with a `results` list of objects containing "
    "`id`, `strategy`, `score`, and `evidence`.\n\n"
    "Output Format:\n"
    '{"results":[{"id":1,"strategy":"...","score":0,"evidence":"..."}]}'
)


@dataclass
class StrategyPrompt:
    sid: int
    name: str
    text: str
    source: Path


def load_prompts(prompt_file: Path) -> Tuple[str, List[StrategyPrompt]]:
    prompt_file = prompt_file.expanduser().resolve()
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Prompt JSON must be a non-empty object.")

    base_prompt = str(data.get("base_system_prompt") or BASE_SYSTEM_PROMPT).strip()
    prompts: List[StrategyPrompt] = []

    if "strategies" in data:
        strategies = data["strategies"]
        if not isinstance(strategies, dict) or not strategies:
            raise ValueError("The 'strategies' field must be a non-empty object.")
        for _, body in strategies.items():
            if not isinstance(body, dict):
                raise ValueError("Each strategy entry must be an object with id, name, and criteria.")
            missing = [key for key in ("id", "name", "criteria") if key not in body]
            if missing:
                raise ValueError(f"Strategy entry missing keys: {', '.join(missing)}")
            sid = int(body["id"])
            name = str(body["name"]).strip()
            criteria = str(body["criteria"]).strip()
            text = f"TARGET STRATEGY: {name}\n{criteria}"
            prompts.append(StrategyPrompt(sid=sid, name=name, text=text, source=prompt_file))
    else:
        for name, body in data.items():
            if not isinstance(body, dict):
                raise ValueError(f"Prompt entry for '{name}' must be an object with id and prompt.")
            if "id" not in body or "prompt" not in body:
                raise ValueError(f"Prompt entry for '{name}' must include 'id' and 'prompt' keys.")
            sid = int(body["id"])
            text = str(body["prompt"]).strip()
            prompts.append(StrategyPrompt(sid=sid, name=name, text=text, source=prompt_file))

    prompts.sort(key=lambda p: p.sid)
    return base_prompt, prompts


def build_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


def iter_structured_fragments(payload: Dict) -> Iterable[Tuple[str, str]]:
    analyses = payload.get("analyses") or payload.get("analysis") or []
    responses = payload.get("responses") or []
    challenge_id = payload.get("challenge_id")
    prefix = f"{challenge_id}-" if challenge_id else ""

    for idx, text in enumerate(analyses, 1):
        label = f"{prefix}analysis-{idx}"
        body = str(text).strip()
        if body:
            yield label, body

    for idx, text in enumerate(responses, 1):
        label = f"{prefix}response-{idx}"
        body = str(text).strip()
        if body:
            yield label, body


def iter_log_fragments(log_path: Path, max_chars: Optional[int]) -> Iterable[Tuple[str, str]]:
    if log_path.suffix.lower() == ".json":
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        yield from iter_structured_fragments(payload)
        return

    text = log_path.read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    if text.strip():
        yield f"{log_path.name}-full", text.strip()


def log_progress(progress_log_path: Optional[Path], message: str) -> None:
    """Write progress message to log file and flush."""
    if progress_log_path:
        try:
            with open(progress_log_path, 'a') as f:
                f.write(f"[{datetime.datetime.now().isoformat()}] {message}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"Warning: Failed to write progress log: {e}", file=sys.stderr)


def score_section_all(
    client: AzureOpenAI,
    model: str,
    section_label: str,
    section_text: str,
    prompts: List[StrategyPrompt],
    max_completion_tokens: int,
    base_system_prompt: str,
    progress_log_path: Optional[Path] = None,
) -> List[Dict]:
    strategies_blob = "\n\n".join([p.text for p in prompts])
    user_content = (
        f"{strategies_blob}\n\n"
        f"Log section ({section_label}):\n{section_text}\n\n"
        "Return JSON with key 'results' containing one object per strategy."
    )

    log_progress(progress_log_path, f"Processing section: {section_label}")

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            max_completion_tokens=max_completion_tokens,
            messages=[
                {"role": "system", "content": base_system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        log_progress(progress_log_path, f"Received response for section: {section_label}")

        # Safely extract content from response
        if not hasattr(response, 'choices') or not response.choices or len(response.choices) == 0:
            log_progress(progress_log_path, f"ERROR: API response has no choices for {section_label}")
            return []

        choice = response.choices[0]
        if not hasattr(choice, 'message') or not choice.message:
            log_progress(progress_log_path, f"ERROR: API response choice has no message for {section_label}")
            return []

        content = choice.message.content
        if content is None:
            log_progress(progress_log_path, f"ERROR: API response message has no content for {section_label}")
            return []

        log_progress(progress_log_path, f"Parsed response for section: {section_label} ({len(content)} chars)")

        parsed = json.loads(content)

        results = parsed.get("results") if isinstance(parsed, dict) else None
        output: List[Dict] = []
        if isinstance(results, list):
            for item in results:
                sid = item.get("id")
                try:
                    sid_int = int(sid)
                except (TypeError, ValueError):
                    continue
                strategy_name = str(item.get("strategy") or "").strip()
                score_raw = item.get("score", 0)
                try:
                    score_val = 1 if int(score_raw) == 1 else 0
                except (TypeError, ValueError):
                    score_val = 0
                evidence = str(item.get("evidence", "")).strip()
                output.append(
                    {
                        "id": sid_int,
                        "strategy": strategy_name,
                        "score": score_val,
                        "evidence": evidence,
                        "section": section_label,
                        "prompt_file": str(prompts[0].source) if prompts else "",
                    }
                )

        # Fill in any missing strategies with defaults.
        seen_ids = {r["id"] for r in output}
        for p in prompts:
            if p.sid not in seen_ids:
                output.append(
                    {
                        "id": p.sid,
                        "strategy": p.name,
                        "score": 0,
                        "evidence": "Missing in model response",
                        "section": section_label,
                        "prompt_file": str(p.source),
                    }
                )

        output.sort(key=lambda r: r["id"])
        return output

    except Exception as e:
        error_msg = f"Exception while scoring section {section_label}: {type(e).__name__}: {str(e)}"
        log_progress(progress_log_path, f"ERROR: {error_msg}")
        print(f"Warning: {error_msg}", file=sys.stderr)
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all strategies per analysis section using a single model call."
    )
    parser.add_argument("--log-file", type=Path, help="Path to a single log file to score.")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Directory containing multiple log files to score (e.g., mlmaster_logs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write per-log JSON outputs when using --batch-dir. Defaults to the batch directory.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("prompts/strategies.json"),
        help="JSON file containing strategy prompts.",
    )
    parser.add_argument(
        "--deployment",
        default=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"),
        help="Azure OpenAI deployment name (default: gpt-5).",
    )
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        help="Azure OpenAI API version.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-ai.hms.edu"),
        help="Azure OpenAI endpoint (resource base, no trailing /openai), e.g., https://<resource>.openai.azure.com",
    )
    parser.add_argument(
        "--api-key",
        help="Azure OpenAI API key.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        dest="max_completion_tokens",
        help="Max completion tokens for each response.",
    )
    parser.add_argument(
        "--truncate-log-chars",
        type=int,
        default=15000,
        help="Optional character cap on the log passed to the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save JSON results (single log only). Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="Optional path to log progress and responses for monitoring.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.log_file and not args.batch_dir:
        raise SystemExit("Provide either --log-file or --batch-dir.")
    if args.log_file and args.batch_dir:
        raise SystemExit("Use only one of --log-file or --batch-dir, not both.")
    if args.batch_dir and args.output:
        raise SystemExit("--output is for single-log mode. Use --output-dir for batch mode.")

    missing = []
    if not args.deployment:
        missing.append("deployment (--deployment or AZURE_OPENAI_DEPLOYMENT)")
    if not args.endpoint:
        missing.append("endpoint (--endpoint or AZURE_OPENAI_ENDPOINT)")
    if not args.api_key:
        missing.append("api key (--api-key or AZURE_OPENAI_API_KEY)")
    if missing:
        raise SystemExit(f"Missing required Azure config: {', '.join(missing)}")


def score_log(
    log_path: Path,
    prompts: List[StrategyPrompt],
    client: AzureOpenAI,
    model: str,
    max_completion_tokens: int,
    truncate_log_chars: Optional[int],
    api_version: str,
    base_system_prompt: str,
    existing_results: Optional[List[Dict]] = None,
    on_progress: Optional[Callable[[List[Dict]], None]] = None,
    progress_log_path: Optional[Path] = None,
    num_threads: int = 16,
) -> Dict:
    fragments = list(iter_log_fragments(log_path, max_chars=truncate_log_chars))
    results: List[Dict] = list(existing_results or [])
    completed_sections = {r.get("section") for r in results if r.get("section")}
    results_lock = threading.Lock()

    # Filter out already completed sections
    pending_fragments = [
        (label, text) for label, text in fragments
        if label not in completed_sections
    ]

    if not pending_fragments:
        return {
            "log_file": str(log_path),
            "model": model,
            "api_version": api_version,
            "results": sorted(results, key=lambda r: (r["section"], r["id"])),
        }

    def process_fragment(section_label: str, section_text: str) -> Tuple[str, List[Dict]]:
        """Process a single fragment and return (section_label, results)."""
        try:
            section_results = score_section_all(
                client=client,
                model=model,
                section_label=section_label,
                section_text=section_text,
                prompts=prompts,
                max_completion_tokens=max_completion_tokens,
                base_system_prompt=base_system_prompt,
                progress_log_path=progress_log_path,
            )
            return section_label, section_results
        except Exception as e:
            error_msg = f"Exception in thread for section {section_label}: {type(e).__name__}: {str(e)}"
            log_progress(progress_log_path, f"THREAD ERROR: {error_msg}")
            print(f"Warning: {error_msg}", file=sys.stderr)
            return section_label, []

    # Use ThreadPoolExecutor to process multiple fragments in parallel
    log_progress(progress_log_path, f"Starting {num_threads} threads to process {len(pending_fragments)} sections")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_fragment, label, text): label
            for label, text in pending_fragments
        }

        # Process completed futures as they finish
        for future in as_completed(futures):
            section_label, section_results = future.result()
            with results_lock:
                results.extend(section_results)
                completed_sections.add(section_label)
                if on_progress:
                    on_progress(results)
            log_progress(progress_log_path, f"Completed section: {section_label}")

    log_progress(progress_log_path, f"All {len(pending_fragments)} sections processed")

    return {
        "log_file": str(log_path),
        "model": model,
        "api_version": api_version,
        "results": sorted(results, key=lambda r: (r["section"], r["id"])),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    base_system_prompt, prompts = load_prompts(args.prompt_file)
    client = build_client(api_key=args.api_key, endpoint=args.endpoint, api_version=args.api_version)

    if args.batch_dir:
        batch_dir = args.batch_dir.expanduser().resolve()
        if not batch_dir.exists():
            raise SystemExit(f"Batch directory not found: {batch_dir}")
        out_dir = (args.output_dir or batch_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        log_files = sorted(
            [
                p
                for p in batch_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".json"
                and not p.stem.endswith("_scores")
            ]
        )
        if not log_files:
            raise SystemExit(f"No log files (*.json) found in {batch_dir}")

        print(f"Spawning separate analysis processes for {len(log_files)} log files...")
        processes = []
        for log_file in log_files:
            # Build command to run this script for a single log file
            cmd = [
                sys.executable,
                __file__,
                "--log-file", str(log_file),
                "--prompt-file", str(args.prompt_file),
                "--deployment", args.deployment,
                "--api-key", args.api_key,
                "--endpoint", args.endpoint,
                "--api-version", args.api_version,
                "--output", str(out_dir / f"{log_file.stem}_scores_all.json"),
            ]
            if args.max_completion_tokens:
                cmd.extend(["--max-completion-tokens", str(args.max_completion_tokens)])
            if args.truncate_log_chars:
                cmd.extend(["--truncate-log-chars", str(args.truncate_log_chars)])

            print(f"  Starting analysis for {log_file.name}...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append((log_file.name, proc))

        # Wait for all processes to complete
        print(f"Waiting for {len(processes)} processes to complete...")
        failed_files = []
        for file_name, proc in processes:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                failed_files.append(file_name)
                print(f"✗ {file_name} failed with return code {proc.returncode}")
                if stderr:
                    print(f"  Error output:\n{stderr}")
            else:
                print(f"✓ {file_name} completed successfully")

        if failed_files:
            print(f"\nWarning: {len(failed_files)} file(s) failed: {', '.join(failed_files)}")
        else:
            print(f"\nAll {len(log_files)} analyses completed successfully!")
    else:
        existing_results: List[Dict] = []
        if args.output and args.output.exists():
            try:
                prior_payload = json.loads(args.output.read_text(encoding="utf-8"))
                if isinstance(prior_payload, dict):
                    existing_results = prior_payload.get("results") or []
            except json.JSONDecodeError:
                existing_results = []

        payload = score_log(
            log_path=args.log_file,
            prompts=prompts,
            client=client,
            model=args.deployment,
            max_completion_tokens=args.max_completion_tokens,
            truncate_log_chars=args.truncate_log_chars,
            api_version=args.api_version,
            base_system_prompt=base_system_prompt,
            existing_results=existing_results,
            progress_log_path=args.progress_log,
        )
        output_text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(output_text, encoding="utf-8")
        print(output_text)


if __name__ == "__main__":
    main()
