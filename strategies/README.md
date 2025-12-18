# Strategy Scoring

This folder scores logs for the 13 challenge strategies via Azure OpenAI. Each agent strategy (aide, mlmaster, rdagent) has its own preprocessing pipeline before scoring.

All prompts live in `prompts/strategies.json` and the scorer returns a 0/1 flag per strategy.

## Installation

Install dependencies (needs the `openai` SDK, version >= 1.42):
```bash
pip install openai
```

Export Azure settings (or pass as CLI flags):
```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-5"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
```

---

## AIDE

See [aide/README.md](aide/README.md) for full details.

1) Create `aide/aide_logs.jsonl` with paths to AIDE descriptive JSONs
2) Run preprocessing:
   ```bash
   python aide/run_logs_postprocessing_aide.py
   ```
3) Score logs:
   ```bash
   python analyze_strategies.py --batch-dir aide/aide_logs_processed
   ```

---

## MLMaster

See [mlmaster/README.md](mlmaster/README.md) for full details.

1) Create `mlmaster/mlmaster_logs.jsonl` with paths to MLMaster runs
2) Run preprocessing:
   ```bash
   python mlmaster/run_logs_postprocessing.py
   ```
3) Score logs:
   ```bash
   python analyze_strategies.py --batch-dir mlmaster/mlmaster_logs
   ```

---

## RDAgent

See [rdagent/README.md](rdagent/README.md) for full details.

1) Create `rdagent/rdagent_runs.jsonl` with paths to RDAgent runs
2) Process all runs:
   ```bash
   python rdagent/process_all_runs.py
   ```
3) Run preprocessing:
   ```bash
   python rdagent/run_logs_postprocessing_rdagent.py
   ```
4) Score logs:
   ```bash
   python analyze_strategies.py --batch-dir rdagent/rdagent_logs_processed
   ```

---

## Common Scoring Options

When running `analyze_strategies.py`:
- `--batch-dir`: directory of logs to score (batch mode).
- `--output-dir`: directory for per-log JSON outputs (batch mode; defaults to the batch dir).
- `--prompt-file`: defaults to `prompts/strategies.json`.
- `--deployment`, `--endpoint`, `--api-key`, `--api-version`: Azure config (env vars also supported).
- `--truncate-log-chars`: cap text sent to the model (default 15000).
- `--max-completion-tokens`: response token budget (default 800).

Aggregate mean scores across `_scores*.json` outputs:
```bash
python aggregate_strategy_scores.py \
  --scores-dir <output-dir> \
  --output <output-dir>/scores.json
```

## Output

Writes JSON of the form:
```json
{
  "log_file": "/path/to/log.txt",
  "model": "gpt-5",
  "api_version": "2024-08-01-preview",
  "results": [
    {"id": 1, "strategy": "Analyzing and handling failure cases", "score": 1, "evidence": "...", "prompt_file": "prompts/strategies.json"},
    {"id": 2, "strategy": "Knowing the state of the art in the field", "score": 0, "evidence": "...", "prompt_file": "prompts/strategies.json"}
  ]
}
```
