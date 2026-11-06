# CLAUDE.md

Course project (AI4015, G2-09): predict SWE agent run success from OpenHands execution traces
(SWE-Gym trajectories). See `docs/architecture.md` for the design.

## Environment
- Python 3.11 venv at `.venv` (created with `uv venv --python 3.11`). Windows host.
- Install: `uv pip install --python .venv -r requirements.txt`
- Run all: `.venv/Scripts/python -m src.run_baseline` (add `--with-llm` for the LLM judge; provider set in `.env`)
- Tests: `.venv/Scripts/python -m pytest -q`

## Rules
- Never use `test_result.report`, `test_result.test_output`, or `apply_patch_output` as features: they leak the label.
- Don't load the whole `messages` column. Stream row groups via `src/data/parser.iter_traces`.
- All evaluation uses `data/splits.csv`: a fixed held-out test set (never used for tuning) and
  GroupKFold by `instance_id` on dev. Don't rebuild it casually; every reported number depends on it.
- Module boundaries pass the Pydantic models in `src/schema.py`.
- `data/` and `.env` are gitignored, except `data/splits.csv`, `data/verifier_scores_lite.csv` and `data/llm_judge_cache.jsonl`, which can't be rebuilt cheaply. `results/` is committed. `logs/` is ignored except `logs/sample/`.
- The governance agent was locked at commit 4a3b1a6 and E3/E5 ran once on test. Don't retune it; anything changed afterwards must be reported as a change.
- Hardware: RTX 4050 6 GB, 16 GB RAM, no paid APIs.
