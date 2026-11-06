"""How many runs saw an empty repository listing? A threat-to-validity check on the dataset.

Some GPT-4o runs repeatedly view the repository root and get back only the root path, with no files
under it, then stop after a few actions. That can come from the environment rather than from the agent's
decisions. A run counts as "empty listing" if the observation of its first root-directory view lists no
paths beneath the root. Dev only, to keep the held-out test set unread.

    python -m experiments.empty_workspace_check
-> results/empty_workspace_check.json
"""
from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from src import config
from src.data.loader import load_meta, shard_paths


def first_view_listing(messages: list[dict]) -> str:
    """-> "empty", "listed", or "none" (no directory view of the workspace among the first actions)."""
    by_id = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m.get("role") == "tool"}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if fn.get("name") != "str_replace_editor" or args.get("command") != "view":
                continue
            content = by_id.get(tc.get("id"), "")
            if "files and directories" not in content:
                continue
            root = (args.get("path") or "").rstrip("/")
            below = [ln for ln in content.splitlines() if ln.strip().startswith(root + "/")]
            return "listed" if below else "empty"
    return "none"


def main() -> None:
    meta = load_meta().merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split"]], on="uid")
    meta = meta.set_index(["shard", "row"])
    rows = []
    for shard, path in enumerate(shard_paths()):
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["messages"]):
            for msgs in batch.column("messages").to_pylist():
                r = meta.loc[(shard, row)]
                row += 1
                if r["split"] != "dev":
                    continue
                n_actions = sum(len(m.get("tool_calls") or []) for m in msgs)
                rows.append({"listing": first_view_listing(msgs), "resolved": bool(r["resolved"]), "n": n_actions,
                             "model": r["policy_model"], "run_id": r["run_id"],
                             "empty_patch": not (r["git_patch"] if isinstance(r["git_patch"], str) else "").strip()})
    df = pd.DataFrame(rows)
    def table(by: list[str]) -> pd.DataFrame:
        return df.groupby(by).agg(runs=("n", "size"), pass_rate=("resolved", "mean"), median_actions=("n", "median"),
                                  empty_patch=("empty_patch", "mean")).round(3).reset_index()
    out = {"n_dev": len(df), "by_listing": table(["listing"]).to_dict("records"),
           "by_model": table(["model", "listing"]).to_dict("records"),
           "by_run_config": table(["run_id", "listing"]).to_dict("records")}
    (config.RESULTS_DIR / "empty_workspace_check.json").write_text(json.dumps(out, indent=1))
    print(table(["listing"]).to_string(index=False))
    print(table(["model", "listing"]).to_string(index=False))
    print(table(["run_id", "listing"]).to_string(index=False))


if __name__ == "__main__":
    main()
