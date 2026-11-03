"""Contamination check: did the SWE-Gym 32B verifier train on the runs we would evaluate it on?

The verifier's training set is public (HF SWE-Gym/OpenHands-Verifier-Trajectories, 2,636 rendered
prompts, no IDs), so overlap is found by matching content:
  task level   the issue text inside <pr_description>
  run level    every assistant turn as 'thought || tool(arg values)', in order, plus the task
               (strict); or the task plus the first 5 and last 5 turns (loose, survives truncation)
Tool arguments and the task are part of the key because GPT-4o turns usually have empty text and
short degenerate runs repeat the same few steps across tasks; without them matching over-counts.

    python -m experiments.verifier_overlap

Downloads the verifier data to data/verifier_traj/ on first run. Writes per-run matches to
data/verifier_overlap.csv and a summary to results/verifier_overlap.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

from src import config
from src.data.loader import load_meta, shard_paths

VT_DIR = config.DATA_DIR / "verifier_traj"
VT_URL = ("https://huggingface.co/datasets/SWE-Gym/OpenHands-Verifier-Trajectories/resolve/main/"
          "data/train.{part}-00000-of-00001.parquet")
OUT_CSV = config.DATA_DIR / "verifier_overlap.csv"
OUT_JSON = config.RESULTS_DIR / "verifier_overlap.json"

PR = re.compile(r"<pr_description>(.*?)</pr_description>", re.S)
TURN = re.compile(r"\*\*\* Turn \d+ - ASSISTANT \*\*\*\n(.*?)(?=\n-{20,}\n|\Z)", re.S)
FN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\r", "")).strip()


def h(s: str) -> str:
    return hashlib.sha1(norm(s).encode()).hexdigest()


def verifier_turn(s: str) -> str:
    """Verifier side: one rendered assistant turn -> 'thought || fn(v1|v2|...)'."""
    th = norm(s.split("<function=")[0])
    calls = [f"{fn.strip()}({'|'.join(norm(v) for _, v in PARAM.findall(body))})" for fn, body in FN.findall(s)]
    return th + " || " + ";".join(calls)


def our_turn(msg: dict) -> str:
    """Our side: the same signature from content + tool_calls."""
    c = msg.get("content") or ""
    if isinstance(c, list):
        c = " ".join(cc.get("text", "") for cc in c)
    th = norm(c.split("<function=")[0])
    calls = []
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        calls.append(f"{tc['function']['name']}({'|'.join(norm(str(v)) for v in args.values())})")
    if not calls and "<function=" in c:
        return verifier_turn(c)
    return th + " || " + ";".join(calls)


def run_key(turns: list[str]) -> str:
    """Strict: every turn, in order, plus the turn count."""
    t = [x for x in turns if x]
    return h(f"{len(turns)}||" + "||".join(t)) if t else ""


def loose_key(turns: list[str]) -> str:
    """Robust to middle truncation: first 5 + last 5 turns."""
    t = [x for x in turns if x]
    return h("||".join(t[:5] + ["..."] + t[-5:])) if len(t) >= 10 else ""


def download() -> None:
    VT_DIR.mkdir(parents=True, exist_ok=True)
    for part in ("offpolicy", "onpolicy", "mixture"):
        dest = VT_DIR / f"{part}.parquet"
        if not dest.exists():
            print(f"  downloading {part}")
            urllib.request.urlretrieve(VT_URL.format(part=part), dest)


def load_verifier() -> tuple[set[str], dict[str, bool], dict[str, bool], int]:
    """-> (task keys, strict run key -> label, loose run key -> label, rows parsed)"""
    tasks, strict, loose = defaultdict(int), {}, {}
    rows = 0
    for part in ("offpolicy", "onpolicy"):
        for r in pq.read_table(VT_DIR / f"{part}.parquet").to_pylist():
            u = r["messages"][1]["content"]
            m = PR.search(u)
            if not m:
                continue
            rows += 1
            tk = h(m.group(1))
            tasks[tk] += 1
            turns = [verifier_turn(x) for x in TURN.findall(u)]
            strict[tk + run_key(turns)] = r["resolved"]
            loose[tk + loose_key(turns)] = r["resolved"]
    return set(tasks), strict, loose, rows


def match_runs(tasks: set[str], strict: dict[str, bool], loose: dict[str, bool]) -> pd.DataFrame:
    """Stream our runs and flag task- and run-level overlap with the verifier's training set."""
    meta = load_meta().set_index(["shard", "row"])
    out = []
    for si, path in enumerate(shard_paths()):
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["messages"]):
            for msgs in batch.column("messages").to_pylist():
                r = meta.loc[(si, row)]
                row += 1
                user = next((m["content"] for m in msgs if m["role"] == "user"), "") or ""
                if isinstance(user, list):
                    user = " ".join(c.get("text", "") for c in user)
                m = PR.search(user)
                turns = [our_turn(x) for x in msgs if x["role"] == "assistant"]
                tk = h(m.group(1)) if m else ""
                k = tk + run_key(turns)
                lk = tk + loose_key(turns) if loose_key(turns) else ""
                out.append({"uid": r["uid"], "pr_found": bool(m), "task_hit": bool(m) and tk in tasks,
                            "run_hit": k in strict, "loose_hit": bool(lk) and lk in loose,
                            "label_match": k in strict and strict[k] == bool(r["resolved"]),
                            "k": k, "n_steps": len(turns), "resolved": bool(r["resolved"])})
    splits = pd.read_csv(config.SPLITS_PATH).set_index("uid")
    return pd.DataFrame(out).join(splits[["split", "instance_id", "judge_sample"]], on="uid")


def _summary(g: pd.DataFrame) -> dict:
    tasks = g.groupby("instance_id")["task_hit"].any()
    ident = g.run_hit | g.loose_hit
    return {"runs": len(g), "tasks": len(tasks), "tasks_seen": int(tasks.sum()),
            "task_overlap_tasks": round(float(tasks.mean()), 4), "task_overlap_runs": round(float(g.task_hit.mean()), 4),
            "identical_runs_strict": int(g.run_hit.sum()), "identical_runs_loose": int(ident.sum()),
            "identical_rate": round(float(ident.mean()), 4)}


def main() -> None:
    download()
    tasks, strict, loose, rows = load_verifier()
    print(f"verifier train rows parsed: {rows}; distinct tasks {len(tasks)}, distinct run keys {len(strict)}")
    df = match_runs(tasks, strict, loose)
    df.to_csv(OUT_CSV, index=False)

    mixed = df.groupby("instance_id")["resolved"].transform(lambda s: 0 < s.sum() < len(s))
    unseen = df[~df.task_hit]
    report = {
        "verifier_rows": rows, "verifier_tasks": len(tasks),
        "all": _summary(df), "dev": _summary(df[df.split == "dev"]), "test": _summary(df[df.split == "test"]),
        "test_mixed_tasks": _summary(df[(df.split == "test") & mixed]),
        "dev_mixed_tasks": _summary(df[(df.split == "dev") & mixed]),
        "judge_sample": _summary(df[df.judge_sample]),
        "strict_label_agreement": f"{int(df.label_match.sum())}/{int(df.run_hit.sum())}",
        "distinct_verifier_rows_hit": int(df.loc[df.run_hit, "k"].nunique()),
        "unseen_task_runs": len(unseen), "unseen_task_resolved_rate": round(float(unseen.resolved.mean()), 4),
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"-> {OUT_JSON}, {OUT_CSV}")


if __name__ == "__main__":
    main()
