"""Render runs into the SWE-Gym 32B verifier's input format, for scoring on Kaggle.

The format is copied from the verifier's public training data (HF SWE-Gym/OpenHands-Verifier-
Trajectories): a fixed judge system message, then one user message holding a header block (the
agent's system prompt and tool descriptions), the run as "*** Turn N - USER/ASSISTANT ***" blocks
with tool calls written as <function=...><parameter=...>, and a fixed closing question. The model
answers "<judgement>YES</judgement>" or "<judgement>NO</judgement>"; p(YES) is the score.

The header, judge message and closing question are taken verbatim from the training data. The
renderer is checked against it: `--check` renders SWE-Gym runs that are exact copies of training
examples (see experiments/verifier_overlap.py) and compares them with the training text.

Output: data/verifier_prompts_lite.jsonl, one line per run in results/verifier_sample_lite.json:
  {uid, run, version, judge_system, head, turns: [block, ...], tail}
Turns stay separate so the notebook can drop middle turns to fit the 32k-token context.

    python -m experiments.verifier_prompts [--check]
"""
from __future__ import annotations

import argparse
import collections
import json

import pyarrow.parquet as pq

from src import config
from src.data import swebench_lite

VT_DIR = config.DATA_DIR / "verifier_traj"
OUT_PATH = config.DATA_DIR / "verifier_prompts_lite.jsonl"
SAMPLE_PATH = config.RESULTS_DIR / "verifier_sample_lite.json"
SEP = "-" * 100 + "\n"
TURN_MARK = "*** Turn 1 - USER ***"


def training_frame() -> tuple[str, str, str]:
    """-> (judge system message, header block, closing tail), the most common variant in training."""
    rows = pq.read_table(VT_DIR / "onpolicy.parquet").to_pylist() + pq.read_table(VT_DIR / "offpolicy.parquet").to_pylist()
    judge = collections.Counter(r["messages"][0]["content"] for r in rows).most_common(1)[0][0]
    users = [r["messages"][1]["content"] for r in rows]
    head = collections.Counter(u[:u.index(TURN_MARK)] for u in users).most_common(1)[0][0]
    tail = collections.Counter(u[u.rindex(SEP) + len(SEP):] for u in users).most_common(1)[0][0]
    return judge, head, tail


def _value(v) -> str:
    """Multi-line values sit on their own lines inside the tag, as in the training data."""
    v = v if isinstance(v, str) else json.dumps(v)
    return f"\n{v}\n" if "\n" in v else v


def _assistant(m: dict) -> str:
    text = (m.get("content") or "").rstrip()
    calls = []
    for tc in m.get("tool_calls") or []:
        raw = tc["function"]["arguments"]
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = {}
        params = "".join(f"<parameter={k}>{_value(v)}</parameter>\n" for k, v in args.items())
        # a call with no argument string at all is written without a closing tag
        calls.append(f"<function={tc['function']['name']}>\n{params}" + ("</function>" if raw else ""))
    body = "\n".join(calls)
    return f"{text}\n\n{body}" if text and body else (text or body)


def render_turns(msgs: list[dict]) -> list[str]:
    """Chat messages (system first) -> turn blocks. Tool results are USER turns, as in training."""
    blocks, turn, last = [], 0, None
    for m in msgs[1:]:
        if m["role"] == "assistant":
            role, content = "ASSISTANT", _assistant(m)
        elif m["role"] == "tool":
            role, content = "USER", f"EXECUTION RESULT of [{m.get('name') or 'error'}]:\n{m.get('content') or ''}"
        else:
            role, content = "USER", m.get("content") or ""
        if role == "USER" and last != "USER":
            turn += 1
        last = role
        blocks.append(f"*** Turn {turn} - {role} ***\n{content}\n{SEP}")
    return blocks


def check(n: int = 50) -> None:
    """Render SWE-Gym runs that are exact copies of training examples and diff against training."""
    import pandas as pd

    from experiments.verifier_overlap import PR, TURN, h, run_key, verifier_turn
    from src.data.loader import load_meta, shard_paths

    train = {}
    for part in ("offpolicy", "onpolicy"):
        for r in pq.read_table(VT_DIR / f"{part}.parquet").to_pylist():
            u = r["messages"][1]["content"]
            m = PR.search(u)
            if m:
                train[h(m.group(1)) + run_key([verifier_turn(x) for x in TURN.findall(u)])] = u
    ov = pd.read_csv(config.DATA_DIR / "verifier_overlap.csv")
    want = dict(zip(ov.loc[ov["run_hit"], "uid"].head(n), ov.loc[ov["run_hit"], "k"].head(n)))
    meta = load_meta().set_index(["shard", "row"])
    _, head, tail = training_frame()
    exact = 0
    for si, p in enumerate(shard_paths()):
        row = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64, columns=["messages"]):
            for msgs in batch.column("messages").to_pylist():
                uid = meta.loc[(si, row), "uid"]
                row += 1
                if uid in want:
                    ours = "".join(render_turns(msgs))
                    ref = train[want[uid]]
                    ref_turns = ref[ref.index(TURN_MARK):ref.rindex(SEP) + len(SEP)]
                    if ours == ref_turns:
                        exact += 1
                    else:
                        i = next((k for k, (a, b) in enumerate(zip(ours, ref_turns)) if a != b), min(len(ours), len(ref_turns)))
                        print(f"{uid}: differs at char {i} of {len(ref_turns)}\n  ours: {ours[max(0, i - 80):i + 80]!r}\n"
                              f"  ref:  {ref_turns[max(0, i - 80):i + 80]!r}")
    print(f"exact matches: {exact}/{len(want)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate the renderer against training examples")
    if ap.parse_args().check:
        check()
        return
    judge, head, tail = training_frame()
    uids = set(json.loads(SAMPLE_PATH.read_text())["uids"])
    order = {u: i for i, u in enumerate(json.loads(SAMPLE_PATH.read_text())["uids"])}
    out = []
    for run, trace, msgs in swebench_lite.iter_lite():
        if trace.uid in uids:
            out.append({"uid": trace.uid, "run": run, "version": swebench_lite.RUNS[run][2], "judge_system": judge,
                        "head": head, "turns": render_turns(msgs), "tail": tail})
    out.sort(key=lambda r: order[r["uid"]])
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} prompts -> {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
