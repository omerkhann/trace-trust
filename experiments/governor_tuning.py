"""Set the governance agent's operating points. Dev only: test and SWE-bench Lite are never read.

Uses out-of-fold dev scores, because models fit on all of dev would look too good on dev runs.
  A. hint patterns   for each pattern: share of runs it fires on, pass rate when it fires vs when it doesn't,
                     share of failing runs covered, and how early it fires. A pattern worth a hint should
                     fire on runs that pass less often than the rest.
  B. abort budget    for each per-checkpoint false-alarm budget: run-level failures stopped, passing runs
                     lost, tokens saved, using only the checkpoints allowed to abort.
  C. review budget   for each end-of-run false-alarm budget: failures flagged, passing runs flagged.
  D. loop escape     runs where repeat_loop fired: pass rate of those that broke out on their own vs not.
                     Observational: stronger attempts may simply be the ones that escape, so this is not
                     evidence that a hint would cause an escape.
  E. empty-patch     does REJECT add anything over the one-line rule "reject if the patch is empty"?
  F. reroute         tasks solved and cost per task (src/agent/reroute.py), with out-of-fold decisions.

    python -m experiments.governor_tuning
-> results/governor_tuning.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.agent.patterns import HINTS, PatternDetector
from src.agent.reroute import over_seeds
from src.agent.risk import RiskConfig, RiskModels
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.prefix import PrefixState, build_swegym
from src.schema import ActionType, Decision

ABORT_FARS = (0.005, 0.01, 0.025, 0.05, 0.10)
REVIEW_FARS = (0.05, 0.10, 0.20)
LOOP_GRID = ((8, 3), (8, 4), (8, 5), (12, 4), (12, 5), (12, 6), (16, 6))
ESCAPE_AFTER = 5
OUT = config.RESULTS_DIR / "governor_tuning"


def pattern_table(dev_uids: set[str]) -> pd.DataFrame:
    fired = []
    for tr in iter_traces(load_meta(), uids=dev_uids):
        st, det, first = PrefixState(tr.uid, tr.instance_id), PatternDetector(), {}
        for s in tr.steps:
            if not st.is_action(s):
                st.update(s)
                continue
            pre = det.before_finish(st) if s.action_type == ActionType.finish else []
            st.update(s)
            det.update(s, st)
            for p in pre + det.active(st):
                first.setdefault(p, st.step)
        fired.append({"uid": tr.uid, "resolved": bool(tr.resolved), **{f"first_{p}": first.get(p) for p in HINTS}})
    df = pd.DataFrame(fired)
    y = df["resolved"].to_numpy(bool)
    rows = []
    for p in HINTS:
        col = pd.to_numeric(df[f"first_{p}"], errors="coerce")
        m = col.notna().to_numpy()
        rows.append({"pattern": p, "runs_fired": float(m.mean()),
                     "pass_rate_fired": float(y[m].mean()) if m.any() else None,
                     "pass_rate_not_fired": float(y[~m].mean()) if (~m).any() else None,
                     "failing_runs_covered": float(m[~y].mean()),
                     "passing_runs_hit": float(m[y].mean()),
                     "median_first_step": float(col.median()) if m.any() else None})
    out = pd.DataFrame(rows)
    out["hint_worthy"] = out["pass_rate_fired"] <= 0.5 * out["pass_rate_not_fired"]
    return out


def loop_table(dev_uids: set[str]) -> pd.DataFrame:
    """repeat_loop under different definitions: (window, repeats)."""
    fired = []
    for tr in iter_traces(load_meta(), uids=dev_uids):
        row = {"resolved": bool(tr.resolved)}
        for w, k in LOOP_GRID:
            st, det, hit = PrefixState("u", "i"), PatternDetector(w, k), False
            for s in tr.steps:
                st.update(s)
                if st.is_action(s):
                    det.update(s, st)
                    hit = hit or "repeat_loop" in det.active(st)
            row[(w, k)] = hit
        fired.append(row)
    df = pd.DataFrame(fired)
    y = df["resolved"].to_numpy(bool)
    rows = []
    for g in LOOP_GRID:
        m = df[g].to_numpy(bool)
        rows.append({"window": g[0], "repeats": g[1], "runs_fired": float(m.mean()),
                     "pass_rate_fired": float(y[m].mean()), "pass_rate_not_fired": float(y[~m].mean()),
                     "failing_runs_covered": float(m[~y].mean()), "passing_runs_hit": float(m[y].mean())})
    return pd.DataFrame(rows)


def abort_table(models: RiskModels, totals: pd.DataFrame) -> pd.DataFrame:
    prefix = build_swegym()
    prefix = prefix[prefix["checkpoint"] == "abs"][["uid", "step", "tokens_so_far"]]
    allowed = [k for k in models.checkpoints if models.can_abort(k)]
    n_pass, n_fail = int(totals["resolved"].sum()), int((~totals["resolved"]).sum())
    tokens = float(totals["total_tokens"].sum())
    rows = []
    for far in ABORT_FARS:
        aborted: dict[str, float] = {}             # uid -> tokens already used at the first abort
        for k in allowed:
            oof = models.checkpoints[k].oof
            tau = float(np.quantile(oof.loc[oof["resolved"], "p_cal"], far))
            hit = oof[oof["p_cal"] < tau].merge(prefix[prefix["step"] == k], on="uid")
            for uid, used in zip(hit["uid"], hit["tokens_so_far"]):
                aborted.setdefault(uid, float(used))
        a = totals[totals["uid"].isin(aborted)]
        saved = float((a["total_tokens"] - a["uid"].map(aborted)).sum())
        rows.append({"far_per_checkpoint": far, "checkpoints": allowed, "runs_aborted": len(a),
                     "failures_stopped": float((~a["resolved"]).sum() / n_fail),
                     "passing_runs_lost": float(a["resolved"].sum() / n_pass),
                     "passes_lost_n": int(a["resolved"].sum()),
                     "tokens_saved": saved / tokens})
    return pd.DataFrame(rows)


def review_table(models: RiskModels) -> pd.DataFrame:
    oof = models.final.oof
    y = oof["resolved"].to_numpy(bool)
    rows = []
    for far in REVIEW_FARS:
        tau = float(np.quantile(oof.loc[y, "p_cal"], far))
        flag = oof["p_cal"].to_numpy() < tau
        rows.append({"far": far, "tau": tau, "runs_flagged": float(flag.mean()),
                     "failures_flagged": float(flag[~y].mean()), "passing_flagged": float(flag[y].mean())})
    return pd.DataFrame(rows)


def loop_escape_table(dev_uids: set[str], patch_empty: dict) -> pd.DataFrame:
    rows = []
    for tr in iter_traces(load_meta(), uids=dev_uids):
        st, det, fired_at = PrefixState("u", "i"), PatternDetector(), None
        for s in tr.steps:
            st.update(s)
            if st.is_action(s):
                det.update(s, st)
                if fired_at is None and "repeat_loop" in det.active(st):
                    fired_at = st.step
        if fired_at is not None:
            after = st.step - fired_at
            rows.append({"resolved": bool(tr.resolved), "after": after, "empty": patch_empty[tr.uid],
                         "group": f"escaped (>= {ESCAPE_AFTER} more actions)" if after >= ESCAPE_AFTER else "stayed stuck"})
    df = pd.DataFrame(rows)
    return df.groupby("group").agg(runs=("resolved", "size"), pass_rate=("resolved", "mean"),
                                   empty_patch=("empty", "mean"), median_more_actions=("after", "median")).reset_index()


def oof_runs(models: RiskModels, feats: pd.DataFrame) -> pd.DataFrame:
    """The agent's decision for each dev run, from out-of-fold scores, as the Governor would decide it."""
    prefix = build_swegym()
    prefix = prefix[prefix["checkpoint"] == "abs"][["uid", "step", "tokens_so_far"]]
    runs = feats[["uid", "instance_id", "resolved", "total_tokens"]].rename(columns={"total_tokens": "tokens_total"})
    used, decision = {}, {}
    for k in sorted(k for k in models.checkpoints if models.can_abort(k)):
        m = models.checkpoints[k]
        hit = m.oof[m.oof["p_cal"] < m.tau].merge(prefix[prefix["step"] == k], on="uid")
        for uid, t in zip(hit["uid"], hit["tokens_so_far"]):
            if uid not in decision:
                decision[uid], used[uid] = Decision.abort.value, float(t)
    low = dict(zip(models.final.oof["uid"], models.final.oof["p_cal"] < models.final.tau))
    runs["decision"] = [decision.get(u) or (Decision.reject.value if low[u] else Decision.flag_for_review.value)
                        for u in runs["uid"]]
    runs["tokens_used"] = [used.get(u, t) for u, t in zip(runs["uid"], runs["tokens_total"])]
    return runs


def empty_rule_table(runs: pd.DataFrame, patch_empty: dict) -> pd.DataFrame:
    empty = runs["uid"].map(patch_empty).to_numpy(bool)
    y = runs["resolved"].to_numpy(bool)
    stopped = runs["decision"].isin([Decision.abort.value, Decision.reject.value]).to_numpy()
    rows = []
    for name, stop in (("agent (ABORT + REJECT)", stopped), ("rule: reject if patch empty", empty)):
        rows.append({"policy": name, "runs_stopped": float(stop.mean()), "failures_stopped": float(stop[~y].mean()),
                     "passing_lost": float(stop[y].mean()), "reviewed": float((~stop).mean()),
                     "pass_rate_in_review": float(y[~stop].mean())})
    rows.append({"policy": "share of agent's stops with an empty patch", "runs_stopped": float(empty[stopped].mean())})
    rows.append({"policy": "stops among non-empty patches only", "runs_stopped": float(stopped[~empty].mean()),
                 "failures_stopped": float(stopped[~empty & ~y].mean()), "passing_lost": float(stopped[~empty & y].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    splits = pd.read_csv(config.SPLITS_PATH)
    dev_uids = set(splits.loc[splits["split"] == "dev", "uid"])
    feats = pd.read_parquet(config.FEATURES_PATH, columns=["uid", "instance_id", "resolved", "total_tokens", "patch_empty"])
    totals = feats[feats["uid"].isin(dev_uids)]
    patch_empty = dict(zip(totals["uid"], totals["patch_empty"]))
    models = RiskModels.fit(RiskConfig())
    tables = {"A. Hint patterns": pattern_table(dev_uids), "A2. Loop definition": loop_table(dev_uids),
              "B. Abort budget": abort_table(models, totals),
              "C. Review budget": review_table(models),
              "D. Loop escape": loop_escape_table(dev_uids, patch_empty)}
    runs = oof_runs(models, totals)
    tables["E. REJECT vs the empty-patch rule"] = empty_rule_table(runs, patch_empty)
    tables["F. Reroute, mean and std over seeds"] = over_seeds(runs, config.SEEDS)
    OUT.with_suffix(".json").write_text(json.dumps({k: v.to_dict("records") for k, v in tables.items()}, indent=1))
    lines = [f"Dev only ({len(totals)} runs, base rate {totals['resolved'].mean():.3f}), out-of-fold scores.", ""]
    for name, t in tables.items():
        lines += [f"### {name}", "", "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        lines += ["| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row) + " |"
                  for row in t.itertuples(index=False)]
        lines.append("")
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
