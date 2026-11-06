"""Command-line interface of the governance agent.

    python -m src.agent.cli models                          # fitted models and thresholds (dev only)
    python -m src.agent.cli replay --uid s0r1               # one run, printed step by step
    python -m src.agent.cli replay --uid s0r1 --interactive # approve / override / stop ABORT and REJECT
    python -m src.agent.cli replay --split dev --limit 200  # many runs, with a summary
    python -m src.agent.cli replay --dataset lite --uid claude35_v2.1:astropy__astropy-12907
    python -m src.agent.cli task --instance getmoto__moto-5699 --seed 42   # abort and reroute across attempts

Ablation switches: --no-calibration (Gap 1), --no-normalizer (Gap 2; Lite v1.9 runs only),
--no-early-warning and --no-telemetry (Gap 3), --no-hints.
Every run writes JSONL to logs/<config>/<uid>.jsonl: one line per step, then the outcome.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from src import config
from src.agent.governor import Governor, GovernorConfig
from src.agent.risk import RiskConfig, RiskModels
from src.schema import Decision, GovernorEvent, RunOutcome, Trace

LOG_DIR = config.ROOT / "logs"


class CliGate:
    """Asks the person at the terminal about every ABORT and REJECT."""
    CHOICES = {"a": "approve", "o": "override", "s": "stop"}

    def ask(self, decision: Decision, uid: str, step: int, reason: str) -> str:
        print(f"\n  >> {decision.value} proposed for {uid} at step {step}: {reason}")
        while True:
            c = input("     [a]pprove  [o]verride  [s]top > ").strip().lower()[:1]
            if c in self.CHOICES:
                return self.CHOICES[c]


def config_name(args: argparse.Namespace) -> str:
    off = [n for n, on in (("calibration", args.calibration), ("normalizer", args.normalizer),
                           ("early_warning", args.early_warning), ("telemetry", args.telemetry),
                           ("hints", args.hints)) if not on]
    return "full" if not off else "no_" + "+".join(off)


def load_traces(args: argparse.Namespace) -> Iterator[Trace]:
    uids = set(args.uid) if args.uid else None
    n = 0
    if args.dataset == "lite":
        from src.data.swebench_lite import iter_lite
        source = (tr for _, tr, _ in iter_lite(normalize=args.normalizer))
    else:
        from src.data.loader import load_meta
        from src.data.parser import iter_traces
        if args.split:
            splits = pd.read_csv(config.SPLITS_PATH)
            uids = set(splits.loc[splits["split"] == args.split, "uid"])
        source = iter_traces(load_meta(), uids=uids)
    for tr in source:
        if uids is not None and tr.uid not in uids:
            continue
        yield tr
        n += 1
        if args.limit and n >= args.limit:
            return


def _safe(uid: str) -> str:
    return re.sub(r"[^\w.-]", "_", uid)


def print_event(e: GovernorEvent) -> None:
    if not e.observed:
        what = "(end of run)"
    else:
        what = e.observed.get("path") or e.observed.get("command") or e.observed.get("tool") or ""
    risk = f"p={e.p_success:.3f}" if e.p_success is not None else "p=  -  "
    print(f"{e.step:>4} {e.phase_to or '-':<9} {risk} {e.decision.value:<15} "
          f"{e.observed.get('action', ''):<7} {str(what).splitlines()[0][:60] if what else ''}")
    for c in e.tool_calls:
        if c.tool == "score_final_risk" and c.output and c.output.get("top_factors"):
            print("     top factors: " + ", ".join(f"{f['feature']} {f['effect']:+.2f}" for f in c.output["top_factors"]))
    if e.reason:
        print(f"     -> {e.reason}")
    if e.hint:
        print(f"     hint: {e.hint}")


def print_outcome(o: RunOutcome) -> None:
    print(f"\n{o.uid}: {o.final_decision.value} after {o.steps_seen}/{o.steps_total} actions, "
          f"{o.tokens_used}/{o.tokens_total} tokens, {len(o.hints)} hints; actually "
          f"{'resolved' if o.resolved else 'failed'} (label read only after the replay)")


def summarize(rows: list[RunOutcome]) -> str:
    n, fail = len(rows), [o for o in rows if not o.resolved]
    ok = [o for o in rows if o.resolved]
    stopped = (Decision.abort, Decision.reject)
    used, total = sum(o.tokens_used for o in rows), sum(o.tokens_total for o in rows)
    reviewed = [o for o in rows if o.final_decision == Decision.flag_for_review]
    lines = [f"{n} runs: " + ", ".join(f"{k.value} {v}" for k, v in Counter(o.final_decision for o in rows).items())]
    if fail:
        lines.append(f"failures stopped: {sum(o.final_decision in stopped for o in fail) / len(fail):.1%} "
                     f"({len(fail)} failing runs)")
    if ok:
        lines.append(f"passing runs aborted: {sum(o.final_decision == Decision.abort for o in ok) / len(ok):.1%}, "
                     f"rejected: {sum(o.final_decision == Decision.reject for o in ok) / len(ok):.1%} "
                     f"({len(ok)} passing runs)")
    if reviewed:
        lines.append(f"sent to review: {len(reviewed) / n:.1%} of runs, pass rate there "
                     f"{sum(bool(o.resolved) for o in reviewed) / len(reviewed):.1%} (overall {len(ok) / n:.1%})")
    if total:
        lines.append(f"tokens used: {used / total:.1%} of what the runs consumed")
    lines.append(f"runs with a hint: {sum(bool(o.hints) for o in rows)}; "
                 f"failed tool calls: {sum(o.degraded_calls for o in rows)}")
    return "\n".join(lines)


def replay_one(gov: Governor, trace: Trace, log_dir: Optional[Path], verbose: bool) -> RunOutcome:
    f = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        f = (log_dir / f"{_safe(trace.uid)}.jsonl").open("w", encoding="utf-8")
    try:
        for item in gov.replay(trace):
            if f:
                f.write(item.model_dump_json() + "\n")
            if verbose and isinstance(item, GovernorEvent):
                print_event(item)
        return item
    finally:
        if f:
            f.close()


def build_governor(args: argparse.Namespace, gate=None) -> Governor:
    models = RiskModels.fit(RiskConfig(calibration=args.calibration, telemetry=args.telemetry,
                                       far_abort=args.far_abort, far_review=args.far_review))
    return Governor(models, GovernorConfig(name=config_name(args), early_warning=args.early_warning,
                                           hints=args.hints and args.telemetry), gate)


def run_task(args: argparse.Namespace) -> int:
    from src.data.loader import load_meta
    from src.data.parser import iter_traces
    meta = load_meta()
    uids = set(meta.loc[meta["instance_id"] == args.instance, "uid"])
    if not uids:
        print(f"no logged runs for task {args.instance}", file=sys.stderr)
        return 1
    traces = list(iter_traces(meta, uids=uids))
    order = np.random.default_rng(args.seed).permutation(len(traces))
    gov = build_governor(args, CliGate() if args.interactive else None)
    log_dir = None if args.no_log else LOG_DIR / config_name(args)
    print(f"{args.instance}: {len(traces)} attempts, seed {args.seed}")
    for n, i in enumerate(order, 1):
        o = replay_one(gov, traces[i], log_dir, verbose=False)
        print(f"  attempt {n}: {o.uid} -> {o.final_decision.value} after {o.steps_seen}/{o.steps_total} actions, "
              f"{o.tokens_used} tokens")
        if o.final_decision == Decision.flag_for_review:
            print(f"  sent to a human; the patch actually {'passes' if o.resolved else 'fails'}")
            return 0
        if o.final_decision == Decision.human_stop:
            print("  stopped by a human")
            return 0
        if n < len(traces):
            print("  rerouting to the next attempt")
    print("  every attempt was stopped; nothing submitted")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="tracetrust", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("models", "replay", "task"):
        p = sub.add_parser(name)
        p.add_argument("--no-calibration", dest="calibration", action="store_false")
        p.add_argument("--no-telemetry", dest="telemetry", action="store_false")
        p.add_argument("--no-early-warning", dest="early_warning", action="store_false")
        p.add_argument("--no-hints", dest="hints", action="store_false")
        p.add_argument("--no-normalizer", dest="normalizer", action="store_false")
        p.add_argument("--far-abort", type=float, default=RiskConfig.far_abort,
                       help="false-alarm budget per abort checkpoint, on dev")
        p.add_argument("--far-review", type=float, default=RiskConfig.far_review,
                       help="false-alarm budget for the end-of-run REJECT, on dev")
    r = sub.choices["replay"]
    r.add_argument("--dataset", choices=["swegym", "lite"], default="swegym")
    r.add_argument("--uid", action="append", help="run id; repeat for several")
    r.add_argument("--split", choices=["dev", "test"], help="every run of a SWE-Gym split")
    r.add_argument("--limit", type=int)
    r.add_argument("--interactive", action="store_true", help="a human approves or overrides ABORT and REJECT")
    r.add_argument("--no-log", action="store_true")
    t = sub.choices["task"]
    t.add_argument("--instance", required=True, help="task id; its logged runs are the attempts")
    t.add_argument("--seed", type=int, default=config.SEED, help="order of the attempts")
    t.add_argument("--interactive", action="store_true")
    t.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "models":
        print(build_governor(args).models.summary().to_string(index=False))
        return 0
    if args.cmd == "task":
        return run_task(args)
    if not args.uid and not args.split and args.dataset == "swegym":
        ap.error("replay needs --uid or --split (or --dataset lite)")

    gov = build_governor(args, CliGate() if args.interactive else None)
    log_dir = None if args.no_log else LOG_DIR / config_name(args)
    rows = []
    verbose = bool(args.uid) and len(args.uid) <= 3
    for tr in load_traces(args):
        rows.append(replay_one(gov, tr, log_dir, verbose))
        if verbose:
            print_outcome(rows[-1])
    if not rows:
        print("no matching runs", file=sys.stderr)
        return 1
    if not verbose:
        print(summarize(rows))
    if log_dir:
        print(f"logs: {log_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
