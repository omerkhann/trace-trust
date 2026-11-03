"""Write evaluation outputs: results/metrics.json, results/metrics.md, results/reliability.png."""
from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.eval.metrics import reliability_curve
from src.schema import EvalReport

SERIES = "#2a78d6"     # categorical slot 1
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def _fmt(x, std=None, ci=None, digits: int = 3) -> str:
    """'0.812 ± 0.004 [0.770, 0.851]'; std shown only when non-zero."""
    if x is None:
        return "n/a"
    s = f"{x:.{digits}f}"
    if std and round(std, digits) > 0:
        s += f" ± {std:.{digits}f}"
    if ci:
        s += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return s


def _gap(r: EvalReport) -> str:
    if r.pooled_auroc is None or r.within_task_auroc is None:
        return "n/a"
    return _fmt(r.pooled_auroc - r.within_task_auroc, ci=r.auroc_gap_ci)


def write_metrics(reports: list[EvalReport], extra: dict, name: str = "metrics") -> None:
    """-> results/<name>.json and results/<name>.md"""
    config.RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"generated": date.today().isoformat(), **extra, "reports": [r.model_dump() for r in reports]}
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=1))

    seeds = extra.get("seeds")
    lines = [f"Point estimates are the mean over seeds {seeds} (± std across seeds, shown only where non-zero). "
             f"Brackets: 95% task-level bootstrap interval ({extra.get('n_boot')} resamples of tasks). "
             "gap = pooled minus within-task AUROC.", ""] if seeds else []
    for eval_set in dict.fromkeys(r.eval_set for r in reports):
        rs = [r for r in reports if r.eval_set == eval_set]
        r0 = rs[0]
        lines += [f"### {eval_set}: {r0.n_runs} runs, base rate {r0.base_rate:.3f}, "
                  f"{r0.n_mixed_tasks} mixed tasks / {r0.n_pairs} same-task pairs", "",
                  "| method | pooled AUROC | within-task AUROC | gap | Brier | ECE |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.method} | {_fmt(r.pooled_auroc, r.pooled_auroc_std, r.pooled_auroc_ci)} "
                  f"| {_fmt(r.within_task_auroc, r.within_task_auroc_std, r.within_task_auroc_ci)} | {_gap(r)} "
                  f"| {_fmt(r.brier, r.brier_std, digits=4)} | {_fmt(r.ece, r.ece_std, digits=4)} |"
                  for r in rs]
        lines.append("")
    (config.RESULTS_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def reliability_plot(preds: dict[str, pd.DataFrame], path=None,
                     title: str = "Reliability on held-out test") -> None:
    """Small multiples, one panel per method: mean predicted p vs observed resolve rate per bin."""
    path = path or config.RESULTS_DIR / "reliability.png"
    curves = {m: reliability_curve(p["label"].to_numpy(float), p["p_success"].to_numpy(float))
              for m, p in preds.items()}
    hi = min(1.0, 1.1 * max(max(c["mean_p"].max(), c["frac_pos"].max()) for c in curves.values()))

    fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 3.3), sharex=True, sharey=True)
    axes = [axes] if len(curves) == 1 else axes
    for ax, (method, c) in zip(axes, curves.items()):
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.plot(c["mean_p"], c["frac_pos"], color=SERIES, lw=2, zorder=2)
        ax.scatter(c["mean_p"], c["frac_pos"], s=36, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_title(method, color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("mean predicted p(resolved)", color=MUTED, fontsize=8)
    axes[0].set_ylabel("observed resolve rate", color=MUTED, fontsize=8)
    fig.suptitle(f"{title} (15 equal-mass bins); dashed = perfect calibration",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
