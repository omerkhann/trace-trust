"""Paper figures, drawn only from files in results/ and logs/sample/. Static, for print: direct labels and distinct markers, so
nothing depends on color alone. Widths suit an IEEE column (3.5 in) or full page width (7.16 in).

    python -m experiments.figures
-> results/figures/fig_*.{png,pdf}
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import config

FIG = config.RESULTS_DIR / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed order, never cycled
ONE_COL, TWO_COL = 3.5, 7.16
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load(name: str) -> dict:
    return json.loads((config.RESULTS_DIR / f"{name}.json").read_text())


def tradeoff() -> None:
    """E5: failures stopped against passes lost, on test and Lite. Up and left is better."""
    r = load("governor_eval")["results"]
    order = [("full", "Full agent", "o"), ("no_early_warning", "No early warning", "s"), ("no_telemetry", "No telemetry", "^"), ("empty_patch_rule", "Empty-patch rule", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.7))
    ax = axes[0]
    nudge = {"full": (0, 9), "no_early_warning": (0, -15), "no_telemetry": (0, -17), "empty_patch_rule": (7, 7)}
    for i, (k, label, mk) in enumerate(order):
        m = r["test"][k]["runs"]
        x, y = 100 * m["passes_lost"], 100 * m["failures_stopped"]
        ax.errorbar(x, y, xerr=[[x - 100 * m["passes_lost_ci"][0]], [100 * m["passes_lost_ci"][1] - x]],
                    yerr=[[y - 100 * m["failures_stopped_ci"][0]], [100 * m["failures_stopped_ci"][1] - y]],
                    fmt=mk, color=SERIES[i], ms=6, elinewidth=1, capsize=0, mec="white", mew=0.6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=nudge[k], color=INK, fontsize=7.5, ha="left" if k == "empty_patch_rule" else "center")
    ax.set(title="(a) SWE-Gym test", xlabel="Passing runs lost (%)", ylabel="Failing runs stopped (%)", xlim=(-2, 30), ylim=(0, 85))
    ax = axes[1]
    order2 = [("full", "Calibrated", "o", 0), ("no_calibration", "No calibration", "v", 4), ("no_early_warning", None, None, None), ("empty_patch_rule", "Empty-patch\nrule", "D", 3)]
    for k, label, mk, c in order2:
        if label is None:
            continue
        m = r["lite"][k]["all"]
        x, y = 100 * m["passes_lost"], 100 * m["failures_stopped"]
        ax.errorbar(x, y, xerr=100 * m.get("passes_lost_std", 0), yerr=100 * m.get("failures_stopped_std", 0), fmt=mk, color=SERIES[c], ms=6, elinewidth=1, mec="white", mew=0.6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(-7, 8) if k == "full" else (7, 8), ha="right" if k == "full" else "left", fontsize=7.5)
    ax.axvline(10, color=MUTED, lw=0.8, ls="--")
    ax.annotate("10% budget set on dev", (10, 78), textcoords="offset points", xytext=(4, 0), fontsize=7, color=MUTED)
    ax.set(title="(b) SWE-bench Lite, unseen repositories", xlabel="Passing runs lost (%)", xlim=(-2, 30), ylim=(0, 85))
    fig.tight_layout()
    save(fig, "fig_tradeoff")


def reroute() -> None:
    """Reroute: three small multiples on one row, one measure each, so no shared scale is implied."""
    r = load("governor_eval")["results"]["test"]
    rows = {x["policy"]: x for x in r["full"]["reroute"] if x["subset"] == "all_tasks"}
    rule = next(x for x in r["empty_patch_rule"]["reroute"] if x["subset"] == "all_tasks" and x["policy"] == "governor_reroute")
    items = [("One random attempt", rows["random_attempt"]), ("Review every attempt", rows["review_all"]), ("Empty-patch rule + reroute", rule), ("Agent + reroute", rows["governor_reroute"])]
    names = [n for n, _ in items]
    panels = [("(a) Tasks solved (%)", lambda x: 100 * x["solved_mean"], lambda x: 100 * x["solved_std"], "{:.1f}"),
              ("(b) Reviews per task", lambda x: x["reviews_per_task_mean"], lambda x: x["reviews_per_task_std"], "{:.2f}"),
              ("(c) Tokens per task (k)", lambda x: x["tokens_per_task_mean"] / 1000, lambda x: x["tokens_per_task_std"] / 1000, "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 1.9), sharey=True)
    for ax, (title, val, err, f) in zip(axes, panels):
        vals = [val(x) for _, x in items]
        ys = list(range(len(items)))[::-1]
        ax.barh(ys, vals, xerr=[err(x) for _, x in items], height=0.55, color=SERIES[0], error_kw={"elinewidth": 0.8, "ecolor": INK})
        for y, v, (_, x) in zip(ys, vals, items):
            ax.text(v + err(x) + max(vals) * 0.04, y, f.format(v), va="center", fontsize=7, color=INK)
        ax.set_title(title, loc="left")
        ax.set_yticks(ys, names)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_ylim(-0.6, len(items) - 0.1)
    fig.tight_layout()
    save(fig, "fig_reroute")


def transfer() -> None:
    """Gap 2: pooled AUROC of each predictor as the data moves away from where it was trained."""
    h = {r["method"]: r for r in load("gap2_moatless")["h2"]}
    methods = [("telemetry_lr", "LR, 11 features"), ("telemetry_gbm", "GBM, 11 features"), ("telemetry20_lr", "LR, 20 features"), ("telemetry20_gbm", "GBM, 20 features"), ("length_only", "Length only")]
    keys = [("test", "SWE-Gym test (OpenHands)", "o", 0), ("oh_lite", "OpenHands on Lite", "s", 1), ("norm", "Moatless on Lite, normalized", "^", 2), ("raw", "Moatless on Lite, raw", "v", 3)]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.6))
    ys = list(range(len(methods)))[::-1]
    for k, label, mk, c in keys:
        ax.scatter([h[m][k] for m, _ in methods], [y + (1.5 - c) * 0.17 for y in ys], marker=mk, s=26, color=SERIES[c], label=label, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.annotate("chance", (0.5, len(methods) - 0.45), textcoords="offset points", xytext=(3, 0), fontsize=7, color=MUTED)
    ax.set_yticks(ys, [n for _, n in methods])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Pooled AUROC (models fit on SWE-Gym dev)", xlim=(0.3, 0.9), ylim=(-0.6, len(methods) - 0.3))
    ax.legend(loc="upper center", bbox_to_anchor=(0.36, -0.2), ncol=2, fontsize=7, handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig_transfer")


def shap() -> None:
    """Gap 3: top features of the end-of-run model on test, with the signals H3 named marked."""
    d = load("shap_importance")
    rows = sorted(d["features"], key=lambda r: r["test"], reverse=True)
    named = set(d["h3_named"])
    fig, ax = plt.subplots(figsize=(ONE_COL, 3.3))
    ys = list(range(len(rows)))[::-1]
    ax.barh(ys, [r["test"] for r in rows], height=0.62, color=[SERIES[1] if r["feature"] in named else SERIES[0] for r in rows])
    ax.set_yticks(ys, [r["feature"] + (" *" if r["feature"] in named else "") for r in rows], fontsize=6.8)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean |SHAP| on SWE-Gym test (log-odds)")
    ax.text(0.98, 0.04, "* named by H3 as a top signal", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    save(fig, "fig_shap")


def timeline() -> None:
    """Execution paths: what each agent did, step by step (phases), with the governance agent's trust curve above it."""
    cases = [("test_s2r1410", "(a) Passing run: sent to a reviewer"), ("test_s0r1005", "(b) Failing run: aborted early")]
    phases = ["explore", "locate", "reproduce", "fix", "verify", "submit"]
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 2.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1, "wspace": 0.18}, sharex="col")
    for col, (name, title) in enumerate(cases):
        events = [json.loads(l) for l in (config.ROOT / "logs" / "sample" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        steps = [e for e in events if "decision" in e]
        out = events[-1]
        top, bot = axes[0][col], axes[1][col]
        curve = [(p["step"], p["p_success"]) for p in out["trust_curve"]]
        if curve:
            top.step([c[0] for c in curve], [100 * c[1] for c in curve], where="post", color=SERIES[0], lw=1.6)
            top.plot([c[0] for c in curve], [100 * c[1] for c in curve], "o", color=SERIES[0], ms=4, mec="white", mew=0.5)
        last = steps[-1]
        top.annotate(f"{last['decision']}\nat action {last['step']}", (last["step"], 100 * (last.get("p_success") or 0)), textcoords="offset points", xytext=(-4, 12) if col == 0 else (8, 4), ha="right" if col == 0 else "left", fontsize=7)
        for e in steps:
            if e["decision"] == "INJECT_HINT":
                top.axvline(e["step"], color=SERIES[1], lw=1, ls=":")
                top.annotate("hint", (e["step"], 1), textcoords="offset points", xytext=(2, 2), fontsize=6.5, color=MUTED)
        top.set_title(title, loc="left")
        top.set_ylabel("p(resolved), %" if col == 0 else "")
        top.set_ylim(0, 40)
        total = out["steps_total"]
        for e in steps:
            if e["observed"]:
                ph = e["phase_to"] or "explore"
                bot.barh(0, 0.92, left=e["step"] - 0.46, height=0.8, color=SERIES[phases.index(ph)])
        if out["steps_seen"] < total:
            bot.barh(0, total - out["steps_seen"], left=out["steps_seen"] + 0.5, height=0.8, color=GRID)
            bot.text((out["steps_seen"] + total) / 2 + 0.5, 0, "not run after abort", ha="center", va="center", fontsize=7, color=INK)
        bot.set_yticks([])
        bot.grid(False)
        bot.spines["left"].set_visible(False)
        bot.set_xlabel("Agent actions")
        bot.set_xlim(0.3, max(total, last["step"]) + 0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(len(phases))]
    fig.legend(handles, phases, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.17), fontsize=7, handlelength=1, columnspacing=1)
    save(fig, "fig_timeline")


def architecture() -> None:
    """The system: each Phase 1 gap's component is marked."""
    boxes = [(0.00, 0.60, 0.15, 0.3, "Logged runs\nOpenHands\nMoatlessTools", None),
             (0.20, 0.60, 0.15, 0.3, "Scaffold adapters\nand normalizer\n(Step, Trace)", "Gap 2"),
             (0.40, 0.60, 0.15, 0.3, "Telemetry\nPrefixState (live)\n20 run features", "Gap 3"),
             (0.60, 0.60, 0.17, 0.3, "Risk models, calibration\nThresholds from dev\nSHAP explanations", "Gap 1, 3"),
             (0.82, 0.60, 0.18, 0.3, "Governance agent\nobserve, score, detect,\ndecide, act", None),
             (0.82, 0.08, 0.18, 0.3, "CONTINUE, INJECT_HINT,\nABORT, REJECT,\nFLAG_FOR_REVIEW", None),
             (0.56, 0.08, 0.19, 0.3, "Human gate\napprove, override, stop", None),
             (0.30, 0.08, 0.19, 0.3, "Reroute\nafter ABORT or REJECT", None),
             (0.00, 0.08, 0.20, 0.3, "Every step logged (JSONL)\nEvaluation: E1 to E6", None)]
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.1))
    ax.set_axis_off()
    ax.set(xlim=(-0.01, 1.01), ylim=(0, 1.05))
    for x, y, w, h, text, gap in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc="#f4f3ef" if not gap else "#dbe8f8", ec=SERIES[0] if gap else MUTED, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.6)
        if gap:
            ax.text(x + w / 2, y + h + 0.035, gap, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=INK)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
    for a, b in [((0.155, 0.75), (0.195, 0.75)), ((0.355, 0.75), (0.395, 0.75)), ((0.555, 0.75), (0.595, 0.75)), ((0.775, 0.75), (0.815, 0.75)),
                 ((0.91, 0.595), (0.91, 0.385)), ((0.815, 0.23), (0.755, 0.23)), ((0.555, 0.23), (0.495, 0.23)), ((0.35, 0.385), (0.10, 0.595))]:
        arrow(a, b)
    ax.text(0.245, 0.50, "next attempt", fontsize=6, color=MUTED, ha="left", va="bottom")
    save(fig, "fig_architecture")


def main() -> None:
    for fn in (architecture, tradeoff, reroute, transfer, shap, timeline):
        fn()
    print("figures ->", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
