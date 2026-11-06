"""TraceTrust viewer: watch the locked agent replay a run, and browse its saved runs and results.
It never retunes the agent and never writes a log or a result.

Run from the repository root:  .venv/Scripts/streamlit run src/app/viewer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # streamlit puts only this file's folder on the path

import altair as alt
import pandas as pd
import streamlit as st

from src.agent.risk import RiskModels
from src.app import data, replay
from src.config import RESULTS_DIR
from src.data.loader import load_meta

INK, MUTED, GRID, ACCENT = "#1a1a1a", "#52514e", "#e4e3df", "#2a78d6"
DECISIONS = {"ABORT": "Aborted", "REJECT": "Rejected", "FLAG_FOR_REVIEW": "Sent to review", "HUMAN_STOP": "Stopped by human"}
OUTCOME_NOTES = {
    "Failure stopped": "Correct: this run failed, and the agent stopped it before a person spent time on it.",
    "Pass reviewed": "Correct: this run passed, and the agent sent it to a person.",
    "Failure missed": "Wrong: this run failed, but the agent sent it to a person.",
    "Pass lost": "Wrong: this run passed, but the agent stopped it.",
}

st.set_page_config(page_title="TraceTrust", layout="wide")
st.markdown("<style>[data-testid='stMainBlockContainer']{max-width:1080px;padding-top:2.5rem}"
            "[data-testid='stMetricValue']{font-size:1.35rem}</style>", unsafe_allow_html=True)


@st.cache_data
def load_runs(dataset: str) -> tuple[pd.DataFrame, int]:
    return data.runs_with_logs(dataset)


def trust_chart(outcome) -> alt.LayerChart:
    curve = pd.DataFrame(outcome.trust_curve)
    x = alt.X("step:Q", title="Agent actions", scale=alt.Scale(domain=[0, max(outcome.steps_total, 1)], nice=False),
              axis=alt.Axis(tickMinStep=1, format="d"))
    top = max(0.2, float(curve["p_success"].max()) * 1.25)  # a common floor, so low scores read as low
    y = alt.Y("p_success:Q", title="Trust, p(resolved)", scale=alt.Scale(domain=[0, top], nice=False),
              axis=alt.Axis(format=".0%", tickCount=5))
    tip = [alt.Tooltip("step:Q", title="Action"), alt.Tooltip("p_success:Q", title="Trust", format=".1%")]
    line = alt.Chart(curve).mark_line(color=ACCENT, strokeWidth=2).encode(x=x, y=y)
    dots = alt.Chart(curve).mark_circle(color=ACCENT, size=70, opacity=1).encode(x=x, y=y, tooltip=tip)
    last = curve.tail(1).assign(label=lambda d: d["p_success"].map("{:.1%}".format))
    end = alt.Chart(last).mark_text(align="right", dx=-2, dy=-13, color=INK, fontWeight=500).encode(x=x, y=y, text="label:N")
    layers = [line, dots, end]
    if outcome.hints:
        hints = pd.DataFrame(outcome.hints)
        layers.insert(0, alt.Chart(hints).mark_rule(color=MUTED, strokeDash=[3, 3], strokeWidth=1).encode(
            x=alt.X("step:Q"), tooltip=[alt.Tooltip("step:Q", title="Hint at action"), alt.Tooltip("pattern:N", title="Pattern")]))
    return (alt.layer(*layers).properties(height=250)
            .configure(font="IBM Plex Sans")
            .configure_axis(grid=True, gridColor=GRID, domainColor=MUTED, tickColor=MUTED, labelColor=MUTED,
                            titleColor=MUTED, titleFontWeight="normal", labelFontSize=12, titleFontSize=12)
            .configure_axisX(grid=False)
            .configure_view(stroke=None))


def factors_table(factors: list[dict]) -> str:
    rows = "\n".join(f"| `{f['feature']}` | {round(f['value'], 2) + 0:g} | {'raises' if f['effect'] > 0 else 'lowers'} ({f['effect']:+.2f}) |"
                     for f in factors)
    return f"| Factor | Value | Effect on trust |\n|---|---|---|\n{rows}"


def show_run(row: pd.Series) -> None:
    events, outcome = data.load_run(row["log"])
    st.subheader(outcome.instance_id)
    st.markdown(OUTCOME_NOTES[row["outcome"]])

    a, b, c, d = st.columns(4)
    a.metric("Agent decision", DECISIONS.get(outcome.final_decision.value, outcome.final_decision.value))
    b.metric("Real result", "Passed" if outcome.resolved else "Failed", help="Hidden tests. The agent never sees this.")
    last = outcome.p_final if outcome.p_final is not None else (outcome.trust_curve[-1]["p_success"] if outcome.trust_curve else None)
    c.metric("Last trust score", "n/a" if last is None else f"{last:.1%}")
    d.metric("Actions run", f"{outcome.steps_seen} of {outcome.steps_total}")

    final = data.final_event(events)
    if final is not None and final.reason:
        st.caption(f"Agent's reason: {final.reason}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Trust during the run**")
        if outcome.trust_curve:
            st.altair_chart(trust_chart(outcome), use_container_width=True, theme=None)
            note = "Each dot is a scored checkpoint; the last one includes the final patch if the run finished."
            st.caption(note + (" Dashed lines mark hints." if outcome.hints else ""))
        else:
            st.caption("The run ended before the first checkpoint, so it has no trust scores.")
    with right:
        st.markdown("**Why this score**")
        factors = data.top_factors(events)
        st.markdown(factors_table(factors) if factors else "No factors were logged for this run.")
        if outcome.hints:
            st.markdown("**Hints fired**")
            st.markdown("\n".join(f"- Action {h['step']}: `{h['pattern']}`" for h in outcome.hints))

    steps_section(events, outcome.uid)


def readable_steps(steps: pd.DataFrame) -> pd.DataFrame:
    return steps.drop(columns="event").assign(
        Error=steps["Error"].map({True: "error", False: ""}),
        Trust=steps["Trust"].map(lambda p: "" if pd.isna(p) else f"{p:.1%}"))


@st.fragment
def steps_section(events: list, uid: str) -> None:
    """Runs on its own, so selecting a row doesn't redraw the rest of the page."""
    st.markdown("**Every action the agent watched**")
    steps = data.steps_frame(events)
    shown = readable_steps(steps)
    picked = st.dataframe(
        shown, hide_index=True, use_container_width=True, height=min(38 + 35 * len(steps), 390),
        on_select="rerun", selection_mode="single-row", key=f"steps-{uid}",
        column_config={"Target": st.column_config.TextColumn(width="large"),
                       "Action no.": st.column_config.NumberColumn(width="small")})
    rows = picked.selection.rows if picked else []
    if rows:
        event = events[int(steps.iloc[rows[0]]["event"])]
        st.markdown(f"**Tool calls at action {event.step}**")
        st.code(json.dumps([t.model_dump() for t in event.tool_calls], indent=2), language="json")
    else:
        st.caption("Select a row to see the agent's tool calls for that action, with inputs and outputs.")


@st.cache_resource(show_spinner=False)
def fitted_models() -> RiskModels:
    return RiskModels.fit()


@st.cache_resource(show_spinner=False)
def run_metadata() -> pd.DataFrame:
    return load_meta()


@st.cache_resource(show_spinner=False, max_entries=32)
def cached_trace(uid: str):
    return replay.load_trace(uid, run_metadata())


def start_session(uid: str, log: Path, live: bool) -> replay.ReplaySession:
    if not live:
        return replay.saved_session(*data.load_run(log))
    with st.spinner("Preparing the agent and loading this run. The first time can take 30 seconds."):
        return replay.live_session(cached_trace(uid), fitted_models())


def gate_box(session: replay.ReplaySession) -> None:
    question = session.pending
    with st.container(border=True):
        verb = "abort this run" if question.decision.value == "ABORT" else "reject this run"
        st.markdown(f"**The agent proposes to {verb} and is waiting for you.**")
        st.caption(f"Agent's reason: {question.reason}")
        for col, (label, choice) in zip(st.columns([1, 1, 1, 3]), (("Approve", "approve"), ("Overrule", "override"), ("Stop", "stop"))):
            if col.button(label, key=f"gate-{choice}", use_container_width=True):
                session.answer(choice)
                session.playing = True
                st.rerun()
        st.caption("Approve lets the decision stand. Overrule continues the run, or sends it to a person at the end. "
                   "Stop ends the replay.")


def finished_box(session: replay.ReplaySession) -> None:
    outcome = session.outcome
    label = data.outcome_label(outcome.final_decision.value, outcome.resolved)
    a, b, c = st.columns(3)
    a.metric("Final decision", DECISIONS.get(outcome.final_decision.value, outcome.final_decision.value))
    b.metric("Real result", "Passed" if outcome.resolved else "Failed", help="Hidden tests. Read only now, after the replay.")
    c.metric("Tokens used", f"{outcome.tokens_used:,} of {outcome.tokens_total:,}")
    if any(a != "approve" for a in session.answers):
        st.markdown(f"This run {'passed' if outcome.resolved else 'failed'}. You overruled or stopped the agent, so this isn't its own "
                    "decision. Reported results use the agent's own decisions, so they aren't affected.")
    else:
        st.markdown(OUTCOME_NOTES[label])
    factors = data.top_factors(session.events)
    if factors:
        st.markdown("**Why this score**")
        st.markdown(factors_table(factors))


def replay_view(session: replay.ReplaySession) -> None:
    """Advances the run by one event each time it runs, then draws it."""
    if session.playing:
        session.step()
        if session.waiting or session.finished:
            session.playing = False
            st.rerun()

    now = session.events[-1] if session.events else None
    curve = session.trust_curve()
    a, b, c, d = st.columns(4)
    done = sum(1 for e in session.events if e.observed and e.decision.value not in data.BLOCKING)
    a.metric("Action", f"{done} of {session.total_actions}")
    b.metric("Phase", (now.phase_to if now else "") or "not started")
    c.metric("Trust", f"{curve[-1]['p_success']:.1%}" if curve else "no score yet")
    d.metric("Status", "Finished" if session.finished else "Waiting for you" if session.waiting else "Running" if session.playing else "Paused")

    if session.waiting:
        gate_box(session)
    if session.finished:
        finished_box(session)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Trust during the run**")
        progress = SimpleNamespace(trust_curve=curve, steps_total=session.total_actions, hints=session.hints())
        if curve:
            st.altair_chart(trust_chart(progress), use_container_width=True, theme=None)
            if progress.hints:
                st.caption("Dashed lines mark hints.")
        else:
            st.caption("The first score comes after 5 actions.")
    with right:
        st.markdown("**What the agent just did**")
        if now is None:
            st.caption("Nothing yet.")
        else:
            if now.observed:
                st.markdown(f"Saw `{now.observed.get('action')}`: {data.target_text(now.observed) or 'no target'}")
            else:
                st.markdown("Scored the finished run, using its final patch.")
            st.markdown("Tools called: " + ", ".join(f"`{t.tool}`" for t in now.tool_calls))
            if now.hint:
                st.markdown(f"Hint, not delivered in replay: *{now.hint}*")

    if session.events:
        st.markdown("**Actions so far, newest first**")
        steps = data.steps_frame(session.events)
        if not steps.empty:
            st.dataframe(readable_steps(steps).iloc[::-1], hide_index=True, use_container_width=True, height=280,
                         column_config={"Target": st.column_config.TextColumn(width="large"),
                                        "Action no.": st.column_config.NumberColumn(width="small")})


def replay_tab() -> None:
    live = replay.has_agent_data()
    runs, _ = load_runs("test")
    kinds = {"End in an abort": "ABORT", "End in a reject": "REJECT", "Go to review": "FLAG_FOR_REVIEW"}
    top = st.columns([2, 4, 1])
    kind = top[0].selectbox("Show runs that", list(kinds), help="From the saved test run. The real result stays hidden until the end.")
    subset = runs[runs["decision"] == kinds[kind]].set_index("uid")
    if subset.empty:
        st.info("No runs of this kind are available here.")
        return
    uid = top[1].selectbox("Run", subset.index, format_func=lambda u: f"{subset.loc[u, 'instance_id']}  ({u})", key="replay-run")
    speeds = {"Slow": 1.0, "Normal": 0.5, "Fast": 0.2}
    speed = top[2].selectbox("Speed", list(speeds), index=1)

    session = st.session_state.get("replay")
    if session is not None and session.uid != uid:
        session = st.session_state["replay"] = None

    buttons = st.columns([1, 1, 5])
    if session is None:
        if buttons[0].button("Start", type="primary", use_container_width=True):
            st.session_state["replay"] = start_session(uid, subset.loc[uid, "log"], live)
            st.rerun()
    else:
        if not session.finished and not session.waiting:
            if buttons[0].button("Pause" if session.playing else "Resume", use_container_width=True):
                session.playing = not session.playing
                st.rerun()
        if buttons[1].button("Restart", use_container_width=True):
            st.session_state["replay"] = start_session(uid, subset.loc[uid, "log"], live)
            st.rerun()

    if live:
        st.caption("A replay of a logged run on the held-out test set. The actions come from the log. The agent's scores, hints "
                   "and decisions are computed now by the locked code. The logged run can't react, so hints aren't delivered and "
                   "this isn't a live monitor. Nothing here is saved.")
    else:
        st.caption("The SWE-Gym data isn't on this machine, so this steps through a saved log. The agent isn't computing anything, "
                   "and you can't overrule it here. To run the real agent, follow the install and run steps in the README first.")
    if session is None:
        return
    st.divider()
    st.fragment(replay_view, run_every=speeds[speed] if session.playing else None)(session)


def runs_tab() -> None:
    top = st.columns([2, 2, 3])
    label = top[0].selectbox("Dataset", list(data.DATASETS))
    dataset = data.DATASETS[label]
    runs, total = load_runs(dataset)
    if runs.empty:
        st.info("No logs found. Run `python -m experiments.governor_eval` to create them.")
        return
    counts = runs["outcome"].value_counts()
    kept = [o for o in data.OUTCOMES.values() if o in counts]
    outcome = top[1].selectbox("Outcome", kept, format_func=lambda o: f"{o} ({counts[o]})")
    subset = runs[runs["outcome"] == outcome]
    uid = top[2].selectbox("Run", subset["uid"], format_func=lambda u: f"{subset.set_index('uid').loc[u, 'instance_id']}  ({u})")
    if len(runs) < total:
        st.caption(f"Showing the {len(runs)} committed sample runs of {total}. "
                   "Run `python -m experiments.governor_eval` to browse them all.")
    st.divider()
    show_run(subset[subset["uid"] == uid].iloc[0])


def results_tab() -> None:
    report = RESULTS_DIR / "governor_eval.md"
    st.subheader("The agent, with each part removed")
    table, prose = data.split_table(data.md_section(report, "### SWE-Gym test: runs"))
    st.markdown(data.readable_table(table))
    st.caption(prose.replace("full, on", "Full agent, on", 1).replace(" %", "%"))
    st.image(str(RESULTS_DIR / "figures" / "fig_tradeoff.png"), width=640)

    st.subheader("On unseen repositories (SWE-bench Lite)")
    table, _ = data.split_table(data.md_section(report, "### SWE-bench Lite: runs"))
    st.markdown(data.readable_table(table, subset="all"))
    st.caption("All Lite runs. Mean ± std over 3 draws of the 20 labelled tasks used to set the pass rate.")

    st.subheader("Retrying failed attempts")
    table, _ = data.split_table(data.md_section(report, "### Reroute on test: tasks solved and cost per task"))
    st.markdown(data.reroute_table(table))
    st.caption("SWE-Gym test, all 488 tasks. Mean ± std over 3 seeds.")

    st.subheader("Gap-resolution table")
    st.markdown(data.md_section(ROOT / "README.md", "## Gap-resolution table"))


def failures_tab() -> None:
    report = RESULTS_DIR / "failure_analysis.md"
    st.subheader("Where the agent goes wrong")
    label = st.columns([2, 5])[0].selectbox("Dataset", list(data.DATASETS), key="failures-dataset")
    dataset = data.DATASETS[label]
    heading = next(h for h in data.headings(report) if h.startswith(f"### {dataset}:"))
    table, prose = data.split_table(data.md_section(report, heading))
    st.markdown(prose.split("\n\n")[0].replace(" %", "%"))
    st.markdown(data.readable_table(table))
    st.caption(data.failure_notes(json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))[dataset]))

    st.subheader("Representative runs")
    st.caption("SWE-Gym test. Chosen by a fixed rule, not by hand: the run nearest its group's median score.")
    cases = RESULTS_DIR / "failure_cases.md"
    options = data.headings(cases)
    case = st.columns([5, 2])[0].selectbox("Case", options, format_func=lambda h: h.removeprefix("### "))
    st.markdown(data.md_section(cases, case))


def main() -> None:
    commit = json.loads((RESULTS_DIR / "governor_eval.json").read_text(encoding="utf-8"))["commit"]
    st.title("TraceTrust")
    st.caption(f"The governance agent, locked at commit {commit}: watch it replay a run, or browse its saved runs and results. "
               "This page doesn't retune the agent or change any result.")
    watch, runs, results, failures = st.tabs(["Replay", "Runs", "Results", "Failures"])
    with watch:
        replay_tab()
    with runs:
        runs_tab()
    with results:
        results_tab()
    with failures:
        failures_tab()


main()
