"""Step through one run for the viewer, one action per tick, with a person answering the agent's gate.

The live source is the real, unchanged Governor, advanced one event at a time. A generator can't wait for
a browser click, so when the agent asks a question the gate raises NeedsHuman, which ends the generator.
After the person answers, a new generator is run from the start with the answers so far and skips the
events already shown. A replay is deterministic and takes milliseconds, so this gives the same run.

Without the dataset, the saved log is stepped through instead: the agent isn't computing anything then.
Nothing here writes a log or a result.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator, Optional

from src import config
from src.agent.governor import Gate, Governor, GovernorConfig
from src.data.parser import iter_traces
from src.prefix import PREFIX_PATH, PrefixState
from src.schema import Decision, GovernorEvent, RunOutcome, Trace

Source = Callable[[Gate], Iterator[GovernorEvent | RunOutcome]]
SCORE_TOOLS = ("score_prefix_risk", "score_final_risk")
HUMAN_ANSWERS = ("approve", "override", "stop")


class NeedsHuman(Exception):
    """The agent proposed an ABORT or REJECT and no answer has been given yet."""

    def __init__(self, decision: Decision, step: int, reason: str):
        super().__init__(reason)
        self.decision, self.step, self.reason = decision, step, reason


class ScriptedGate:
    """Gives the answers so far, in order, then asks for the next one."""

    def __init__(self, answers: list[str]):
        self.answers, self.asked = answers, 0

    def ask(self, decision: Decision, uid: str, step: int, reason: str) -> str:
        if self.asked == len(self.answers):
            raise NeedsHuman(decision, step, reason)
        self.asked += 1
        return self.answers[self.asked - 1]


class ReplaySession:
    """One run in progress: what has been shown, and what it is waiting for."""

    def __init__(self, uid: str, total_actions: int, source: Source, live: bool):
        self.uid, self.total_actions, self.source, self.live = uid, total_actions, source, live
        self.answers: list[str] = []
        self.events: list[GovernorEvent] = []
        self.outcome: Optional[RunOutcome] = None
        self.pending: Optional[NeedsHuman] = None
        self.playing = True
        self._gen = source(ScriptedGate(self.answers))

    @property
    def waiting(self) -> bool:
        return self.pending is not None

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    def step(self) -> None:
        """Advance by one event, unless the run is over or waiting for a person."""
        if self.finished or self.waiting:
            return
        try:
            item = next(self._gen)
        except NeedsHuman as question:
            self.pending = question
            return
        if isinstance(item, RunOutcome):
            self.outcome = item
        else:
            self.events.append(item)

    def answer(self, choice: str) -> None:
        if choice not in HUMAN_ANSWERS:
            raise ValueError(f"unknown answer {choice!r}")
        self.answers.append(choice)
        self.pending = None
        self._gen = self.source(ScriptedGate(self.answers))
        for shown in self.events:      # already on screen; a replay is deterministic
            assert next(self._gen).seq == shown.seq

    def trust_curve(self) -> list[dict]:
        """The scores so far: the same points the agent's RunOutcome reports at the end. While the agent waits
        for an answer, its event isn't out yet, so the score it is asking about comes from its stated reason."""
        curve = [{"step": c.input["step"], "p_success": c.output["p_success"]}
                 for e in self.events for c in e.tool_calls if c.tool in SCORE_TOOLS and c.output]
        if self.pending is not None:
            stated = re.search(r"p\(resolved\) ([\d.]+)", self.pending.reason)
            if stated:
                curve.append({"step": self.pending.step, "p_success": float(stated.group(1))})
        return curve

    def hints(self) -> list[dict]:
        return [{"step": e.step, "pattern": e.reason.removeprefix("pattern ")} for e in self.events if e.hint]


def agent_data_paths() -> list[Path]:
    """What fitting the agent and loading a run need. Checked first, so the viewer never starts a large download."""
    return [config.META_PATH, config.FEATURES_PATH, PREFIX_PATH, *[config.RAW_DIR / s for s in config.HF_SHARDS]]


def has_agent_data() -> bool:
    return all(p.exists() for p in agent_data_paths())


def load_trace(uid: str, meta) -> Trace:
    return next(iter_traces(meta, uids={uid}))


def live_session(trace: Trace, models) -> ReplaySession:
    """The real agent, with the settings of the E3 full configuration."""
    st = PrefixState(trace.uid, trace.instance_id)
    total = sum(st.is_action(s) for s in trace.steps)
    return ReplaySession(trace.uid, total, lambda gate: Governor(models, GovernorConfig(), gate).replay(trace), live=True)


def saved_session(events: list[GovernorEvent], outcome: RunOutcome) -> ReplaySession:
    return ReplaySession(outcome.uid, outcome.steps_total, saved_source(events, outcome), live=False)


def saved_source(events: list[GovernorEvent], outcome: RunOutcome) -> Source:
    return lambda gate: iter([*events, outcome])
