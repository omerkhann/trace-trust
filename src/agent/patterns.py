"""Rule-based telemetry patterns the governance agent turns into hints, and its phase tracker.

Each pattern is cheap, uses only what a live monitor can see, and has a fixed hint. Whether a pattern
is worth acting on is measured on dev by experiments/governor_tuning.py (share of runs it fires on,
pass rate when it fires).

Phases follow the action categories of Bouzenia & Pradel (Phase 1 report, B.5):
explore -> locate -> reproduce -> fix -> verify -> submit. The phase is a logged belief about what the
agent is doing, derived from its actions.
"""
from __future__ import annotations

from collections import deque

from src.features import _TEST_RE, phase_of
from src.prefix import PrefixState
from src.schema import ActionType

LOOP_WINDOW = 8        # look for a repeated action among this many recent actions
LOOP_REPEATS = 4       # the same action this many times in the window is a loop (set on dev)
ERROR_STREAK = 3       # this many consecutive failing actions
NO_EDIT_BY = 15        # still no edit after this many actions

# Patterns allowed to trigger a hint. Dev rule (results/governor_tuning.md): runs where the pattern fires
# must pass at most half as often as runs where it doesn't. The others stay logged but send no hint.
HINT_PATTERNS = ("finish_without_edit", "repeat_loop")

HINTS = {
    "finish_without_edit": "You are about to finish without changing any code. Re-read the issue, locate the "
                           "code responsible and make the fix before finishing.",
    "finish_unverified": "You are about to finish without running tests since your last edit. Run the relevant "
                         "tests or a reproduction script first.",
    "repeat_loop": "You have run the same action several times without new results. Stop repeating it, state what "
                   "you learned, and try a different approach.",
    "error_streak": "Your last few actions all failed. Read the error messages, check your assumptions about paths "
                    "and environment, and change approach instead of retrying.",
    "no_edit_yet": "You have taken many actions without editing any code. Decide which file is most likely "
                   "responsible and attempt a fix.",
    "blind_edit": "You edited a file you have not viewed. View it first to confirm your change fits the surrounding "
                  "code.",
}


class PatternDetector:
    """Follows one run alongside its PrefixState. `active()` lists what currently holds, most urgent first."""

    def __init__(self, loop_window: int = LOOP_WINDOW, loop_repeats: int = LOOP_REPEATS):
        self.recent: deque[str] = deque(maxlen=loop_window)
        self.loop_repeats = loop_repeats
        self.err_streak = 0
        self.tested_since_edit = False
        self.last_blind = 0
        self.fresh_blind = False

    def update(self, s, st: PrefixState) -> None:
        """Call after st.update(s), for agent actions only."""
        if s.args_sig:
            self.recent.append(s.args_sig)
        self.err_streak = self.err_streak + 1 if s.is_error else 0
        if s.action_type in (ActionType.edit, ActionType.create):
            self.tested_since_edit = False
        elif s.action_type == ActionType.bash and s.command and _TEST_RE.search(s.command):
            self.tested_since_edit = True
        self.fresh_blind = st.blind_edits > self.last_blind
        self.last_blind = st.blind_edits

    def active(self, st: PrefixState) -> list[str]:
        out = []
        if self.recent and max(self.recent.count(x) for x in set(self.recent)) >= self.loop_repeats:
            out.append("repeat_loop")
        if self.err_streak >= ERROR_STREAK:
            out.append("error_streak")
        if st.step >= NO_EDIT_BY and st.edits == 0:
            out.append("no_edit_yet")
        if self.fresh_blind:
            out.append("blind_edit")
        return out

    def before_finish(self, st: PrefixState) -> list[str]:
        """Checked when the agent tries to finish, before the finish goes through."""
        if st.edits == 0:
            return ["finish_without_edit"]
        return [] if self.tested_since_edit else ["finish_unverified"]


def next_phase(prev: str, s, st: PrefixState) -> str:
    """Belief about the agent's phase after action s (st already includes s)."""
    return phase_of(prev, s, st.edits)
