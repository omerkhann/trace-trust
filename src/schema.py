"""Structured contracts passed between pipeline stages.

Trace / Step   : normalised telemetry for one agent run (parsed from OpenHands messages)
RunFeatures    : fixed-width feature vector extracted from a Trace
JudgeVerdict   : validated structured output of the LLM-as-judge
Prediction     : one method's probability for one run (out-of-fold on dev, or held out on test)
EvalReport     : metrics for one method
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ActionType(str, Enum):
    # agent actions (assistant turns)
    bash = "bash"
    view = "view"
    create = "create"
    edit = "edit"          # str_replace / insert / undo_edit
    finish = "finish"
    message = "message"    # assistant text with no tool call
    other_tool = "other_tool"
    # environment side
    observation = "observation"
    prompt = "prompt"      # system / user messages


class Step(BaseModel):
    """One event in a trajectory. Each assistant tool call becomes its own Step, and its
    observation is attached to the same Step (exit_code / is_error / obs_tokens)."""
    index: int = Field(ge=0)
    role: Role
    action_type: ActionType
    tool_name: Optional[str] = None
    command: Optional[str] = None          # bash command, or editor sub-command
    path: Optional[str] = None             # editor target path
    args_sig: Optional[str] = None         # short hash of tool name + raw arguments (repeat detection)
    exit_code: Optional[int] = None        # bash only, parsed from observation
    is_error: bool = False
    tokens: int = Field(0, ge=0)           # estimated tokens of the action text (chars/4)
    obs_tokens: int = Field(0, ge=0)       # estimated tokens of the observation
    thought_files: list[str] = []          # file names mentioned in the assistant's text for this turn


class Trace(BaseModel):
    uid: str
    instance_id: str
    run_id: str
    resolved: Optional[bool] = None        # label; None at inference time
    task: str = ""                         # first user message (issue text), truncated
    steps: list[Step]
    final_patch: str = ""

    @computed_field
    @property
    def n_actions(self) -> int:
        return sum(s.role == Role.assistant for s in self.steps)


PROGRESS_FEATURES = ("last_check_passed", "checked_after_last_edit", "fail_then_pass", "max_identical_streak",
                     "max_error_streak", "error_responsiveness", "first_edit_step", "phase_entropy",
                     "thought_action_mismatch")


class RunFeatures(BaseModel):
    uid: str
    instance_id: str
    resolved: Optional[bool] = None
    n_steps: int                  # agent actions (tool calls + plain messages)
    total_tokens: int             # estimated, actions + observations + prompts
    n_errors: int                 # non-zero exit / editor ERROR / runtime failure
    n_repeated_commands: int      # actions identical to an earlier one in the same run
    n_distinct_tools: int
    n_edits_without_view: int     # edits to a path never viewed/created before
    ran_tests: bool               # any bash command invoking a test runner
    patch_lines: int              # changed (+/-) lines in final git patch
    patch_files: int
    patch_empty: bool
    finished: bool                # agent called `finish` (vs. hitting iteration cap)
    # Added in Phase 2, see features.ProgressSignals. Live signals, so PrefixFeatures has them too.
    last_check_passed: bool = False       # last test/script run exited 0 (False if none, or exit code unknown)
    checked_after_last_edit: bool = False
    fail_then_pass: bool = False          # a check failed, and a later check passed
    max_identical_streak: int = 0         # longest run of the exact same action
    max_error_streak: int = 0             # longest run of failing actions
    error_responsiveness: float = 1.0     # share of errors followed by a different action (1 if no errors)
    first_edit_step: int = 0              # action of the first edit (= actions so far if none yet)
    phase_entropy: float = 0.0            # bits, over explore/locate/reproduce/fix/verify/submit
    thought_action_mismatch: float = 0.0  # share of file actions whose path isn't among the files the text names

    @classmethod
    def feature_names(cls) -> list[str]:
        """The Phase 1 feature set. Reported baselines use this, so their results stay reproducible."""
        return [f for f in cls.extended_feature_names() if f not in PROGRESS_FEATURES]

    @classmethod
    def extended_feature_names(cls) -> list[str]:
        return [f for f in cls.model_fields if f not in {"uid", "instance_id", "resolved"}]


class PrefixFeatures(BaseModel):
    """Telemetry of a run that is still going, after its first `step` agent actions (and their
    observations). Only what a live monitor can see: no final patch, no total length.
    Used by the early-warning model and the governance agent (Gap 3)."""
    uid: str
    instance_id: str
    resolved: Optional[bool] = None
    checkpoint: str               # "abs" (after a fixed number of actions) or "frac" (share of the run)
    prefix_frac: Optional[float] = None   # for "frac" rows: 0.25 / 0.5 / 0.75 (uses the total, so offline only)
    step: int                     # agent actions so far
    tokens_so_far: int
    n_errors: int
    recent_errors: int            # errors in the observations of the last 5 actions
    n_repeated_commands: int
    n_distinct_tools: int
    n_edits: int                  # edit/create actions
    n_files_edited: int
    n_edits_without_view: int
    steps_since_edit: int         # actions since the last edit (= step if none yet)
    ran_tests: bool
    last_check_passed: bool = False
    checked_after_last_edit: bool = False
    fail_then_pass: bool = False
    max_identical_streak: int = 0
    max_error_streak: int = 0
    error_responsiveness: float = 1.0
    first_edit_step: int = 0
    phase_entropy: float = 0.0
    thought_action_mismatch: float = 0.0

    @classmethod
    def feature_names(cls) -> list[str]:
        return [f for f in cls.extended_feature_names() if f not in PROGRESS_FEATURES]

    @classmethod
    def extended_feature_names(cls) -> list[str]:
        return [f for f in cls.model_fields
                if f not in {"uid", "instance_id", "resolved", "checkpoint", "prefix_frac"}]


class Decision(str, Enum):
    """What the governance agent does at a step of a run it is watching."""
    continue_ = "CONTINUE"
    inject_hint = "INJECT_HINT"          # send the agent a rule-based reflection hint
    abort = "ABORT"                      # stop the run; its budget can go to a fresh attempt
    reject = "REJECT"                    # run finished but looks failed: no reviewer time, try again
    flag_for_review = "FLAG_FOR_REVIEW"  # run finished and passes the trust gate: a human reviews it
    human_stop = "HUMAN_STOP"            # a human stopped the replay


class ToolCall(BaseModel):
    """One tool the governance agent used, with its input and output, as logged."""
    tool: str
    input: dict = {}
    output: Optional[dict] = None
    ok: bool = True
    error: Optional[str] = None


class GovernorEvent(BaseModel):
    """One logged step of the governance agent: what it saw, what it called, what it decided."""
    uid: str
    seq: int                              # event number within the run
    step: int                             # agent actions completed so far
    observed: dict = {}                   # the agent action being looked at (empty at end-of-run triage)
    tool_calls: list[ToolCall] = []
    phase_from: str = ""
    phase_to: str = ""
    p_success: Optional[float] = None     # latest calibrated p(resolved), if any
    decision: Decision
    reason: str = ""
    hint: Optional[str] = None
    human: Optional[str] = None           # approve / override / stop, when a human was asked


class RunOutcome(BaseModel):
    """How one replayed run ended under the governance agent. `resolved` is copied in only after
    the replay finishes, for scoring; the agent never reads it."""
    uid: str
    instance_id: str
    config: str                           # ablation name, e.g. "full" or "no_calibration"
    final_decision: Decision
    steps_seen: int
    steps_total: int
    tokens_used: int
    tokens_total: int
    hints: list[dict] = []                # {"step": int, "pattern": str}
    patterns_seen: list[str] = []
    trust_curve: list[dict] = []          # {"step": int, "p_success": float}
    p_final: Optional[float] = None
    abort_step: Optional[int] = None
    human_overrides: int = 0
    degraded_calls: int = 0               # tool calls that failed; the agent fails open
    resolved: Optional[bool] = None


class TaskFeatures(BaseModel):
    """Pre-execution features of the task alone (no trajectory). Used only by the
    prompt-only baseline, which measures how much signal is just task difficulty."""
    uid: str
    repo: str                     # from instance_id, e.g. "getmoto__moto"
    issue_tokens: int             # estimated tokens of the PR/issue description
    issue_code_blocks: int        # fenced ``` blocks in the issue
    issue_has_traceback: bool

    @classmethod
    def numeric_names(cls) -> list[str]:
        return ["issue_tokens", "issue_code_blocks", "issue_has_traceback"]


class JudgeVerdict(BaseModel):
    """Structured LLM-judge output, validated from the model's JSON reply."""
    verdict: Literal["PASS", "FAIL"]
    confidence: int = Field(ge=0, le=100)   # confidence in the verdict, 0-100
    rationale: str = ""

    @property
    def p_success(self) -> float:
        c = self.confidence / 100
        return c if self.verdict == "PASS" else 1 - c


class Prediction(BaseModel):
    uid: str
    instance_id: str
    method: str
    fold: int                     # dev fold, -1 for test runs
    p_success: float = Field(ge=0.0, le=1.0)
    label: bool


class EvalReport(BaseModel):
    method: str
    eval_set: str = "dev"         # "dev" (out-of-fold), "test" (held out) or "judge_sample" (subset of test)
    n_runs: int
    pooled_auroc: Optional[float]
    within_task_auroc: Optional[float]
    n_mixed_tasks: int
    n_pairs: int
    brier: float
    ece: float
    base_rate: float
    # Point metrics above are the mean over seeds; std is across seeds (None for a single seed).
    n_seeds: int = 1
    pooled_auroc_std: Optional[float] = None
    within_task_auroc_std: Optional[float] = None
    brier_std: Optional[float] = None
    ece_std: Optional[float] = None
    # 95% task-level bootstrap intervals, from the first seed's predictions.
    pooled_auroc_ci: Optional[tuple[float, float]] = None
    within_task_auroc_ci: Optional[tuple[float, float]] = None
    auroc_gap_ci: Optional[tuple[float, float]] = None       # pooled minus within-task
    # One-sided Wilcoxon signed-rank p that per-task AUROC > 0.5 (first seed's predictions).
    within_task_wilcoxon_p: Optional[float] = None
    # Best@k / Pass@k / Random@k over tasks with >= k runs (first seed's), see metrics.best_at_k.
    best_at_k: list[dict] = []
