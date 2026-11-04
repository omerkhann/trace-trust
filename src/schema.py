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

    @classmethod
    def feature_names(cls) -> list[str]:
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

    @classmethod
    def feature_names(cls) -> list[str]:
        return [f for f in cls.model_fields
                if f not in {"uid", "instance_id", "resolved", "checkpoint", "prefix_frac"}]


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
