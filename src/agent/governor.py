"""The governance agent's loop: observe a step, call its tools, decide, act, log.

The agent intercepts each action of a replayed run just before it would take effect:
  1. score_prefix_risk   at an absolute checkpoint, score the run so far with the calibrated model
  2. abort_run           if risk passes the threshold and this checkpoint is allowed to abort
  3. detect_patterns     check rule-based patterns; if one is new, inject_hint
  4. score_final_risk    when the run ends, score it using the final patch: REJECT or FLAG_FOR_REVIEW

Nothing is ever accepted automatically. At an 8 % pass rate even high-trust runs mostly fail, so the
agent's job at the end is to save reviewer time: REJECT low-trust runs, send the rest to a human.

ABORT and REJECT go through a gate, so a human can approve, override or stop. Any tool failure is
logged and the agent fails open (the run continues, or goes to a human), because a broken monitor
should not kill runs.

The agent never reads trace.resolved, and it reads trace.final_patch only at end-of-run triage.
Replay limit: a logged run can't react, so a hint can't change what happens next. Only how accurately
hints trigger can be measured, not their effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Protocol

from src.agent.patterns import HINT_PATTERNS, HINTS, PatternDetector, next_phase
from src.agent.risk import RiskModels
from src.features import extract
from src.prefix import PrefixState
from src.schema import ActionType, Decision, GovernorEvent, RunOutcome, ToolCall, Trace

COMMAND_CHARS = 200


@dataclass
class GovernorConfig:
    name: str = "full"
    early_warning: bool = True    # Gap 3 switch: off = no checkpoint scoring and no aborts
    hints: bool = True
    hint_cooldown: int = 5        # actions between hints
    max_hints: int = 3
    patterns: tuple[str, ...] = HINT_PATTERNS     # which patterns may trigger a hint


class Gate(Protocol):
    def ask(self, decision: Decision, uid: str, step: int, reason: str) -> str:
        """-> "approve", "override" or "stop"."""


class AutoGate:
    """No human in the loop: the agent's decisions stand."""
    def ask(self, decision: Decision, uid: str, step: int, reason: str) -> str:
        return "approve"


class Governor:
    def __init__(self, models: RiskModels, cfg: Optional[GovernorConfig] = None, gate: Optional[Gate] = None):
        self.models, self.cfg, self.gate = models, cfg or GovernorConfig(), gate or AutoGate()

    def _tool(self, calls: list[ToolCall], name: str, inp: dict, fn: Callable[[], dict]) -> Optional[dict]:
        """Run a tool and log it. On failure, log the error and return None; the caller fails open."""
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 (a monitor must survive any tool failure)
            calls.append(ToolCall(tool=name, input=inp, ok=False, error=f"{type(e).__name__}: {e}"))
            return None
        calls.append(ToolCall(tool=name, input=inp, output=out))
        return out

    def replay(self, trace: Trace) -> Iterator[GovernorEvent | RunOutcome]:
        """Yield one GovernorEvent per agent action, then the RunOutcome."""
        cfg = self.cfg
        st = PrefixState(trace.uid, trace.instance_id)       # label deliberately not passed
        det = PatternDetector()
        phase, seq, p_latest = "", 0, None
        hints, curve, seen = [], [], []
        last_hint, overrides, degraded = -10**6, 0, 0
        abort_step, tokens_used = None, None
        steps_total = sum(st.is_action(s) for s in trace.steps)
        tokens_total = sum(s.tokens + s.obs_tokens for s in trace.steps)

        def outcome(final: Decision, p_final: Optional[float] = None) -> RunOutcome:
            return RunOutcome(uid=trace.uid, instance_id=trace.instance_id, config=cfg.name, final_decision=final,
                              steps_seen=st.step, steps_total=steps_total,
                              tokens_used=st.tokens if tokens_used is None else tokens_used,
                              tokens_total=tokens_total, hints=hints, patterns_seen=seen, trust_curve=curve,
                              p_final=p_final, abort_step=abort_step, human_overrides=overrides,
                              degraded_calls=degraded, resolved=trace.resolved)

        for s in trace.steps:
            if not st.is_action(s):
                st.update(s)
                continue
            calls: list[ToolCall] = []
            decision, reason, hint, human = Decision.continue_, "", None, None
            k = st.step

            # 1-2. checkpoint risk, seen before action k+1 takes effect
            if cfg.early_warning and k in self.models.checkpoints:
                m = self.models.checkpoints[k]
                snap = st.snapshot().model_dump()
                out = self._tool(calls, "score_prefix_risk", {"step": k, "tokens": st.tokens},
                                 lambda: dict(zip(("p_raw", "p_success"), m.score(snap))))
                if out is None:
                    degraded += 1
                else:
                    p_latest = out["p_success"]
                    curve.append({"step": k, "p_success": p_latest})
                    if self.models.can_abort(k) and p_latest < m.tau:
                        reason = f"p(resolved) {p_latest:.3f} below abort threshold {m.tau:.3f} after {k} actions"
                        human = self.gate.ask(Decision.abort, trace.uid, k, reason)
                        if human == "stop":
                            decision = Decision.human_stop
                        elif human == "override":
                            overrides += 1
                            reason += "; overridden by human"
                        else:
                            decision = Decision.abort
                            abort_step, tokens_used = k, st.tokens
                            calls.append(ToolCall(tool="abort_run", input={"step": k},
                                                  output={"tokens_saved": tokens_total - st.tokens}))

            observed = {"action": s.action_type.value, "tool": s.tool_name, "path": s.path,
                        "command": (s.command or "")[:COMMAND_CHARS] or None}
            if decision in (Decision.abort, Decision.human_stop):     # the action never takes effect
                yield GovernorEvent(uid=trace.uid, seq=seq, step=k, observed=observed, tool_calls=calls,
                                    phase_from=phase, phase_to=phase, p_success=p_latest, decision=decision,
                                    reason=reason, human=human)
                yield outcome(decision)
                return

            # 3. patterns: a finish attempt is checked before it goes through, the rest after the action
            finishing = s.action_type == ActionType.finish
            pre = det.before_finish(st) if finishing else []
            st.update(s)
            det.update(s, st)
            observed |= {"is_error": s.is_error, "exit_code": s.exit_code, "tokens": s.tokens + s.obs_tokens}
            new_phase = next_phase(phase, s, st)
            active = self._tool(calls, "detect_patterns", {"step": st.step},
                                lambda: {"active": pre + det.active(st)})
            if active is None:
                degraded += 1
            else:
                for p in active["active"]:
                    if p not in seen:
                        seen.append(p)
                if cfg.hints and len(hints) < cfg.max_hints and st.step - last_hint >= cfg.hint_cooldown:
                    used = {h["pattern"] for h in hints}
                    fresh = [p for p in active["active"] if p in cfg.patterns and p not in used]
                    if fresh:
                        p, hint = fresh[0], HINTS[fresh[0]]
                        decision, reason = Decision.inject_hint, f"pattern {p}"
                        hints.append({"step": st.step, "pattern": p})
                        last_hint = st.step
                        calls.append(ToolCall(tool="inject_hint", input={"pattern": p},
                                              output={"delivered": False, "note": "replay: logged run cannot react"}))
            yield GovernorEvent(uid=trace.uid, seq=seq, step=st.step, observed=observed, tool_calls=calls,
                                phase_from=phase, phase_to=new_phase, p_success=p_latest, decision=decision,
                                reason=reason, hint=hint, human=human)
            phase, seq = new_phase, seq + 1

        # 4. end-of-run triage: only now is the final patch available
        calls, human = [], None
        m = self.models.final
        def score_final() -> dict:
            row = extract(trace).model_dump()
            raw, cal = m.score(row)
            return {"p_raw": raw, "p_success": cal, "top_factors": m.explain(row)}
        out = self._tool(calls, "score_final_risk", {"step": st.step}, score_final)
        if out is None:
            degraded += 1
            decision, reason, p_final = Decision.flag_for_review, "final scoring failed; sent to a human", None
        else:
            p_final = p_latest = out["p_success"]
            curve.append({"step": st.step, "p_success": p_final})
            low = p_final < m.tau
            decision = Decision.reject if low else Decision.flag_for_review
            reason = f"final p(resolved) {p_final:.3f} {'below' if low else 'at or above'} trust threshold {m.tau:.3f}"
        if decision == Decision.reject:
            human = self.gate.ask(decision, trace.uid, st.step, reason)
            if human == "override":
                overrides += 1
                decision, reason = Decision.flag_for_review, reason + "; overridden by human"
            elif human == "stop":
                decision = Decision.human_stop
        yield GovernorEvent(uid=trace.uid, seq=seq, step=st.step, tool_calls=calls, phase_from=phase, phase_to=phase,
                            p_success=p_latest, decision=decision, reason=reason, human=human)
        yield outcome(decision, p_final)

    def run(self, trace: Trace, sink: Optional[Callable[[GovernorEvent | RunOutcome], None]] = None) -> RunOutcome:
        for item in self.replay(trace):
            if sink:
                sink(item)
        return item
