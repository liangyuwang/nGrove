"""Read-only, trusted Operator views; never an Agent or Worker control API."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .results import normalize_result_report


DEFAULT_MAX_AGE_SECONDS = 60.0
_EXECUTION_STATES = {"queued", "running", "terminated", "unknown"}
_CLEANUP_STATES = {"confirmed", "unconfirmed", "unknown"}
_OBSERVATION_FIELDS = {
    "schema_version",
    "episode_id",
    "source",
    "observed_at",
    "execution_state",
    "workload_cleanup",
    "storage_cleanup",
    "detail",
}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("observed_at must be an ISO 8601 timestamp with timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observed_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _freshness(timestamp: datetime, now: datetime, max_age: float) -> Dict[str, Any]:
    age = (now - timestamp).total_seconds()
    return {
        "age_seconds": round(age, 3),
        "freshness": "future" if age < 0 else "stale" if age > max_age else "fresh",
    }


def _read_json(path: Path) -> tuple[Dict[str, Any], Any]:
    evidence: Dict[str, Any] = {"path": str(path), "availability": "available"}
    try:
        with path.open("r", encoding="utf-8") as stream:
            # Capture metadata from the same opened snapshot, even if a writer
            # atomically replaces the pathname while this reader is running.
            modified = datetime.fromtimestamp(
                os.fstat(stream.fileno()).st_mtime, timezone.utc
            )
            evidence["modified_at"] = modified.isoformat()
            return evidence, json.load(stream)
    except FileNotFoundError:
        evidence.update(availability="missing", error="file not found")
    except (ValueError, UnicodeError) as exc:
        evidence.update(availability="invalid", error=str(exc))
    except OSError as exc:
        evidence.update(availability="unreadable", error=str(exc))
    return evidence, None


def _validate_observation(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) - _OBSERVATION_FIELDS:
        raise ValueError("deployment observation must be an object with known fields")
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("unsupported deployment observation schema_version")
    for field in ("episode_id", "source"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"deployment observation requires {field}")
    _timestamp(payload.get("observed_at"))
    if payload.get("execution_state") not in _EXECUTION_STATES:
        raise ValueError("invalid deployment execution_state")
    result = dict(payload)
    for field in ("workload_cleanup", "storage_cleanup"):
        result.setdefault(field, "unknown")
        if result[field] not in _CLEANUP_STATES:
            raise ValueError(f"invalid deployment {field}")
    if "detail" in result and not isinstance(result["detail"], str):
        raise ValueError("deployment detail must be text")
    return result


def _core_phase(state: Dict[str, Any]) -> str:
    status = state["status"]
    if status == "created":
        return "before_research"
    if status == "active":
        return "research"
    if status == "closed":
        if (state.get("agent_system") or {}).get("termination_reason") is None or (
            (state.get("agent_control") or {}).get("cleanup_complete") is not True
            or state.get("active_lease")
            or state.get("pending_lease_teardowns")
        ):
            return "research_teardown"
        if state.get("test_status") in {None, "pending"}:
            return "test_pending"
        if state.get("analysis_status") == "pending":
            return "analysis_pending"
        return "publication_pending"
    return "terminal"


def read_operator_status(
    episode: Path,
    *,
    observation_path: Optional[Path] = None,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Combine independent evidence without modifying or finalizing an episode.

    A recent Core mtime is not a heartbeat. Only an episode-bound, fresh
    deployment observation can describe current execution. Terminal Core
    results remain valid regardless of age or external progress snapshots.
    """
    if (
        isinstance(max_age_seconds, bool)
        or not math.isfinite(max_age_seconds)
        or max_age_seconds <= 0
    ):
        raise ValueError("max_age_seconds must be finite and positive")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    now = now.astimezone(timezone.utc)
    core, payload = _read_json(episode.resolve() / "state.json")
    state = None
    if core["availability"] == "available":
        try:
            state = normalize_result_report(payload)
            if (
                not isinstance(state.get("episode_id"), str)
                or not state["episode_id"].strip()
            ):
                raise ValueError("Core snapshot requires episode_id")
            core.update(
                _freshness(_timestamp(core["modified_at"]), now, max_age_seconds)
            )
            core.update(
                status=state["status"],
                outcome=state["outcome"],
                phase=_core_phase(state),
                environment=state.get("environment"),
                execution_issues=state.get("execution_issues"),
                closure_reason=state.get("closure_reason"),
                research_sealed_at=state.get("research_sealed_at"),
                leases_used=state.get("leases_used", 0),
                active_lease_id=(state.get("active_lease") or {}).get("lease_id"),
                candidate_count=len(state["candidate_commits"]),
                incumbent_id=state.get("incumbent_id"),
                final_candidate_id=state.get("final_candidate_id"),
                test_status=state.get("test_status"),
                analysis_status=state.get("analysis_status"),
                test_performance_score=state.get("test_performance_score"),
                test_cost=state.get("test_cost"),
                failure=state.get("failure"),
                agent_termination_reason=(state.get("agent_system") or {}).get(
                    "termination_reason"
                ),
                agent_control_cleanup=(state.get("agent_control") or {}).get(
                    "cleanup_complete"
                ),
                pending_lease_teardowns=state.get("pending_lease_teardowns", []),
                deployment_teardown_failures=state.get(
                    "deployment_teardown_failures", []
                ),
            )
            if "benchmark" in state:
                core["benchmark"] = {
                    name: state["benchmark"][name]
                    for name in (
                        "comparison_key",
                        "policy_compliant",
                        "leaderboard_eligible",
                    )
                }
            elif state.get("benchmark_release") is not None:
                core["benchmark"] = {
                    name: state.get(name)
                    for name in (
                        "comparison_key",
                        "policy_compliant",
                        "leaderboard_eligible",
                    )
                }
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            core.update(availability="invalid", error=str(exc))
            state = None

    deployment: Dict[str, Any] = {"availability": "not_provided"}
    execution = "unknown"
    workload_cleanup = storage_cleanup = "unknown"
    if observation_path is not None:
        deployment, payload = _read_json(observation_path.resolve())
        if deployment["availability"] == "available":
            try:
                observation = _validate_observation(payload)
                deployment.update(observation)
                deployment.update(
                    _freshness(
                        _timestamp(observation["observed_at"]), now, max_age_seconds
                    )
                )
                if state is None:
                    deployment.update(
                        availability="unbound",
                        error="Core episode identity unavailable",
                    )
                elif observation["episode_id"] != state["episode_id"]:
                    deployment.update(
                        availability="mismatched",
                        error="deployment episode_id does not match Core",
                    )
                elif deployment["freshness"] == "fresh":
                    execution = observation["execution_state"]
                    workload_cleanup = observation["workload_cleanup"]
                    storage_cleanup = observation["storage_cleanup"]
            except (ValueError, TypeError) as exc:
                deployment.update(availability="invalid", error=str(exc))

    summary = "unknown"
    if state is not None and state["outcome"] is not None:
        summary = state["status"]
    elif execution == "terminated":
        summary = "interrupted"
    elif execution in {"queued", "running"}:
        summary = execution
    return {
        "schema_version": 1,
        "checked_at": now.isoformat(),
        "episode_id": state["episode_id"] if state is not None else None,
        "summary": summary,
        "execution_state": execution,
        "workload_cleanup": workload_cleanup,
        "storage_cleanup": storage_cleanup,
        "core": core,
        "deployment": deployment,
    }


def format_operator_status(view: Dict[str, Any]) -> str:
    """Render a compact terminal view, keeping every evidence source separate."""

    def display(value: Any) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)[1:-1]
        return json.dumps(value, ensure_ascii=False)

    core, deployment = view["core"], view["deployment"]
    lines = [
        f"Episode: {display(view['episode_id'])}",
        f"Checked: {view['checked_at']}",
        f"Summary: {view['summary']}",
        f"Execution: {view['execution_state']}",
    ]
    if view["summary"] == "unknown":
        lines.append(
            "Current execution cannot be confirmed from the available evidence."
        )
    elif view["summary"] == "interrupted":
        lines.append("Execution terminated; Core has no terminal outcome.")
    if core["availability"] == "available":
        lines.extend(
            [
                f"Core snapshot: {core['status']} / {core['phase']} ({core['freshness']}; written {core['modified_at']})",
                f"Outcome: {display(core['outcome'])}; failure: {display(core['failure'])}",
                f"Execution issues: {display(core.get('execution_issues'))}",
                f"Research sealed: {display(core['research_sealed_at'])}; reason: {display(core['closure_reason'])}",
                f"Lease snapshot: {core['leases_used']} started; active: {display(core['active_lease_id'])}",
                f"Candidates: {core['candidate_count']}; incumbent: {display(core['incumbent_id'])}; final: {display(core['final_candidate_id'])}",
                f"TEST: {display(core['test_status'])}; ANALYSIS: {display(core['analysis_status'])}",
                f"Agent termination: {display(core['agent_termination_reason'])}; control cleanup: {display(core['agent_control_cleanup'])}",
                f"Pending Lease teardowns: {display(core['pending_lease_teardowns'])}; teardown failures: {display(core['deployment_teardown_failures'])}",
            ]
        )
        if "benchmark" in core:
            benchmark = core["benchmark"]
            lines.extend(
                [
                    f"Benchmark comparison key: {display(benchmark['comparison_key'])}",
                    f"Benchmark policy compliant: {display(benchmark['policy_compliant'])}; leaderboard eligible: {display(benchmark['leaderboard_eligible'])}",
                ]
            )
        if core["outcome"] == "success":
            lines.append(
                f"TEST score: {core['test_performance_score']}; seconds/task: {core['test_cost']['time_per_task']}"
            )
    else:
        lines.append(
            f"Core snapshot: {core['availability']} ({display(core.get('error'))})"
        )
    lines.append(f"Core source: {display(core['path'])}")
    lines.append(f"Deployment observation: {deployment['availability']}")
    if "source" in deployment:
        lines.append(
            f"Deployment source: {display(deployment['source'])}; observed {display(deployment['observed_at'])} ({deployment['freshness']})"
        )
        lines.append(
            f"Last observed execution: {display(deployment['execution_state'])}"
        )
        lines.append(
            f"Last observed cleanup: workloads={display(deployment['workload_cleanup'])}; storage={display(deployment['storage_cleanup'])}"
        )
    for field in ("path", "error", "detail"):
        if field in deployment:
            lines.append(f"Deployment {field}: {display(deployment[field])}")
    lines.append(
        f"Deployment cleanup: workloads={view['workload_cleanup']}; storage={view['storage_cleanup']}"
    )
    return "\n".join(lines)
