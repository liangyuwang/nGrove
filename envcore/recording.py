"""Private research evidence and configurable artifact retention.

The Core event journal remains the source of ordering. This module stores
bounded, content-addressed payloads and reads references; it does not schedule
research, interpret Agent reasoning, or expose hidden evaluation to an Agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from .config import ConfigError


def _encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class RecordingConfig:
    """Launch-owned capture and post-evaluation retention policy."""

    capture_payloads: bool = True
    max_payload_bytes: int = 4 * 1024 * 1024
    max_total_payload_bytes: int = 64 * 1024 * 1024
    candidates: str = "all"
    lease_outputs: str = "all"

    def __post_init__(self) -> None:
        if not isinstance(self.capture_payloads, bool):
            raise ConfigError("recording.capture_payloads must be boolean")
        for name in ("max_payload_bytes", "max_total_payload_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"recording.{name} must be a positive integer")
        if self.candidates not in ("all", "final", "metadata"):
            raise ConfigError(
                "recording.artifacts.candidates must be all, final or metadata"
            )
        if self.lease_outputs not in ("all", "discard_after_finalize"):
            raise ConfigError(
                "recording.artifacts.lease_outputs must be all or discard_after_finalize"
            )

    @classmethod
    def from_manifest(cls, value: Any) -> "RecordingConfig":
        if not isinstance(value, dict):
            raise ConfigError("recording configuration must be an object")
        fields = {
            "schema_version",
            "capture_payloads",
            "max_payload_bytes",
            "max_total_payload_bytes",
            "artifacts",
        }
        if (
            set(value) - fields
            or isinstance(value.get("schema_version"), bool)
            or not isinstance(value.get("schema_version"), int)
            or value["schema_version"] != 1
        ):
            raise ConfigError("recording requires schema_version 1 and known fields")
        artifacts = value.get("artifacts", {})
        if not isinstance(artifacts, dict) or set(artifacts) - {
            "candidates",
            "lease_outputs",
        }:
            raise ConfigError("invalid recording.artifacts")
        return cls(
            **{
                k: v
                for k, v in value.items()
                if k not in {"schema_version", "artifacts"}
            },
            **artifacts,
        )

    @classmethod
    def load(cls, path: Path) -> "RecordingConfig":
        try:
            value = json.loads(path.expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read recording configuration: {exc}") from exc
        return cls.from_manifest(value)

    def manifest(self) -> Dict[str, Any]:
        value = asdict(self)
        artifacts = {name: value.pop(name) for name in ("candidates", "lease_outputs")}
        return {"schema_version": 1, **value, "artifacts": artifacts}

    def record(self) -> Dict[str, Any]:
        value = self.manifest()
        return {
            "configuration": value,
            "configuration_sha256": hashlib.sha256(_encode(value)).hexdigest(),
        }

    @classmethod
    def from_record(cls, record: Any) -> "RecordingConfig":
        if not isinstance(record, dict) or set(record) != {
            "configuration",
            "configuration_sha256",
        }:
            raise ConfigError("invalid recording configuration record")
        config = cls.from_manifest(record["configuration"])
        if config.record() != record:
            raise ConfigError("recording configuration identity mismatch")
        return config


class RecordingStore:
    """Private bounded payload store; capture failures do not change research."""

    def __init__(self, episode: Path, config: RecordingConfig) -> None:
        self.episode = episode
        self.config = config
        self.root = episode / "recording" / "payloads"
        if self.root.parent.is_symlink():
            raise ValueError("recording directory must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=False)
        self.root.parent.chmod(0o700)
        self.root.chmod(0o700)
        self.lock = threading.RLock()
        self.stored_bytes = 0
        self._saved: set[str] = set()
        self._failed_payload_bytes: Dict[str, int] = {}

    def payload(self, value: Any) -> Dict[str, Any]:
        """Retain the exact JSON value or an explicit reason for omission."""
        try:
            encoded = _encode(value)
        except (TypeError, ValueError, RecursionError, UnicodeError):
            return {"status": "invalid_json"}
        digest = hashlib.sha256(encoded).hexdigest()
        result = {"sha256": digest, "size_bytes": len(encoded)}
        if not self.config.capture_payloads:
            return {**result, "status": "not_captured"}
        if len(encoded) > self.config.max_payload_bytes:
            return {**result, "status": "omitted_size_limit"}
        with self.lock:
            destination = self.root / f"{digest}.json"
            if digest not in self._saved:
                created = False
                try:
                    if digest in self._failed_payload_bytes:
                        # Only retry files owned by a previous failed write.
                        # Release their reservation after cleanup succeeds.
                        destination.unlink(missing_ok=True)
                        self.stored_bytes -= self._failed_payload_bytes.pop(digest)
                    if (
                        self.stored_bytes + len(encoded)
                        > self.config.max_total_payload_bytes
                    ):
                        return {**result, "status": "omitted_total_limit"}
                    # A failed write is never advertised as a saved object.
                    descriptor = os.open(
                        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    created = True
                    try:
                        with os.fdopen(descriptor, "wb") as stream:
                            descriptor = -1
                            stream.write(encoded)
                            stream.flush()
                            os.fsync(stream.fileno())
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                except OSError as exc:
                    if created:
                        try:
                            destination.unlink(missing_ok=True)
                        except OSError:
                            # Reserve the full attempted size even if the file
                            # is partial or cannot currently be inspected.
                            self._failed_payload_bytes[digest] = len(encoded)
                            self.stored_bytes += len(encoded)
                    return {
                        **result,
                        "status": "capture_failed",
                        "error_type": type(exc).__name__,
                    }
                self._saved.add(digest)
                self.stored_bytes += len(encoded)
            return {
                **result,
                "status": "saved",
                "path": destination.relative_to(self.episode).as_posix(),
            }


def read_recording(episode: Path) -> Dict[str, Any]:
    """Read an Operator-only evidence view without rewriting source records.

    No log content or artifact reference is executed. Payload paths are derived
    from validated digests, never followed from a producer-supplied path.
    """
    from .results import normalize_result_report

    episode = episode.resolve()
    try:
        state = normalize_result_report(
            json.loads((episode / "state.json").read_text(encoding="utf-8"))
        )
    except OSError as exc:
        raise ConfigError(f"cannot read episode state: {exc}") from exc
    config = RecordingConfig.from_record(state.get("recording"))
    events = []
    issues = []
    seen = set()
    previous_seq = 0
    try:
        journal = (episode / "events.jsonl").open(encoding="utf-8")
    except OSError as exc:
        if state["status"] != "created" or not isinstance(exc, FileNotFoundError):
            issues.append({"code": "event_journal_unavailable"})
        journal = nullcontext(())
    with journal as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(
                    event.get("type"), str
                ):
                    raise ValueError("invalid event")
                seq = event["seq"]
                event_id = event["event_id"]
                if (
                    isinstance(seq, bool)
                    or not isinstance(seq, int)
                    or seq <= previous_seq
                    or event_id != f"{state['episode_id']}:event:{seq}"
                    or event_id in seen
                ):
                    raise ValueError("invalid event identity or sequence")
                links = event.get("links", [])
                if not isinstance(links, list) or any(
                    not isinstance(link, dict)
                    or not isinstance(link.get("event_id"), str)
                    or not isinstance(link.get("relation"), str)
                    for link in links
                ):
                    raise ValueError("invalid event links")
                for name in ("action_id", "execution_id"):
                    if event.get(name) is not None and not isinstance(event[name], str):
                        raise ValueError("invalid event association")
            except (ValueError, TypeError, KeyError):
                issues.append({"line": number, "code": "invalid_event"})
                continue
            if seq != previous_seq + 1:
                issues.append({"event_id": event_id, "code": "event_gap"})
            previous_seq = seq
            seen.add(event_id)
            events.append(event)

    events_by_seq = {event["seq"]: event for event in events}

    def check_state_event(sequence, event_type, **fields):
        event = events_by_seq.get(sequence)
        if event is None or any(
            event.get(name) != value
            for name, value in {"type": event_type, **fields}.items()
        ):
            issues.append(
                {
                    "code": "state_event_missing"
                    if event is None
                    else "state_event_mismatch",
                    "event_id": f"{state['episode_id']}:event:{sequence}",
                    "expected_type": event_type,
                }
            )

    # Journal continuity alone cannot detect a lost suffix. These references
    # were persisted only after their corresponding events were written.
    for commit in state["candidate_commits"]:
        check_state_event(
            commit["event_seq"],
            "CANDIDATE_COMMITTED",
            candidate_id=commit["candidate_id"],
            commit_index=commit["commit_index"],
            lease_id=commit["source_lease"],
        )
    for declaration in state["incumbent_declarations"]:
        check_state_event(
            declaration["event_seq"],
            "INCUMBENT_DECLARED",
            candidate_id=declaration["candidate_id"],
            declaration_index=declaration["declaration_index"],
            previous_candidate_id=declaration["previous_candidate_id"],
        )
    publication = state.get("result_publication")
    if publication is not None:
        check_state_event(
            publication["prepared_event_seq"],
            "RESULT_PUBLICATION_PREPARED",
            publication_id=publication["publication_id"],
            candidate_id=state["final_candidate_id"],
        )

    payloads = {}
    artifacts = {}
    starts = {}
    completed = set()
    executions = set()
    finished_executions = set()
    for event in events:
        if event["type"] == "ARTIFACT_RETENTION_DEFERRED":
            issues.append(
                {"event_id": event["event_id"], "code": "artifact_retention_deferred"}
            )
        if event["type"] == "ACTION_STARTED":
            starts[event.get("action_id")] = event["event_id"]
        elif event["type"] == "ACTION_RETURNED":
            completed.add(event.get("action_id"))
        elif event["type"] == "LEASE_COMMAND_STARTED":
            executions.add(event.get("execution_id"))
        elif event["type"] in {"LEASE_COMMAND_FINISHED", "LEASE_COMMAND_FAILED"}:
            finished_executions.add(event.get("execution_id"))
        for relation in event.get("links", []):
            if relation.get("event_id") not in seen:
                issues.append(
                    {
                        "event_id": event["event_id"],
                        "code": "unresolved_event_reference",
                    }
                )
        artifact = event.get("artifact")
        if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str):
            key = artifact["artifact_id"]
            artifacts[key] = {**artifacts.get(key, {}), **artifact}
        ref = event.get("payload")
        if not isinstance(ref, dict):
            continue
        if ref.get("status") != "saved":
            issues.append(
                {
                    "event_id": event["event_id"],
                    "code": "payload_" + str(ref.get("status")),
                }
            )
            continue
        digest = ref.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            issues.append(
                {"event_id": event["event_id"], "code": "invalid_payload_identity"}
            )
            continue
        if digest in payloads:
            continue
        path = episode / "recording" / "payloads" / f"{digest}.json"
        try:
            if (
                path.is_symlink()
                or path.parent.is_symlink()
                or path.parent.parent.is_symlink()
            ):
                raise ValueError("payload path is a link")
            if path.stat().st_size > config.max_payload_bytes:
                raise ValueError("payload exceeds capture limit")
            data = path.read_bytes()
            if (
                len(data) != ref.get("size_bytes")
                or hashlib.sha256(data).hexdigest() != digest
            ):
                raise ValueError("payload identity mismatch")
            payloads[digest] = json.loads(data)
        except (OSError, ValueError):
            issues.append(
                {
                    "event_id": event["event_id"],
                    "code": "payload_unavailable_or_corrupt",
                }
            )
    for action_id in starts.keys() - completed:
        issues.append({"action_id": action_id, "code": "action_completion_unknown"})
    for execution_id in executions - finished_executions:
        issues.append(
            {"execution_id": execution_id, "code": "execution_completion_unknown"}
        )
    # Keep state-confirmed Candidates visible even when their journal evidence
    # is missing. Do not infer retention or removal from the configured policy.
    for candidate_id in state["candidates"]:
        artifact = artifacts.setdefault(
            candidate_id,
            {"artifact_id": candidate_id, "storage_status": "unknown"},
        )
        artifact["kind"] = "candidate_bundle"
    for artifact in artifacts.values():
        if artifact.get("storage_status") == "retention_failed":
            issues.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "code": "artifact_retention_failed",
                }
            )
        if artifact.get("kind") != "candidate_bundle":
            continue
        digest = artifact["artifact_id"].removeprefix("sha256:")
        if (
            not artifact["artifact_id"].startswith("sha256:")
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            issues.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "code": "invalid_artifact_identity",
                }
            )
            continue
        path = episode / "candidates" / digest
        artifact["path"] = path.relative_to(episode).as_posix()
        present = (
            path.is_dir() and not path.is_symlink() and not path.parent.is_symlink()
        )
        artifact["observed_storage"] = "present" if present else "missing"
        commits = [
            c
            for c in state["candidate_commits"]
            if c["candidate_id"] == artifact["artifact_id"]
        ]
        artifact["commit_event_ids"] = [
            f"{state['episode_id']}:event:{c['event_seq']}" for c in commits
        ]
        artifact["source_leases"] = sorted({c["source_lease"] for c in commits})
        if not present and artifact.get("storage_status") != "removed_by_policy":
            issues.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "code": "artifact_missing_without_confirmed_removal",
                }
            )
        if present and artifact.get("storage_status") == "removed_by_policy":
            issues.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "code": "artifact_present_after_recorded_removal",
                }
            )
    # The authoritative result remains the source of postmortem measurements.
    return {
        "schema_version": 1,
        "episode_id": state["episode_id"],
        "episode_status": state["status"],
        "environment": state["environment"],
        "recording": state["recording"],
        "scope": {
            "core_events": "recorded",
            "agent_control": "handler_and_available_transport_evidence",
            "agent_internal_tools": "adapter_observations_only",
            "workspace_edit_history": "adapter_observations_only",
            "inference_contents": "not_automatically_collected",
        },
        "events": events,
        "observed_through_event_id": events[-1]["event_id"] if events else None,
        "payloads": payloads,
        "artifacts": list(artifacts.values()),
        "candidate_commits": state["candidate_commits"],
        "incumbent_declarations": state["incumbent_declarations"],
        "usage": {
            "timing": state.get("timing_summary"),
            "resource_allocations": state.get("resource_allocations"),
            "agent_usage": state.get("agent_usage"),
            "inference_usage": state.get("inference_usage"),
        },
        "issues": issues,
        "postmortem": {
            "analysis_trajectory": state["analysis_trajectory"],
            "final_candidate_id": state["final_candidate_id"],
            "test_status": state["test_status"],
            "test_performance_score": state["test_performance_score"],
            "test_raw_metrics": state["test_raw_metrics"],
            "test_cost": state["test_cost"],
        },
    }
