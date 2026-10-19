"""Explicit research-visible projection of existing episode recording evidence."""

from __future__ import annotations

from pathlib import Path

from envcore.recording import read_recording

from .contracts import BranchError, detached, identity


_EXCHANGES = {"ACTION_STARTED", "ACTION_RETURNED", "CONTROL_RESPONSE"}


def episode_segment(episode: Path, *, after_seq: int = 0, through_seq: int) -> dict:
    """Freeze a selected prefix/delta after the runner has quiesced the episode.

    Only explicitly Agent-visible exchanges are exported. Private events, usage,
    Operator diagnostics, observations and all postmortem measurements are excluded.
    A missing event in the selected interval prevents publication. Completion
    issues are derived from public exchanges as of through_seq, not private issues
    or later evidence in the source recording.
    This is history, not a restorable context export. It makes no consumption claim.
    """
    if (
        isinstance(after_seq, bool)
        or isinstance(through_seq, bool)
        or not isinstance(after_seq, int)
        or not isinstance(through_seq, int)
        or after_seq < 0
        or through_seq <= after_seq
    ):
        raise BranchError("invalid_recording_interval")
    recording = read_recording(Path(episode))
    events = recording["events"]
    if not any(e.get("seq") == through_seq for e in events):
        raise BranchError("recording_boundary_missing")
    interval = [e for e in events if after_seq < e.get("seq", 0) <= through_seq]
    # read_recording validates increasing sequence numbers and skips corrupt
    # entries. Count the entire interval before filtering private event types so
    # omissions cannot be mistaken for intentional visibility filtering.
    if len(interval) != through_seq - after_seq:
        raise BranchError("recording_interval_incomplete")
    returned_actions = {
        e["action_id"]
        for e in interval
        if e.get("phase") == "research"
        and e.get("visibility") == "agent_exchange"
        and e.get("type") == "ACTION_RETURNED"
        and isinstance(e.get("action_id"), str)
    }
    selected = []
    issues = []
    for event in interval:
        if (
            event.get("phase") != "research"
            or event.get("type") not in _EXCHANGES
            or event.get("visibility") != "agent_exchange"
        ):
            continue
        ref = event.get("payload", {})
        payload = recording["payloads"].get(ref.get("sha256"))
        item = {
            "event_id": event["event_id"],
            "seq": event["seq"],
            "type": event["type"],
            "payload_status": ref.get("status", "unknown"),
            "consumption": "unknown",
        }
        if payload is not None and identity(payload) == ref.get("sha256"):
            item["payload"] = detached(payload)
        elif ref.get("status") == "saved":
            item["payload_status"] = "unavailable_or_corrupt"
        selected.append(item)
        if (
            event["type"] == "ACTION_STARTED"
            and event.get("action_id") not in returned_actions
        ):
            issues.append(
                {"code": "action_completion_unknown", "event_id": event["event_id"]}
            )
    segment = {
        "episode_id": recording["episode_id"],
        "after_seq": after_seq,
        "through_seq": through_seq,
        "events": selected,
        "issues": issues,
        "scope": "agent_exchange_only",
        "complete_agent_context": False,
    }
    return {"sha256": identity(segment), **segment}
