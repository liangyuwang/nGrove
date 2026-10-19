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
    selected = []
    for event in events:
        if not after_seq < event.get("seq", 0) <= through_seq:
            continue
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
    segment = {
        "episode_id": recording["episode_id"],
        "after_seq": after_seq,
        "through_seq": through_seq,
        "events": selected,
        "scope": "agent_exchange_only",
        "complete_agent_context": False,
    }
    return {"sha256": identity(segment), **segment}
