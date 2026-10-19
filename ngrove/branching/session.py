"""Durable branch graph and operation ledger; no platform execution or Agent policy.

One SQLite transaction commits state and its event together. Monitor side effects
happen outside transactions and are fenced by operation tokens. Lost acknowledgements
remain uncertain until the deployment reconciles ownership; they are never replayed
as fresh allocations automatically.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from .contracts import (
    BranchError,
    RunnerCapabilities,
    SessionPolicy,
    components,
    detached,
    encoded,
    identifier,
    identity,
    number,
    vector,
)


_PREPARE = {"fork", "checkout", "merge"}
_TERMINAL = {"succeeded", "failed", "cancelled"}


class BranchSession:
    """Trusted service object. Never mount its database into an Agent sandbox."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path).resolve()
        self.clock = clock
        if not self.path.is_file():
            raise BranchError("session_not_found")

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        policy: SessionPolicy,
        capabilities: RunnerCapabilities,
        initial_state: dict,
        resources: dict,
        context_mode: str = "exact",
        clock: Callable[[], float] = time.time,
    ) -> BranchSession:
        path = Path(path).resolve()
        contract, runner = policy.manifest(), capabilities.manifest()
        state_components = components(initial_state, tuple(policy.required_components))
        cls._admit_state(runner, state_components, context_mode, "shared")
        cls._admit_resources(contract, resources)
        now = number(clock())
        node = {
            "parents": [],
            "components": state_components,
            "context_mode": context_mode,
            "segments": {},
            "records": [],
            "source": "initial_state",
        }
        if len(encoded(node)) > policy.max_payload_bytes:
            raise BranchError("snapshot_metadata_limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids replacing another session, including an empty DB.
        with path.open("xb"):
            pass
        path.chmod(0o600)
        with sqlite3.connect(str(path)) as db:
            db.executescript(
                "CREATE TABLE session (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT);"
                "CREATE TABLE snapshots (id TEXT PRIMARY KEY, body TEXT NOT NULL);"
                "CREATE TABLE events (seq INTEGER PRIMARY KEY, body TEXT NOT NULL);"
            )
            node_id = identity(node)
            db.execute("INSERT INTO snapshots VALUES (?, ?)", (node_id, encoded(node)))
            state = {
                "schema_version": 1,
                "session_id": uuid.uuid4().hex,
                "policy": contract,
                "runner": runner,
                "contract_sha256": identity({"policy": contract, "runner": runner}),
                "started_at": now,
                "last_accounted_at": now,
                "status": "open",
                "research_seconds": 0.0,
                "resource_seconds": {key: 0.0 for key in contract["capacity"]},
                "branches": {},
                "operations": {},
                "requests": {},
                "final_snapshot": None,
                "sequence": 0,
            }
            branch, op = cls._new_branch(
                state,
                node_id,
                node,
                resources,
                "fork",
                "shared",
                context_mode,
                requested_by="operator_bootstrap",
            )
            state["root_branch"] = branch["id"]
            cls._event(db, state, "SESSION_CREATED", operation_id=op["id"])
            db.execute("INSERT INTO session VALUES (1, ?)", (encoded(state),))
        return cls(path, clock=clock)

    @staticmethod
    def _admit_resources(policy: dict, requested: dict) -> None:
        requested = vector(requested)
        if any(
            k not in policy["capacity"] or v > policy["capacity"][k]
            for k, v in requested.items()
        ):
            raise BranchError("resource_request_exceeds_contract")

    @staticmethod
    def _admit_state(runner: dict, value: dict, mode: str, prefix_mode: str) -> None:
        if (
            mode not in runner["context_modes"]
            or prefix_mode not in runner["prefix_modes"]
        ):
            raise BranchError("unsupported_restore_mode")
        for key, ref in value.items():
            if ref["format"] not in runner["component_formats"].get(key, []):
                raise BranchError("unsupported_state_format")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            state = json.loads(
                db.execute("SELECT body FROM session WHERE id=1").fetchone()[0]
            )
            if state["schema_version"] != 1:
                raise BranchError("unsupported_session_schema")
            now = number(self.clock())
            if now < state["last_accounted_at"]:
                raise BranchError("clock_regressed")
            self._account(db, state, now)
            yield db, state
            db.execute("UPDATE session SET body=? WHERE id=1", (encoded(state),))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _event(db, state: dict, kind: str, **data: Any) -> None:
        state["sequence"] += 1
        value = {
            "event_id": f"{state['session_id']}:{state['sequence']}",
            "seq": state["sequence"],
            "timestamp": state["last_accounted_at"],
            "kind": kind,
            **data,
        }
        db.execute(
            "INSERT INTO events VALUES (?, ?)", (state["sequence"], encoded(value))
        )

    @staticmethod
    def _snapshot(db, node_id: str) -> dict:
        row = db.execute(
            "SELECT body FROM snapshots WHERE id=?", (identifier(node_id),)
        ).fetchone()
        if row is None:
            raise BranchError("unknown_snapshot")
        node = json.loads(row[0])
        if identity(node) != node_id:
            raise BranchError("snapshot_identity_mismatch")
        return node

    @staticmethod
    def _save_snapshot(db, state: dict, node: dict) -> str:
        if len(encoded(node)) > state["policy"]["max_payload_bytes"]:
            raise BranchError("snapshot_metadata_limit")
        node_id = identity(node)
        db.execute(
            "INSERT OR IGNORE INTO snapshots VALUES (?, ?)", (node_id, encoded(node))
        )
        return node_id

    @staticmethod
    def _operation(state: dict, branch: dict, kind: str, requested_by: str) -> dict:
        op_id = uuid.uuid4().hex
        op = {
            "id": op_id,
            "branch_id": branch["id"],
            "kind": kind,
            "requested_by": requested_by,
            "status": "queued",
            "token": None,
            "cancel_requested": False,
            "created_at": state["last_accounted_at"],
            "error": None,
            "receipt_sha256": None,
        }
        state["operations"][op_id] = op
        branch["pending_operation"] = op_id
        return op

    @classmethod
    def _new_branch(
        cls, state, node_id, node, resources, kind, prefix_mode, mode, requested_by
    ):
        if len(state["branches"]) >= state["policy"]["max_branches"]:
            raise BranchError("branch_limit")
        branch_id = uuid.uuid4().hex
        branch = {
            "id": branch_id,
            "head": node_id,
            "base": node_id,
            "prefix_mode": prefix_mode,
            "context_mode": mode,
            "status": "queued",
            "resources": vector(resources),
            "reserved": False,
            "allocation_id": None,
            "clock_running": False,
            "active_seconds": 0.0,
            "inherited_segments": dict(node["segments"]),
            "pending_operation": None,
            "episode_id": None,
        }
        state["branches"][branch_id] = branch
        op = cls._operation(state, branch, kind, requested_by)
        return branch, op

    @classmethod
    def _cancel_operation(cls, state, op):
        if op["status"] in _TERMINAL:
            return
        op["cancel_requested"] = True
        branch = state["branches"][op["branch_id"]]
        if op["status"] == "queued":
            op["status"] = "cancelled"
            branch["pending_operation"] = None
            if op["kind"] in _PREPARE:
                branch["status"] = "stopped"
        else:
            op["status"] = "needs_recovery"

    @classmethod
    def _close(cls, db, state, reason):
        if state["status"] == "sealed":
            return
        if state["status"] == "open":
            state["status"] = "closing"
            cls._event(db, state, "SESSION_CLOSING", reason=reason)
        for branch in state["branches"].values():
            op_id = branch["pending_operation"]
            if op_id:
                op = state["operations"][op_id]
                if op["kind"] != "stop":
                    cls._cancel_operation(state, op)
            if not branch["pending_operation"] and (
                branch["clock_running"] or branch["reserved"]
            ):
                cls._operation(state, branch, "stop", "environment_limit")

    @classmethod
    def _account(cls, db, state, now):
        delta = now - state["last_accounted_at"]
        state["last_accounted_at"] = now
        for branch in state["branches"].values():
            if branch["clock_running"]:
                branch["active_seconds"] += delta
                state["research_seconds"] += delta
            if branch["allocation_id"] is not None:
                for key, amount in branch["resources"].items():
                    state["resource_seconds"][key] += delta * amount
        policy = state["policy"]
        exceeded = (
            now - state["started_at"] >= policy["wall_seconds"]
            or state["research_seconds"] >= policy["research_seconds"]
            or any(
                state["resource_seconds"][k] >= limit
                for k, limit in policy["resource_seconds"].items()
            )
        )
        if exceeded:
            cls._close(db, state, "session_budget_exhausted")
        for branch in state["branches"].values():
            if cls._path_seconds(branch) >= policy["branch_seconds"]:
                pending = branch["pending_operation"]
                if pending and state["operations"][pending]["kind"] != "stop":
                    cls._cancel_operation(state, state["operations"][pending])
                if not branch["pending_operation"] and branch["clock_running"]:
                    cls._operation(state, branch, "stop", "branch_budget_exhausted")

    @staticmethod
    def _path_seconds(branch):
        return sum(branch["inherited_segments"].values()) + branch["active_seconds"]

    @staticmethod
    def _public_operation(op):
        return {
            key: detached(op[key])
            for key in (
                "id",
                "branch_id",
                "kind",
                "status",
                "cancel_requested",
                "error",
            )
        }

    @classmethod
    def _public_branch(cls, branch):
        return {
            **{
                key: detached(branch[key])
                for key in (
                    "id",
                    "head",
                    "status",
                    "prefix_mode",
                    "context_mode",
                    "resources",
                    "active_seconds",
                    "pending_operation",
                )
            },
            "ancestry_research_seconds": cls._path_seconds(branch),
            "research_active": branch["clock_running"],
        }

    def submit(self, actor_branch: str, request: dict) -> dict:
        """Authenticated ingress binds actor_branch; the Agent cannot supply it.

        Retries with the same actor/request ID return the same operation. Request
        IDs must be retained after transport timeouts. No private locators, receipts,
        snapshots or episode journals are accepted from the Agent.
        """
        if not isinstance(request, dict) or set(request) != {"id", "op", "args"}:
            raise BranchError("invalid_request")
        identifier(request["id"])
        identifier(request["op"])
        if not isinstance(request["args"], dict):
            raise BranchError("invalid_arguments")
        with self._transaction() as (db, state):
            if len(encoded(request)) > state["policy"]["max_payload_bytes"]:
                raise BranchError("request_too_large")
            key = identifier(actor_branch) + ":" + request["id"]
            fingerprint = identity(request)
            previous = state["requests"].get(key)
            if previous:
                if previous["sha256"] != fingerprint:
                    raise BranchError("request_id_conflict")
                return self._public_operation(
                    state["operations"][previous["operation_id"]]
                )
            if state["status"] != "open":
                raise BranchError("session_closed")
            actor = state["branches"].get(actor_branch)
            if (
                not actor
                or not actor["clock_running"]
                or self._path_seconds(actor) >= state["policy"]["branch_seconds"]
            ):
                raise BranchError("actor_not_running")
            if len(state["requests"]) >= state["policy"]["max_requests"]:
                raise BranchError("request_limit")
            kind, args = request["op"], detached(request["args"])
            if kind in _PREPARE:
                allowed = {"resources", "prefix_mode", "context_mode"}
                allowed |= {"sources", "selection"} if kind == "merge" else {"snapshot"}
                if set(args) != allowed:
                    raise BranchError("invalid_arguments")
                if kind == "merge":
                    node_id, node = self._merge(
                        db, state, args["sources"], args["selection"]
                    )
                else:
                    node_id = args["snapshot"]
                    node = self._snapshot(db, node_id)
                mode = args["context_mode"]
                if node["context_mode"] == "replay" and mode == "exact":
                    raise BranchError("cannot_upgrade_context_fidelity")
                self._admit_state(
                    state["runner"], node["components"], mode, args["prefix_mode"]
                )
                self._admit_resources(state["policy"], args["resources"])
                if sum(node["segments"].values()) >= state["policy"]["branch_seconds"]:
                    raise BranchError("branch_budget_exhausted")
                branch, op = self._new_branch(
                    state,
                    node_id,
                    node,
                    args["resources"],
                    kind,
                    args["prefix_mode"],
                    mode,
                    actor_branch,
                )
            elif kind in {"checkpoint", "stop"}:
                if set(args) != {"branch"}:
                    raise BranchError("invalid_arguments")
                branch = state["branches"].get(identifier(args["branch"]))
                if not branch:
                    raise BranchError("unknown_branch")
                if branch["pending_operation"]:
                    if kind != "stop":
                        raise BranchError("branch_busy")
                    op = state["operations"][branch["pending_operation"]]
                    if op["kind"] != "stop":
                        self._cancel_operation(state, op)
                        if not branch["pending_operation"] and branch["clock_running"]:
                            op = self._operation(state, branch, "stop", actor_branch)
                else:
                    if branch["status"] not in {"running", "paused"}:
                        raise BranchError("invalid_branch_state")
                    op = self._operation(state, branch, kind, actor_branch)
            else:
                raise BranchError("unknown_operation")
            state["requests"][key] = {"sha256": fingerprint, "operation_id": op["id"]}
            self._event(
                db,
                state,
                "REQUEST_ACCEPTED",
                actor=actor_branch,
                request=request,
                operation_id=op["id"],
                branch_id=branch["id"],
            )
            return self._public_operation(op)

    @classmethod
    def _merge(cls, db, state, sources, selection):
        if (
            not isinstance(sources, list)
            or len(sources) < 2
            or any(not isinstance(source, str) for source in sources)
            or len(set(sources)) != len(sources)
            or not isinstance(selection, dict)
        ):
            raise BranchError("invalid_merge")
        nodes = {node_id: cls._snapshot(db, node_id) for node_id in sources}
        names = set().union(*(node["components"] for node in nodes.values()))
        if set(selection) != names:
            raise BranchError("merge_requires_explicit_component_selection")
        merged = {}
        for name, source in selection.items():
            if (
                not isinstance(source, str)
                or source not in nodes
                or name not in nodes[source]["components"]
            ):
                raise BranchError("invalid_merge_selection")
            merged[name] = nodes[source]["components"][name]
        segments = {}
        for node in nodes.values():
            for key, value in node["segments"].items():
                segments[key] = max(segments.get(key, 0), value)
        node = {
            "parents": sources,
            "components": merged,
            "context_mode": nodes[selection["agent_context"]]["context_mode"],
            "segments": segments,
            "records": [],
            "source": "target_merge",
            "selection": selection,
        }
        return cls._save_snapshot(db, state, node), node

    def claim(self, monitor_id: str) -> Optional[dict]:
        """Atomically reserve capacity and claim one request, without doing I/O."""
        identifier(monitor_id)
        with self._transaction() as (db, state):
            used = {key: 0.0 for key in state["policy"]["capacity"]}
            for branch in state["branches"].values():
                if branch["reserved"]:
                    for key, amount in branch["resources"].items():
                        used[key] += amount
            for op in state["operations"].values():
                if op["status"] != "queued":
                    continue
                branch = state["branches"][op["branch_id"]]
                if op["kind"] in _PREPARE:
                    if state["status"] != "open" or any(
                        used[k] + v > state["policy"]["capacity"][k]
                        for k, v in branch["resources"].items()
                    ):
                        continue
                    branch["reserved"] = True
                    branch["status"] = "preparing"
                op.update(
                    status="claimed", token=uuid.uuid4().hex, monitor_id=monitor_id
                )
                self._event(
                    db,
                    state,
                    "OPERATION_CLAIMED",
                    operation_id=op["id"],
                    monitor_id=monitor_id,
                )
                return self._work(db, state, op)
            return None

    @classmethod
    def _work(cls, db, state, op):
        branch = state["branches"][op["branch_id"]]
        node = cls._snapshot(db, branch["head"])
        return {
            "operation": detached(op),
            "branch": detached(branch),
            "snapshot": node,
            "runner": detached(state["runner"]),
            "limits": cls._limits(state, branch),
        }

    @staticmethod
    def _limits(state, branch):
        p = state["policy"]
        return {
            "wall_deadline": state["started_at"] + p["wall_seconds"],
            "research_seconds_remaining": max(
                0,
                min(
                    p["branch_seconds"] - BranchSession._path_seconds(branch),
                    p["research_seconds"] - state["research_seconds"],
                ),
            ),
            "resource_seconds_remaining": {
                k: max(0, limit - state["resource_seconds"][k])
                for k, limit in p["resource_seconds"].items()
            },
        }

    @staticmethod
    def _owned(state, op_id, token):
        op = state["operations"].get(op_id)
        if not op or not token or token != op["token"]:
            raise BranchError("stale_operation_token")
        return op, state["branches"][op["branch_id"]]

    def allocated(self, op_id: str, token: str, *, allocation_id: str, resources: dict):
        """Trusted acknowledgement: billing starts on actual allocation, before READY."""
        identifier(allocation_id)
        resources = vector(resources)
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            if op["kind"] not in _PREPARE or op["status"] != "claimed":
                raise BranchError("operation_not_preparing")
            if resources != branch["resources"]:
                raise BranchError("allocation_mismatch")
            if branch["allocation_id"]:
                if branch["allocation_id"] != allocation_id:
                    raise BranchError("allocation_conflict")
                return
            if any(
                prior.get("allocation_id") == allocation_id
                for prior in state["operations"].values()
            ):
                raise BranchError("allocation_already_bound")
            branch["allocation_id"] = allocation_id
            op["allocation_id"] = allocation_id
            self._event(
                db,
                state,
                "RESOURCES_ALLOCATED",
                operation_id=op_id,
                allocation_id=allocation_id,
            )

    def ready(self, op_id: str, token: str, receipt: dict) -> dict:
        """Arm accounting BEFORE the adapter releases execution; never wait for Agent ACK.

        The adapter has restored all components, verified identities and isolation,
        and still holds the execution gate. A committed permit authorizes release.
        """
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            digest = identity(receipt)
            if op["status"] == "succeeded":
                if op["receipt_sha256"] != digest:
                    raise BranchError("receipt_conflict")
                # This is an acknowledgement retry, NOT permission to launch twice.
                return {"already_applied": True, "branch_id": branch["id"]}
            if (
                op["kind"] not in _PREPARE
                or op["status"] != "claimed"
                or op["cancel_requested"]
                or state["status"] != "open"
            ):
                raise BranchError("ready_not_admissible")
            expected = {
                "snapshot",
                "allocation_id",
                "episode_id",
                "execution_held",
                "isolated_writes",
                "context_mode",
                "prefix_mode",
                "evidence_id",
            }
            if not isinstance(receipt, dict) or set(receipt) != expected:
                raise BranchError("invalid_ready_receipt")
            if (
                receipt["snapshot"] != branch["head"]
                or receipt["allocation_id"] != branch["allocation_id"]
                or branch["allocation_id"] is None
                or receipt["execution_held"] is not True
                or receipt["isolated_writes"] is not True
                or receipt["context_mode"] != branch["context_mode"]
                or receipt["prefix_mode"] != branch["prefix_mode"]
            ):
                raise BranchError("restore_receipt_mismatch")
            identifier(receipt["episode_id"])
            identifier(receipt["evidence_id"])
            if any(
                b["episode_id"] == receipt["episode_id"]
                for b in state["branches"].values()
            ):
                raise BranchError("episode_already_bound")
            limits = self._limits(state, branch)
            if limits["research_seconds_remaining"] <= 0:
                raise BranchError("branch_budget_exhausted")
            branch.update(
                status="running",
                clock_running=True,
                episode_id=receipt["episode_id"],
                pending_operation=None,
            )
            op.update(status="succeeded", receipt_sha256=digest)
            self._event(db, state, "BRANCH_READY", operation_id=op_id, receipt=receipt)
            return {
                "already_applied": False,
                "branch_id": branch["id"],
                "limits": limits,
            }

    def quiesced(self, op_id: str, token: str, *, evidence_id: str):
        """The deployment confirms ALL branch research execution is stopped."""
        identifier(evidence_id)
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            if op["status"] not in {"claimed", "needs_recovery"}:
                raise BranchError("operation_not_owned")
            if branch["clock_running"]:
                branch["clock_running"] = False
                self._event(
                    db,
                    state,
                    "BRANCH_QUIESCED",
                    operation_id=op_id,
                    evidence_id=evidence_id,
                )
            branch["status"] = (
                "checkpointing" if op["kind"] == "checkpoint" else "stopping"
            )
            op["quiescence_evidence"] = evidence_id

    def released(self, op_id: str, token: str, *, evidence_id: str):
        """Release capacity ONLY after confirmed allocation/workload cleanup."""
        identifier(evidence_id)
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            if (
                op["status"] not in {"claimed", "needs_recovery"}
                or not op.get("quiescence_evidence")
                or branch["clock_running"]
            ):
                raise BranchError("cleanup_requires_quiescence")
            branch.update(reserved=False, allocation_id=None)
            op["cleanup_evidence"] = evidence_id
            self._event(
                db,
                state,
                "RESOURCES_RELEASED",
                operation_id=op_id,
                evidence_id=evidence_id,
            )

    def complete(
        self,
        op_id: str,
        token: str,
        *,
        state_components=None,
        records=(),
        execution_held: bool = False,
    ):
        """Checkpoint a quiescent branch and arm its continuation, or finish a stop.

        Successful checkpoints retain their allocation. Capture time is excluded
        from active research, but still charged to allocation. Commit BEFORE the
        runner releases the execution gate. A repeated receipt never grants another
        release. Records must be explicitly research-visible runner deltas.
        """
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            receipt = {
                "components": state_components,
                "records": list(records),
                "execution_held": execution_held,
            }
            digest = identity(receipt)
            if op["status"] == "succeeded":
                if digest != op["receipt_sha256"]:
                    raise BranchError("receipt_conflict")
                return {"already_applied": True, **self._public_branch(branch)}
            if (
                op["status"] != "claimed"
                or op["cancel_requested"]
                or not op.get("quiescence_evidence")
                or branch["clock_running"]
            ):
                raise BranchError("completion_not_admissible")
            if op["kind"] == "checkpoint":
                if (
                    execution_held is not True
                    or branch["allocation_id"] is None
                    or not branch["reserved"]
                    or state["status"] != "open"
                    or self._limits(state, branch)["research_seconds_remaining"] <= 0
                ):
                    raise BranchError("checkpoint_continuation_not_admissible")
                value = components(
                    state_components, tuple(state["policy"]["required_components"])
                )
                self._admit_state(
                    state["runner"], value, branch["context_mode"], "shared"
                )
                segments = dict(branch["inherited_segments"])
                segments[branch["id"]] = branch["active_seconds"]
                node = {
                    "parents": [branch["head"]],
                    "components": value,
                    "context_mode": branch["context_mode"],
                    "segments": segments,
                    "records": detached(list(records)),
                    "source": branch["id"],
                }
                branch["head"] = self._save_snapshot(db, state, node)
                branch.update(status="running", clock_running=True)
            elif op["kind"] == "stop" and state_components is None and not records:
                if not op.get("cleanup_evidence") or branch["reserved"]:
                    raise BranchError("stop_requires_cleanup")
                branch["status"] = "stopped"
            else:
                raise BranchError("invalid_completion")
            branch["pending_operation"] = None
            op.update(status="succeeded", receipt_sha256=digest)
            self._event(
                db,
                state,
                "OPERATION_COMPLETED",
                operation_id=op_id,
                snapshot=branch["head"],
            )
            return {
                "already_applied": False,
                **self._public_branch(branch),
                "limits": self._limits(state, branch),
            }

    def uncertain(self, op_id: str, token: str, *, detail: str):
        """Private diagnostic only. Keep reservations and clocks until proof of cleanup."""
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            if op["status"] in _TERMINAL:
                # A release/launch can fail after READY was durably acknowledged.
                if (
                    op["kind"] not in _PREPARE | {"checkpoint"}
                    or not branch["clock_running"]
                ):
                    raise BranchError("operation_terminal")
                if branch["pending_operation"]:
                    raise BranchError("branch_busy")
                branch["pending_operation"] = op_id
            op.update(status="needs_recovery", error="deployment_recovery_required")
            self._event(
                db,
                state,
                "OPERATION_UNCERTAIN",
                operation_id=op_id,
                detail=str(detail)[:8192],
            )

    def recover(self, op_id: str, *, monitor_id: str, fencing_evidence: str) -> dict:
        """Deployment must fence the previous executor BEFORE rotating its token.

        No automatic claim expiry: an elapsed heartbeat is not proof of fencing.
        Recovery is cleanup-only, never permission to relaunch an uncertain operation.
        """
        identifier(monitor_id)
        identifier(fencing_evidence)
        with self._transaction() as (db, state):
            op = state["operations"].get(op_id)
            if not op or op["status"] not in {"claimed", "needs_recovery"}:
                raise BranchError("operation_not_recoverable")
            op.update(
                status="needs_recovery", token=uuid.uuid4().hex, monitor_id=monitor_id
            )
            self._event(
                db,
                state,
                "RECOVERY_CLAIMED",
                operation_id=op_id,
                evidence_id=fencing_evidence,
            )
            return self._work(db, state, op)

    def fail_clean(self, op_id: str, token: str, *, code: str = "execution_failed"):
        identifier(code)
        with self._transaction() as (db, state):
            op, branch = self._owned(state, op_id, token)
            if op["status"] in _TERMINAL:
                if op["status"] == "failed" and op["error"] == code:
                    return
                raise BranchError("operation_terminal")
            if (
                not op.get("cleanup_evidence")
                or branch["reserved"]
                or branch["clock_running"]
            ):
                raise BranchError("failure_requires_cleanup")
            op.update(status="failed", error=code)
            branch.update(status="failed", pending_operation=None)
            self._event(db, state, "OPERATION_FAILED", operation_id=op_id, code=code)

    def request_seal(self, actor_branch: str, *, final_snapshot: str):
        identifier(actor_branch)
        identifier(final_snapshot)
        with self._transaction() as (db, state):
            if state.get("seal_requested_by") == actor_branch:
                if state["final_snapshot"] != final_snapshot:
                    raise BranchError("seal_intent_conflict")
                return
            if state["status"] != "open" or not state["branches"].get(
                actor_branch, {}
            ).get("clock_running"):
                raise BranchError("actor_not_running")
            self._snapshot(db, final_snapshot)
            state["final_snapshot"] = final_snapshot
            state["seal_requested_by"] = actor_branch
            self._close(db, state, "target_finished")

    def confirm_sealed(self, *, control_revoked: bool, evidence_id: str):
        identifier(evidence_id)
        with self._transaction() as (db, state):
            if (
                state["status"] != "closing"
                or control_revoked is not True
                or any(
                    b["clock_running"] or b["reserved"] or b["pending_operation"]
                    for b in state["branches"].values()
                )
            ):
                raise BranchError("research_teardown_incomplete")
            state["status"] = "sealed"
            state["sealed_at"] = state["last_accounted_at"]
            self._event(
                db,
                state,
                "RESEARCH_SEALED",
                evidence_id=evidence_id,
                final_snapshot=state["final_snapshot"],
            )

    def status(self, *, private: bool = False) -> dict:
        """Also ticks budget supervision. Deployment must call regularly, independently
        of any LLM, and enforce Worker-local deadlines when this service is lost.
        """
        with self._transaction() as (_, state):
            if private:
                return detached(state)
            return {
                "session_id": state["session_id"],
                "status": state["status"],
                "root_branch": state["root_branch"],
                "contract_sha256": state["contract_sha256"],
                "research_seconds": state["research_seconds"],
                "wall_seconds": state.get("sealed_at", state["last_accounted_at"])
                - state["started_at"],
                "budget": {
                    key: detached(state["policy"][key])
                    for key in (
                        "capacity",
                        "wall_seconds",
                        "research_seconds",
                        "branch_seconds",
                        "resource_seconds",
                    )
                },
                "resource_accounting_basis": "trusted_monitor_acknowledgements",
                "restore_capabilities": {
                    key: detached(state["runner"][key])
                    for key in ("component_formats", "context_modes", "prefix_modes")
                },
                "resource_seconds": detached(state["resource_seconds"]),
                "branches": [
                    self._public_branch(b) for b in state["branches"].values()
                ],
                "operations": [
                    self._public_operation(o) for o in state["operations"].values()
                ],
            }

    def branch_status(self, branch_id: str) -> dict:
        """A bounded observation for event-driven transports and targeted polling."""
        with self._transaction() as (_, state):
            branch = state["branches"].get(identifier(branch_id))
            if branch is None:
                raise BranchError("unknown_branch")
            return self._public_branch(branch)

    def operation_status(self, operation_id: str) -> dict:
        with self._transaction() as (_, state):
            operation = state["operations"].get(identifier(operation_id))
            if operation is None:
                raise BranchError("unknown_operation")
            return self._public_operation(operation)

    @staticmethod
    def _node_view(node_id: str, node: dict, include_state: bool) -> dict:
        item = {"id": node_id, **node}
        if not include_state:
            item["components"] = {
                k: {"format": v["format"], "sha256": v["sha256"]}
                for k, v in node["components"].items()
            }
        return item

    def snapshot(self, snapshot: str, *, include_state: bool = False) -> dict:
        """Read one bounded node, with parent IDs for incremental DAG traversal."""
        with self._transaction() as (db, _):
            return self._node_view(
                snapshot, self._snapshot(db, snapshot), include_state
            )

    def history(
        self, snapshot: str, *, include_state: bool = False, max_nodes: int = 256
    ) -> dict:
        """Expand a DAG prefix once per immutable node; no fabricated total causal order.
        include_state is Operator-only because component locators are private.
        Large histories use snapshot() traversal instead of unbounded expansion.
        """
        if (
            isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or max_nodes < 1
        ):
            raise BranchError("invalid_limit")
        with self._transaction() as (db, state):
            ordered, visited, discovered = [], set(), set()
            total_bytes = 0
            stack = [(snapshot, False)]
            while stack:
                node_id, expanded = stack.pop()
                if node_id in visited:
                    continue
                node = self._snapshot(db, node_id)
                if node_id not in discovered:
                    discovered.add(node_id)
                    total_bytes += len(encoded(node))
                    if (
                        len(discovered) > max_nodes
                        or total_bytes > state["policy"]["max_payload_bytes"]
                    ):
                        raise BranchError("history_limit_use_snapshot")
                if expanded:
                    visited.add(node_id)
                    ordered.append(self._node_view(node_id, node, include_state))
                else:
                    stack.append((node_id, True))
                    stack.extend(
                        (parent, False) for parent in reversed(node["parents"])
                    )
            return {
                "snapshot": snapshot,
                "nodes": ordered,
                "scope": "runner_selected_research_evidence",
                "complete_agent_state": False,
            }

    def events(self) -> list:
        """Private Operator evidence. Never publish this journal to target Agents."""
        with sqlite3.connect(str(self.path)) as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM events ORDER BY seq")
            ]
