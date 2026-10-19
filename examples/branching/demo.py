"""Runnable branching exercise with real files and a synthetic JSON-state runner.

No model, cloud service or accelerator is used. The fixture demonstrates the
adapter seam; its exact JSON restoration is not a provider capability claim.
Run from the checkout: PYTHONPATH=. python examples/branching/demo.py
"""

import json
import tempfile
from pathlib import Path

from ngrove.branching import (
    BranchClient,
    BranchSession,
    RunnerCapabilities,
    SessionPolicy,
)
from ngrove.branching.artifacts import ArtifactStore
from ngrove.branching.control import ControlEndpoint
from ngrove.branching.monitor import Monitor


class FixtureRunner:
    def __init__(self, root, store):
        self.root, self.store = root, store
        self.contexts = {}

    def apply(self, work, session):
        op, branch = work["operation"], work["branch"]
        oid, token, bid = op["id"], op["token"], branch["id"]
        directory = self.root / bid
        if op["kind"] in {"fork", "checkout", "merge"}:
            session.allocated(
                oid,
                token,
                allocation_id="fixture-" + bid,
                resources=branch["resources"],
            )
            source = self.store
            if branch["prefix_mode"] == "copy":
                source = ArtifactStore(self.root / (bid + "-prefix"))
                for node in session.history(branch["head"], include_state=True)[
                    "nodes"
                ]:
                    for ref in node["components"].values():
                        self.store.copy_component(ref, source)
            source.restore_directory(
                work["snapshot"]["components"]["workspace"], directory
            )
            self.contexts[bid] = json.loads(
                source.read(work["snapshot"]["components"]["agent_context"])
            )
            permit = session.ready(
                oid,
                token,
                {
                    "snapshot": branch["head"],
                    "allocation_id": "fixture-" + bid,
                    "episode_id": "fixture-" + bid,
                    "execution_held": True,
                    "isolated_writes": True,
                    "context_mode": branch["context_mode"],
                    "prefix_mode": branch["prefix_mode"],
                    "evidence_id": "fixture-restored",
                },
            )
            assert not permit[
                "already_applied"
            ]  # A real adapter now releases execution once.
        elif op["kind"] == "checkpoint":
            session.quiesced(oid, token, evidence_id="fixture-quiescent")
            session.complete(
                oid,
                token,
                execution_held=True,
                state_components={
                    "workspace": self.store.capture_directory(directory),
                    "agent_context": self.store.put_json(
                        self.contexts[bid], format="fixture-context-v1"
                    ),
                },
                records=[
                    {"kind": "fixture_observation", "step": self.contexts[bid]["step"]}
                ],
            )
        elif op["kind"] == "stop":
            session.quiesced(oid, token, evidence_id="fixture-stopped")
            session.released(oid, token, evidence_id="fixture-released")
            session.complete(oid, token)


def run(root):
    workspace = root / "initial"
    workspace.mkdir()
    (workspace / "research.txt").write_text("initial hypothesis\n")
    store = ArtifactStore(root / "objects")
    initial = {
        "workspace": store.capture_directory(workspace),
        "agent_context": store.put_json(
            {"step": 0, "messages": []}, format="fixture-context-v1"
        ),
    }
    session = BranchSession.create(
        root / "session.db",
        policy=SessionPolicy(
            {"slot": 4}, wall_seconds=300, research_seconds=600, branch_seconds=300
        ),
        capabilities=RunnerCapabilities(
            "json-fixture",
            "fixture-only",
            {"workspace": ("directory-v1",), "agent_context": ("fixture-context-v1",)},
            context_modes=("exact",),
            prefix_modes=("shared", "copy"),
        ),
        initial_state=initial,
        resources={"slot": 1},
    )
    runner = FixtureRunner(root, store)
    monitor = Monitor(session, "fixture-monitor", runner)
    monitor.run_once()
    root_branch = session.status()["root_branch"]
    target = BranchClient(ControlEndpoint(session, root_branch))
    runner.contexts[root_branch]["step"] = 1
    (root / root_branch / "research.txt").write_text("shared experiment\n")
    target.checkpoint(root_branch)
    monitor.run_once()
    common = session.status()["branches"][0]["head"]
    branches = []
    for mode in ("shared", "copy"):
        branches.append(
            target.fork(
                common, resources={"slot": 1}, context_mode="exact", prefix_mode=mode
            )["branch_id"]
        )
        monitor.run_once()
    nodes = []
    for index, branch in enumerate(branches):
        runner.contexts[branch]["step"] = index + 2
        (root / branch / "research.txt").write_text(f"direction {index}\n")
        target.checkpoint(branch)
        monitor.run_once()
        nodes.append(
            next(b["head"] for b in session.status()["branches"] if b["id"] == branch)
        )
    merged = target.merge(
        nodes,
        {"agent_context": nodes[0], "workspace": nodes[1]},
        resources={"slot": 1},
        context_mode="exact",
    )
    monitor.run_once()
    head = next(
        b["head"]
        for b in session.status()["branches"]
        if b["id"] == merged["branch_id"]
    )
    history = target.history(head)
    assert len({n["id"] for n in history["nodes"]}) == len(history["nodes"])
    assert (root / merged["branch_id"] / "research.txt").read_text() == "direction 1\n"
    target.finish(head)
    while monitor.run_once():
        pass
    session.confirm_sealed(
        control_revoked=True, evidence_id="fixture-all-controls-revoked"
    )
    return {
        "status": session.status()["status"],
        "branches": len(session.status()["branches"]),
        "history_nodes": len(history["nodes"]),
        "shared_prefix_occurrences": sum(n["id"] == common for n in history["nodes"]),
        "runner": "synthetic JSON fixture; no model/provider certification",
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ngrove-branches-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
