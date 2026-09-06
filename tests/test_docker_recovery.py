import copy
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.docker_recovery import DockerRecovery, OWNER_LABEL, RESOURCE_LABEL
from autotrain_bench.worker_protocol import WorkerFailure


class FakeDocker:
    def __init__(self):
        self.backend = "fixture-daemon"
        self.objects = {}
        self.calls = []
        self.before_create = None
        self.after_create = None
        self.after_remove = None

    def run(self, args, **kwargs):
        args = list(args)
        self.calls.append(args)
        if args[:1] == ["info"]:
            return subprocess.CompletedProcess(args, 0, self.backend, "")
        if args[:1] == ["start"]:
            return subprocess.CompletedProcess(args, 0, "workload stdout", "")
        if args[:1] == ["wait"]:
            return subprocess.CompletedProcess(args, 0, "0", "")
        creation = DockerRecovery._creation(args)
        if creation:
            if self.before_create:
                self.before_create(args)
            kind, name, _ = creation
            labels = dict(
                args[i + 1].split("=", 1)
                for i, value in enumerate(args)
                if value == "--label"
            )
            identity = uuid.uuid4().hex * 2
            obj = {
                "Id": identity,
                "Name": "/" + name if kind == "container" else name,
                "CreatedAt": "2026-09-06T12:00:00Z",
                "Labels": labels,
                "Config": {"Labels": labels},
            }
            self.objects[(kind, name)] = obj
            if self.after_create:
                self.after_create(args)
            if "--rm" in args:
                del self.objects[(kind, name)]
            return subprocess.CompletedProcess(args, 0, identity, "")
        if len(args) > 1 and args[1] == "inspect":
            obj = self.objects.get((args[0], args[2]))
            if obj is None:
                obj = next(
                    (
                        value
                        for (kind, _), value in self.objects.items()
                        if kind == args[0] and value["Id"] == args[2]
                    ),
                    None,
                )
            return subprocess.CompletedProcess(
                args, 0 if obj else 1, json.dumps([obj]) if obj else "", ""
            )
        if len(args) > 1 and args[1] == "ls":
            names = [name for kind, name in self.objects if kind == args[0]]
            if "--no-trunc" in args:
                names = [
                    self.objects[(args[0], name)]["Id"] + "\t" + name for name in names
                ]
            return subprocess.CompletedProcess(args, 0, "\n".join(names), "")
        if args[:1] == ["rm"] or (len(args) > 1 and args[1] == "rm"):
            kind = "container" if args[0] == "rm" else args[0]
            for key, obj in list(self.objects.items()):
                if key[0] == kind and args[-1] in {key[1], obj["Id"]}:
                    del self.objects[key]
            if self.after_remove:
                self.after_remove(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


@unittest.skipUnless(os.name == "posix", "Worker recovery uses local POSIX locks")
class DockerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode"
        self.episode.mkdir()
        self.result = self.episode / "result.json"
        self.result.write_text('{"TEST":"already_saved"}')
        self.backend = FakeDocker()
        self.journal = DockerRecovery(self.root / "worker", episode=self.episode)
        self.addCleanup(self.journal.close)
        self.journal.bind_backend(self.backend.run)

    def create(self, kind="container", name="candidate"):
        args = (
            ["run", "--detach", "--name", name, "trusted-image"]
            if kind == "container"
            else [kind, "create", name]
        )
        return self.journal.execute(args, self.backend.run)

    def reopen(self):
        self.journal.close()
        journal = DockerRecovery(self.root / "worker")
        self.addCleanup(journal.close)
        return journal

    def test_write_ahead_ownership_and_cleanup_preserve_episode_results(self):
        def verify_intent(args):
            state = json.loads((self.root / "worker/resources.json").read_text())
            kind, name, _ = DockerRecovery._creation(args)
            record = state["resources"][kind + ":" + name]
            self.assertEqual(record["state"], "intent")
            self.assertIn(OWNER_LABEL + "=" + state["owner"], args)
            self.assertIn(RESOURCE_LABEL + "=" + record["token"], args)

        self.backend.before_create = verify_intent
        for kind, name in (
            ("container", "agent"),
            ("container", "candidate"),
            ("container", "atb-prediction-holder-1"),
            ("network", "inference"),
            ("volume", "predictions"),
        ):
            self.create(kind, name)
        private = self.journal.private_directory("hidden-")
        (private / "fixture-targets").write_bytes(b"private fixture")
        reopened = self.reopen()
        report = reopened.recover(self.backend.run)
        self.assertTrue(report["cleanup_complete"])
        self.assertFalse(report["resumed"])
        self.assertEqual(self.backend.objects, {})
        self.assertFalse(private.exists())
        self.assertEqual(self.result.read_text(), '{"TEST":"already_saved"}')
        calls = len(self.backend.calls)
        self.assertTrue(reopened.recover(self.backend.run)["cleanup_complete"])
        self.assertFalse(
            any(
                call[:1] == ["rm"] or call[:2] == ["volume", "rm"]
                for call in self.backend.calls[calls:]
            )
        )
        with self.assertRaisesRegex(WorkerFailure, "cannot launch"):
            reopened.execute(["run", "--name", "restart", "image"], self.backend.run)

    def test_live_owner_blocks_recovery_without_docker_mutations(self):
        calls = list(self.backend.calls)
        with self.assertRaisesRegex(WorkerFailure, "still holds ownership"):
            DockerRecovery(self.root / "worker")
        self.assertEqual(calls, self.backend.calls)

    def test_runtime_child_retains_lock_after_owner_exit(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; print('ready',flush=True); time.sleep(60)",
            ],
            pass_fds=(self.journal.lock_fd,),
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            self.journal.close()
            with self.assertRaisesRegex(WorkerFailure, "still holds ownership"):
                DockerRecovery(self.root / "worker")
        finally:
            process.terminate()
            process.wait(timeout=5)
            process.stdout.close()
        recovered = self.reopen()
        self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])

    def test_creation_without_a_response_is_reconciled_by_verified_identity(self):
        def interrupted(args):
            raise WorkerFailure("Worker died before creation acknowledgement")

        self.backend.after_create = interrupted
        with self.assertRaises(WorkerFailure):
            self.create()
        self.assertEqual(
            self.journal.state["resources"]["container:candidate"]["state"], "intent"
        )
        recovered = self.reopen()
        self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])
        self.assertFalse(self.backend.objects)

    def test_unknown_absent_creation_keeps_holder_and_private_storage(self):
        self.create(name="atb-prediction-holder-1")
        private = self.journal.private_directory("hidden-")
        attempted = []

        def interrupted(args):
            attempted.append(args)
            raise WorkerFailure("creation request outcome unknown")

        self.backend.before_create = interrupted
        with self.assertRaises(WorkerFailure):
            self.create()
        recovered = self.reopen()
        report = recovered.recover(self.backend.run)
        self.assertFalse(report["cleanup_complete"])
        self.assertIn("unknown", report["failures"][0]["reason"])
        self.assertTrue(private.exists())
        self.assertIn(("container", "atb-prediction-holder-1"), self.backend.objects)
        # A delayed daemon creation becomes observable on a later retry.
        self.backend.before_create = None
        self.backend.run(attempted[0])
        self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])

    def test_interrupted_removal_can_be_repeated_after_restart(self):
        self.create()

        def interrupted(args):
            raise WorkerFailure("delete acknowledgement lost")

        self.backend.after_remove = interrupted
        with self.assertRaises(WorkerFailure):
            self.journal.remove("container", "candidate", self.backend.run)
        self.assertFalse(self.backend.objects)
        self.backend.after_remove = None
        recovered = self.reopen()
        self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])

    def test_foreign_replacement_is_not_deleted_and_blocks_storage_cleanup(self):
        self.create()
        self.create("volume", "predictions")
        self.backend.objects[("container", "candidate")]["Config"]["Labels"] = {}
        recovered = self.reopen()
        report = recovered.recover(self.backend.run)
        self.assertFalse(report["cleanup_complete"])
        self.assertIn("unverified", report["failures"][0]["reason"])
        self.assertEqual(len(self.backend.objects), 2)
        self.assertFalse(any(call[0] == "rm" for call in self.backend.calls))

    def test_renamed_container_cannot_be_mistaken_for_absent(self):
        self.create()
        actual = self.backend.objects.pop(("container", "candidate"))
        actual["Name"] = "/renamed"
        self.backend.objects[("container", "renamed")] = actual
        report = self.reopen().recover(self.backend.run)
        self.assertFalse(report["cleanup_complete"])
        self.assertIn(("container", "renamed"), self.backend.objects)

    def test_copied_journal_cannot_bypass_live_owner_lock(self):
        self.create()
        shutil.copytree(self.root / "worker", self.root / "copied")
        with self.assertRaisesRegex(WorkerFailure, "damaged"):
            DockerRecovery(self.root / "copied")
        self.assertEqual(len(self.backend.objects), 1)

    def test_participant_arguments_are_preserved_when_splitting_create_start(self):
        args = [
            "run",
            "--name",
            "agent",
            "--rm",
            "--env",
            "X=1",
            "image",
            "agent",
            "--rm",
            "--detach",
        ]
        create, detached, auto_remove = DockerRecovery._container_create_args(args)
        self.assertFalse(detached)
        self.assertTrue(auto_remove)
        self.assertEqual(
            create,
            [
                "create",
                "--name",
                "agent",
                "--env",
                "X=1",
                "image",
                "agent",
                "--rm",
                "--detach",
            ],
        )

    def test_wrong_daemon_and_damaged_journal_fail_before_deletion(self):
        self.create()
        self.backend.backend = "different-daemon"
        recovered = self.reopen()
        with self.assertRaisesRegex(WorkerFailure, "daemon identity differs"):
            recovered.recover(self.backend.run)
        self.assertEqual(len(self.backend.objects), 1)
        recovered.close()
        (self.root / "worker/resources.json").write_text("{truncated")
        with self.assertRaisesRegex(WorkerFailure, "damaged"):
            DockerRecovery(self.root / "worker")
        self.assertFalse(any(call[0] == "rm" for call in self.backend.calls))

    def test_journal_failure_prevents_runtime_creation(self):
        with patch.object(
            self.journal, "_save", side_effect=WorkerFailure("disk full")
        ):
            with self.assertRaisesRegex(WorkerFailure, "disk full"):
                self.create()
        self.assertFalse(self.backend.objects)
        self.assertFalse(
            any(DockerRecovery._creation(call) for call in self.backend.calls)
        )

    def test_capacity_rejection_does_not_block_existing_workload_cleanup(self):
        self.create(name="agent")
        path = self.root / "worker/resources.json"
        saved = json.loads(path.read_text())
        with patch(
            "autotrain_bench.docker_recovery.MAX_JOURNAL_BYTES",
            path.stat().st_size + 10,
        ):
            with self.assertRaisesRegex(WorkerFailure, "capacity reached"):
                self.create(name="training")
            self.assertEqual(self.journal.state, saved)
            self.assertEqual(json.loads(path.read_text()), saved)
            self.journal.remove("container", "agent", self.backend.run)
        self.assertFalse(self.backend.objects)
        self.assertFalse(
            any(
                DockerRecovery._creation(call) == ("container", "training", 1)
                for call in self.backend.calls
            )
        )
        self.assertTrue(self.reopen().recover(self.backend.run)["cleanup_complete"])

    def test_capacity_for_only_an_intent_rejects_every_kind_before_creation(self):
        saved = copy.deepcopy(self.journal.state)
        for kind in ("container", "network", "volume"):
            with self.subTest(kind=kind):
                proposed = copy.deepcopy(saved)
                proposed["resources"][kind + ":candidate"] = {
                    "kind": kind,
                    "name": "candidate",
                    "token": "a" * 32,
                    "state": "intent",
                    "identity": None,
                }
                limit = len((json.dumps(proposed, sort_keys=True) + "\n").encode())
                with patch("autotrain_bench.docker_recovery.MAX_JOURNAL_BYTES", limit):
                    with self.assertRaisesRegex(WorkerFailure, "capacity reached"):
                        self.create(kind)
                self.assertEqual(self.journal.state, saved)
                self.assertEqual(
                    json.loads((self.root / "worker/resources.json").read_text()), saved
                )
                self.assertFalse(self.backend.objects)
                self.assertFalse(
                    any(DockerRecovery._creation(call) for call in self.backend.calls)
                )

    def test_exact_lifecycle_capacity_allows_create_and_restarted_recovery(self):
        final = copy.deepcopy(self.journal.state)
        final["phase"] = "recovering"
        final["resources"]["container:candidate"] = {
            "kind": "container",
            "name": "candidate",
            "token": "a" * 32,
            "state": "identified",
            "identity": "b" * 64,
        }
        limit = len((json.dumps(final, sort_keys=True) + "\n").encode())
        with patch("autotrain_bench.docker_recovery.MAX_JOURNAL_BYTES", limit - 1):
            with self.assertRaisesRegex(WorkerFailure, "capacity reached"):
                self.create()
            self.assertFalse(self.backend.objects)
        with patch("autotrain_bench.docker_recovery.MAX_JOURNAL_BYTES", limit):
            self.create()
            recovered = self.reopen()
            self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])
            self.assertFalse(self.backend.objects)
            self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])

    def test_pending_creations_share_reservations_and_recover_near_capacity(self):
        pending = (
            ("container", "agent"),
            ("container", "gateway"),
            ("container", "candidate"),
            ("network", "inference"),
            ("volume", "predictions"),
        )
        final = copy.deepcopy(self.journal.state)
        final["phase"] = "recovering"
        for kind, name in pending:
            final["resources"][kind + ":" + name] = {
                "kind": kind,
                "name": name,
                "token": "a" * 32,
                "state": "identified",
                "identity": "b" * (126 if kind == "volume" else 64),
            }
        limit = len((json.dumps(final, sort_keys=True) + "\n").encode())

        def lose_acknowledgement(args):
            if args[:2] == ["volume", "create"]:
                # Exercise the maximum supported JSON identity size, including
                # escaping, rather than relying on this fixture's short date.
                self.backend.objects[("volume", "predictions")]["CreatedAt"] = "\t" * 63
            raise WorkerFailure("creation acknowledgement lost")

        self.backend.after_create = lose_acknowledgement
        with patch("autotrain_bench.docker_recovery.MAX_JOURNAL_BYTES", limit):
            for kind, name in pending:
                with self.assertRaisesRegex(WorkerFailure, "acknowledgement lost"):
                    self.create(kind, name)
            saved = copy.deepcopy(self.journal.state)
            # Reserving only the next record would incorrectly admit it: the
            # five prior identities have yet to be captured in the journal.
            incomplete_reservation = copy.deepcopy(saved)
            incomplete_reservation["phase"] = "recovering"
            incomplete_reservation["resources"]["container:overflow"] = {
                "kind": "container",
                "name": "overflow",
                "token": "a" * 32,
                "state": "identified",
                "identity": "b" * 64,
            }
            self.assertLessEqual(
                len(
                    (json.dumps(incomplete_reservation, sort_keys=True) + "\n").encode()
                ),
                limit,
            )
            with self.assertRaisesRegex(WorkerFailure, "capacity reached"):
                self.create(name="overflow")
            self.assertEqual(self.journal.state, saved)
            self.assertEqual(set(self.backend.objects), set(pending))
            recovered = self.reopen()
            self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])
            self.assertFalse(self.backend.objects)
            self.assertTrue(recovered.recover(self.backend.run)["cleanup_complete"])

    def test_oversized_volume_identity_is_rejected_on_inspect_and_reload(self):
        def oversized_identity(args):
            self.backend.objects[("volume", "predictions")]["CreatedAt"] = "\t" * 64

        self.backend.after_create = oversized_identity
        with self.assertRaisesRegex(WorkerFailure, "unverified volume ownership"):
            self.create("volume", "predictions")
        with self.assertRaisesRegex(WorkerFailure, "unverified volume ownership"):
            self.journal.remove("volume", "predictions", self.backend.run)
        self.assertIn(("volume", "predictions"), self.backend.objects)
        self.journal.close()
        path = self.root / "worker/resources.json"
        state = json.loads(path.read_text())
        state["resources"]["volume:predictions"]["identity"] = "\t" * 64
        path.write_text(json.dumps(state))
        with self.assertRaisesRegex(WorkerFailure, "damaged"):
            DockerRecovery(self.root / "worker")

    def test_persistent_save_failure_does_not_prevent_durable_workload_stop(self):
        names = ("agent", "gateway", "training")
        for name in names:
            self.create(name=name)
        saved = (self.root / "worker/resources.json").read_bytes()
        with patch.object(
            self.journal, "_save", side_effect=WorkerFailure("read-only journal")
        ):
            for name in names:
                with self.assertRaisesRegex(WorkerFailure, "read-only journal"):
                    self.journal.remove("container", name, self.backend.run)
                self.assertNotIn(("container", name), self.backend.objects)
        self.assertEqual((self.root / "worker/resources.json").read_bytes(), saved)
        self.assertTrue(self.reopen().recover(self.backend.run)["cleanup_complete"])

    def test_recovery_stops_workloads_even_when_phase_and_ack_writes_fail(self):
        self.create(name="agent")
        self.create(name="gateway")
        self.create(name="atb-prediction-holder-1")
        self.create("volume", "predictions")
        private = self.journal.private_directory("hidden-")
        recovered = self.reopen()
        with patch.object(recovered, "_save", side_effect=WorkerFailure("disk full")):
            report = recovered.recover(self.backend.run)
        self.assertFalse(report["cleanup_complete"])
        self.assertEqual(
            {item["name"] for item in report["failures"]},
            {"resources.json", "agent", "gateway"},
        )
        self.assertEqual(
            set(self.backend.objects),
            {("container", "atb-prediction-holder-1"), ("volume", "predictions")},
        )
        self.assertTrue(private.exists())
        recovered.close()
        retry = self.reopen()
        self.assertEqual(retry.state["phase"], "active")
        with self.assertRaisesRegex(WorkerFailure, "cannot launch"):
            retry.execute(["run", "--name", "restart", "image"], self.backend.run)
        with self.assertRaisesRegex(WorkerFailure, "cannot launch"):
            retry.private_directory("hidden-")
        self.assertTrue(retry.recover(self.backend.run)["cleanup_complete"])
        self.assertFalse(private.exists())

    def test_newly_discovered_identity_still_requires_persistence_before_delete(self):
        def interrupted(args):
            raise WorkerFailure("creation acknowledgement lost")

        self.backend.after_create = interrupted
        with self.assertRaisesRegex(WorkerFailure, "acknowledgement lost"):
            self.create()
        with patch.object(
            self.journal, "_save", side_effect=WorkerFailure("disk full")
        ):
            # Repeated calls must not mistake the uncommitted RAM identity for
            # durable proof, even though the runtime's ownership labels match.
            for _ in range(2):
                with self.assertRaisesRegex(WorkerFailure, "disk full"):
                    self.journal.remove("container", "candidate", self.backend.run)
        self.assertIn(("container", "candidate"), self.backend.objects)
        self.assertFalse(any(call[0] == "rm" for call in self.backend.calls))
        self.assertTrue(self.reopen().recover(self.backend.run)["cleanup_complete"])

    def test_uncertain_commit_is_retained_without_blocking_durable_stops(self):
        self.create(name="agent")
        real_fsync = os.fsync

        def fail_directory_sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.ENOSPC, "directory fsync failed after rename")
            return real_fsync(fd)

        with patch(
            "autotrain_bench.docker_recovery.os.fsync", side_effect=fail_directory_sync
        ):
            with self.assertRaisesRegex(WorkerFailure, "cannot persist"):
                self.create(name="training")
            self.assertIn("container:training", self.journal.state["resources"])
            self.assertIn(
                "container:training",
                json.loads((self.root / "worker/resources.json").read_text())[
                    "resources"
                ],
            )
            with self.assertRaisesRegex(WorkerFailure, "uncommitted changes"):
                self.create(name="gateway")
            with self.assertRaisesRegex(WorkerFailure, "cannot persist"):
                self.journal.remove("container", "agent", self.backend.run)
        self.assertFalse(self.backend.objects)
        self.assertEqual(
            [
                creation[1]
                for call in self.backend.calls
                if (creation := DockerRecovery._creation(call))
            ],
            ["agent"],
        )
        report = self.reopen().recover(self.backend.run)
        self.assertFalse(report["cleanup_complete"])
        self.assertEqual(report["failures"][0]["name"], "training")
        self.assertIn("unknown", report["failures"][0]["reason"])

    def test_foreground_auto_removed_failures_are_not_unknown_creations(self):
        original = self.backend.run

        def failed_workload(args, **kwargs):
            result = original(args, **kwargs)
            if args[:1] == ["wait"]:
                result.stdout = "1"
            if args[:1] == ["start"]:
                result.returncode = 1
            return result

        with self.assertRaises(WorkerFailure):
            self.journal.execute(
                ["run", "--rm", "--name", "scorer", "image"], failed_workload
            )
        self.assertEqual(
            self.journal.state["resources"]["container:scorer"]["state"], "removed"
        )
        self.assertTrue(self.reopen().recover(self.backend.run)["cleanup_complete"])

    def test_symlinked_state_and_invalid_resource_id_are_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root / "worker", target_is_directory=True)
        with self.assertRaisesRegex(WorkerFailure, "private"):
            DockerRecovery(alias)
        self.create()
        self.journal.close()
        path = self.root / "worker/resources.json"
        state = json.loads(path.read_text())
        state["resources"]["container:candidate"]["identity"] = "--all"
        path.write_text(json.dumps(state))
        with self.assertRaisesRegex(WorkerFailure, "damaged"):
            DockerRecovery(self.root / "worker")


if __name__ == "__main__":
    unittest.main()
