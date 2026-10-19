import copy
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from envcore.agent_control import ProtocolError
from envcore.cli import build_parser, command_trajectory
from envcore.recording import RecordingConfig, RecordingStore, read_recording
from envcore.results import normalize_result_report, ResultSchemaError
from envcore.worker_protocol import WorkerTeardownFailure
from tests import test_supervisor


class RecordingConfigTests(unittest.TestCase):
    def test_config_round_trip_and_strict_validation(self):
        config = RecordingConfig(
            candidates="metadata", lease_outputs="discard_after_finalize"
        )
        self.assertEqual(RecordingConfig.from_record(config.record()), config)
        changed = copy.deepcopy(config.record())
        changed["configuration"]["artifacts"]["candidates"] = "all"
        with self.assertRaisesRegex(ValueError, "identity"):
            RecordingConfig.from_record(changed)
        for value in (
            {},
            {"schema_version": True},
            {"schema_version": 1, "typo": True},
            {"schema_version": 1, "capture_payloads": 1},
            {"schema_version": 1, "max_payload_bytes": False},
            {"schema_version": 1, "artifacts": {"candidates": "everything"}},
            {"schema_version": 1, "artifacts": {"lease_outputs": "immediate"}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RecordingConfig.from_manifest(value)

    def test_payload_dedup_limits_and_failures_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            store = RecordingStore(
                Path(root),
                RecordingConfig(max_payload_bytes=10, max_total_payload_bytes=10),
            )
            first = store.payload({"a": 1})
            self.assertEqual(first["status"], "saved")
            self.assertEqual(store.payload({"a": 1}), first)
            self.assertEqual(store.stored_bytes, first["size_bytes"])
            self.assertEqual(store.payload({"b": 2})["status"], "omitted_total_limit")
            self.assertEqual(
                store.payload({"long": "payload"})["status"], "omitted_size_limit"
            )
            self.assertEqual(store.payload(float("nan"))["status"], "invalid_json")
        with tempfile.TemporaryDirectory() as root:
            store = RecordingStore(Path(root), RecordingConfig())
            with patch("envcore.recording.os.open", side_effect=OSError("full")):
                self.assertEqual(store.payload({"a": 1})["status"], "capture_failed")
            self.assertEqual(store.payload({"a": 1})["status"], "saved")
        with tempfile.TemporaryDirectory() as root:
            store = RecordingStore(Path(root), RecordingConfig(capture_payloads=False))
            self.assertEqual(store.payload({"a": 1})["status"], "not_captured")
            self.assertEqual(list(store.root.iterdir()), [])

    def test_failed_payload_writes_are_removed_and_can_be_retried(self):
        fdopen = os.fdopen
        for stage in ("write", "flush", "fsync"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                store = RecordingStore(
                    Path(root), RecordingConfig(max_total_payload_bytes=14)
                )

                @contextmanager
                def failing_stream(descriptor, mode):
                    with fdopen(descriptor, mode) as stream:
                        write = stream.write

                        def partial_write(data):
                            write(data[:2])
                            raise OSError("partial write failed")

                        if stage == "write":
                            failure = patch.object(
                                stream, "write", side_effect=partial_write
                            )
                        elif stage == "flush":
                            failure = patch.object(
                                stream, "flush", side_effect=OSError("flush failed")
                            )
                        else:
                            failure = patch(
                                "envcore.recording.os.fsync",
                                side_effect=OSError("fsync failed"),
                            )
                        with failure:
                            yield stream

                with patch("envcore.recording.os.fdopen", side_effect=failing_stream):
                    for n in range(10):
                        self.assertEqual(
                            store.payload({"n": n})["status"], "capture_failed"
                        )
                self.assertEqual(list(store.root.iterdir()), [])
                self.assertEqual(store.stored_bytes, 0)
                saved = store.payload({"n": 0})
                self.assertEqual(saved["status"], "saved")
                self.assertEqual(store.payload({"n": 0}), saved)
                self.assertEqual(store.stored_bytes, saved["size_bytes"])

    def test_failed_payload_cleanup_still_consumes_capacity_until_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            store = RecordingStore(
                Path(root), RecordingConfig(max_total_payload_bytes=14)
            )
            with patch(
                "envcore.recording.os.fsync", side_effect=OSError("fsync failed")
            ), patch.object(Path, "unlink", side_effect=OSError("cleanup failed")):
                statuses = [store.payload({"n": n})["status"] for n in range(10)]
                self.assertEqual(
                    statuses, ["capture_failed"] * 2 + ["omitted_total_limit"] * 8
                )
                self.assertEqual(store.payload({"n": 0})["status"], "capture_failed")
            self.assertEqual(store.stored_bytes, 14)
            self.assertEqual(sum(p.stat().st_size for p in store.root.iterdir()), 14)
            for n in (0, 1):
                self.assertEqual(store.payload({"n": n})["status"], "saved")
                self.assertEqual(store.stored_bytes, 14)
            self.assertEqual(store.payload({"n": 2})["status"], "omitted_total_limit")
            self.assertEqual(sum(p.stat().st_size for p in store.root.iterdir()), 14)


class RecordingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_supervisor.SupervisorTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def events(self, supervisor):
        return [
            json.loads(line) for line in supervisor.events_path.read_text().splitlines()
        ]

    def exchange(self, supervisor, operation, **payload):
        mailbox = self.fixture.mailbox
        request_id = "request-" + str(supervisor.sequence)
        (mailbox.inbox / f"{request_id}.json").write_text(
            json.dumps({"op": operation, **payload})
        )
        self.assertTrue(supervisor.process_one_request())
        response = json.loads((mailbox.outbox / f"{request_id}.json").read_text())
        (mailbox.outbox / f"{request_id}.json").unlink()
        return response

    def test_actions_feedback_executions_and_candidates_join_without_hidden_feedback(
        self,
    ):
        supervisor = self.fixture.supervisor()
        self.exchange(supervisor, "lease_start")
        response = self.exchange(supervisor, "lease_exec", argv=["train"])
        candidate = self.exchange(
            supervisor, "lease_commit", candidate_path="/output/candidate"
        )["result"]
        self.exchange(
            supervisor, "declare_incumbent", candidate_id=candidate["candidate_id"]
        )
        self.exchange(supervisor, "finish")
        self.fixture.finalize_after_agent_stop(supervisor)
        view = read_recording(supervisor.episode_dir)
        self.assertEqual(view["issues"], [])
        events = view["events"]
        execution = next(e for e in events if e["type"] == "LEASE_COMMAND_FINISHED")
        self.assertEqual(execution["execution_id"], response["result"]["execution_id"])
        action = next(
            e
            for e in events
            if e["type"] == "ACTION_STARTED" and e["operation"] == "lease_exec"
        )
        delivery = next(
            e
            for e in events
            if e["type"] == "CONTROL_RESPONSE" and e["action_id"] == action["action_id"]
        )
        self.assertEqual(delivery["delivery"], "published")
        self.assertEqual(delivery["consumption"], "unknown")
        self.assertEqual(view["payloads"][delivery["payload"]["sha256"]], response)
        self.assertEqual(view["artifacts"][0]["artifact_id"], candidate["candidate_id"])
        self.assertEqual(
            view["postmortem"]["final_candidate_id"], candidate["candidate_id"]
        )
        self.assertEqual(len(view["postmortem"]["analysis_trajectory"]), 1)
        self.assertNotIn("recording", supervisor.public_status())
        self.assertNotIn(
            "recording", json.loads(supervisor.public_task_path.read_text())
        )
        for event in events:
            if event.get("visibility") == "agent_exchange" and "payload" in event:
                contents = view["payloads"][event["payload"]["sha256"]]
                result = contents.get("result", {})
                # finish can return pending placeholders; it cannot return
                # measurements that only exist after Agent termination.
                self.assertFalse(result.get("analysis_trajectory"))
                self.assertIsNone(result.get("test_performance_score"))
                for metadata in result.get("candidates", {}).values():
                    self.assertIsNone(metadata.get("analysis_raw_metrics"))

    def test_transport_records_the_bounded_response_not_the_handler_result(self):
        supervisor = self.fixture.supervisor()
        self.fixture.mailbox.maximum_response_bytes = 256
        response = self.exchange(supervisor, "status")
        view = read_recording(supervisor.episode_dir)
        generated = next(e for e in view["events"] if e["type"] == "ACTION_RETURNED")
        delivered = next(e for e in view["events"] if e["type"] == "CONTROL_RESPONSE")
        self.assertTrue(view["payloads"][generated["payload"]["sha256"]]["ok"])
        self.assertEqual(response["error"]["code"], "response_too_large")
        self.assertEqual(view["payloads"][delivered["payload"]["sha256"]], response)

    def test_rejected_publication_is_not_recorded_as_received(self):
        supervisor = self.fixture.supervisor()
        mailbox = self.fixture.mailbox
        (mailbox.inbox / "blocked.json").write_text('{"op":"status"}')
        (mailbox.outbox / "blocked.json").mkdir()
        supervisor.process_one_request()
        delivery = next(
            e for e in self.events(supervisor) if e["type"] == "CONTROL_RESPONSE"
        )
        self.assertEqual(delivery["delivery"], "not_published")

    def test_failures_and_capture_omissions_preserve_operation_semantics(self):
        supervisor = self.fixture.supervisor(
            recording=RecordingConfig(capture_payloads=False)
        )
        with self.assertRaises(ProtocolError):
            supervisor.handle({"op": "unknown"})
        supervisor.handle({"op": "lease_start"})
        with patch.object(
            self.fixture.worker, "exec_lease", side_effect=OSError("failure")
        ):
            with self.assertRaises(ProtocolError):
                supervisor.handle({"op": "lease_exec", "argv": ["train"]})
        view = read_recording(supervisor.episode_dir)
        self.assertTrue(
            any(e["type"] == "LEASE_COMMAND_FAILED" for e in view["events"])
        )
        self.assertFalse(
            any(i["code"] == "action_completion_unknown" for i in view["issues"])
        )
        self.assertTrue(
            any(i["code"] == "payload_not_captured" for i in view["issues"])
        )
        self.assertEqual(view["payloads"], {})

    def test_concurrent_observations_do_not_share_a_current_action(self):
        supervisor = self.fixture.supervisor()
        barrier = threading.Barrier(2)

        def handle(request):
            barrier.wait(timeout=5)
            supervisor.record_observation(
                source="runner",
                source_event_id=request["id"],
                kind="tool",
                data=request,
            )
            return {"id": request["id"]}

        with patch.object(supervisor, "_handle_request", side_effect=handle):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(
                        supervisor.handle,
                        [{"op": "status", "id": "a"}, {"op": "status", "id": "b"}],
                    )
                )
        self.assertEqual({r["id"] for r in results}, {"a", "b"})
        events = self.events(supervisor)
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        observations = [e for e in events if e["type"] == "RESEARCH_OBSERVATION"]
        self.assertEqual(len({e["action_id"] for e in observations}), 2)
        for observation in observations:
            action = [
                e for e in events if e.get("action_id") == observation["action_id"]
            ]
            self.assertEqual(
                [e["type"] for e in action],
                ["ACTION_STARTED", "RESEARCH_OBSERVATION", "ACTION_RETURNED"],
            )
        joined = supervisor.record_observation(
            source="runner",
            source_event_id="merge",
            kind="code_change",
            data={"diff": "+change"},
            links=tuple(
                {"relation": "uses", "event_id": e["event_id"]} for e in observations
            ),
        )
        view = read_recording(supervisor.episode_dir)
        self.assertEqual(
            len(next(e for e in view["events"] if e["event_id"] == joined)["links"]), 2
        )
        self.assertEqual(view["issues"], [])

    def test_reader_reports_corruption_missing_artifacts_and_open_actions(self):
        supervisor = self.fixture.supervisor()
        supervisor.handle({"op": "lease_start"})
        candidate = supervisor.handle(
            {"op": "lease_commit", "candidate_path": "/output/candidate"}
        )
        view = read_recording(supervisor.episode_dir)
        ref = next(e["payload"] for e in view["events"] if "payload" in e)
        (supervisor.episode_dir / ref["path"]).write_text("corrupt")
        supervisor._remove_staging_tree(
            supervisor.candidates_dir / candidate["candidate_id"].split(":")[1]
        )
        supervisor._event("ACTION_STARTED", action_id="interrupted")
        with supervisor.events_path.open("a") as stream:
            stream.write('{"incomplete":')
        view = read_recording(supervisor.episode_dir)
        codes = {issue["code"] for issue in view["issues"]}
        self.assertTrue(
            {
                "invalid_event",
                "payload_unavailable_or_corrupt",
                "artifact_missing_without_confirmed_removal",
                "action_completion_unknown",
            }.issubset(codes)
        )

    def test_recording_configuration_is_frozen_and_cli_export_is_read_only(self):
        supervisor = self.fixture.supervisor()
        supervisor.handle({"op": "status"})
        before = {
            p: p.read_bytes() for p in supervisor.episode_dir.rglob("*") if p.is_file()
        }
        args = build_parser().parse_args(
            ["trajectory", "--episode", str(supervisor.episode_dir)]
        )
        with patch("envcore.cli._print") as output:
            self.assertEqual(command_trajectory(args), 0)
        self.assertEqual(
            output.call_args.args[0]["episode_id"], supervisor.state["episode_id"]
        )
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        changed = copy.deepcopy(supervisor.state)
        changed["recording"]["configuration"]["capture_payloads"] = False
        with self.assertRaises(ResultSchemaError):
            normalize_result_report(changed)

    def test_reader_cross_checks_truncated_journal_against_committed_state(self):
        supervisor = self.fixture.supervisor()
        supervisor.handle({"op": "lease_start"})
        prefix = supervisor.events_path.read_text()
        candidate = supervisor.handle(
            {"op": "lease_commit", "candidate_path": "/output/candidate"}
        )
        supervisor.handle(
            {"op": "declare_incumbent", "candidate_id": candidate["candidate_id"]}
        )
        supervisor.handle({"op": "finish"})
        self.fixture.finalize_after_agent_stop(supervisor)

        for label, journal in (("truncated", prefix), ("empty", ""), ("missing", None)):
            with self.subTest(journal=label):
                if journal is None:
                    supervisor.events_path.unlink()
                else:
                    supervisor.events_path.write_text(journal)
                view = read_recording(supervisor.episode_dir)
                missing = [
                    i for i in view["issues"] if i["code"] == "state_event_missing"
                ]
                self.assertEqual(
                    {i["expected_type"] for i in missing},
                    {
                        "CANDIDATE_COMMITTED",
                        "INCUMBENT_DECLARED",
                        "RESULT_PUBLICATION_PREPARED",
                    },
                )
                self.assertEqual(len(view["artifacts"]), 1)
                artifact = view["artifacts"][0]
                self.assertEqual(artifact["artifact_id"], candidate["candidate_id"])
                self.assertEqual(artifact["observed_storage"], "present")
                self.assertEqual(artifact["storage_status"], "unknown")
                self.assertEqual(artifact["source_leases"], [candidate["source_lease"]])
                self.assertEqual(
                    artifact["commit_event_ids"],
                    [
                        f"{supervisor.state['episode_id']}:event:{candidate['event_seq']}"
                    ],
                )
        supervisor._remove_staging_tree(
            supervisor.candidates_dir / candidate["candidate_id"].split(":")[1]
        )
        view = read_recording(supervisor.episode_dir)
        self.assertEqual(view["artifacts"][0]["observed_storage"], "missing")
        self.assertIn(
            "artifact_missing_without_confirmed_removal",
            {i["code"] for i in view["issues"]},
        )

    def test_reader_rejects_state_references_to_mismatched_events(self):
        supervisor = self.fixture.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        self.fixture.finalize_after_agent_stop(supervisor)
        events = self.events(supervisor)
        for event_type, field in (
            ("CANDIDATE_COMMITTED", "candidate_id"),
            ("INCUMBENT_DECLARED", "declaration_index"),
            ("RESULT_PUBLICATION_PREPARED", "publication_id"),
        ):
            for changed_field in ("type", field):
                with self.subTest(event_type=event_type, field=changed_field):
                    changed = copy.deepcopy(events)
                    event = next(e for e in changed if e["type"] == event_type)
                    event[changed_field] = "mismatched"
                    supervisor.events_path.write_text(
                        "".join(json.dumps(e) + "\n" for e in changed)
                    )
                    issues = read_recording(supervisor.episode_dir)["issues"]
                    self.assertIn(
                        {
                            "code": "state_event_mismatch",
                            "event_id": event["event_id"],
                            "expected_type": event_type,
                        },
                        issues,
                    )

    def test_missing_journal_is_unknown_after_research_starts(self):
        supervisor = self.fixture.supervisor(begin=False)
        self.assertEqual(read_recording(supervisor.episode_dir)["events"], [])
        supervisor.begin()
        supervisor.events_path.unlink()
        self.assertIn(
            {"code": "event_journal_unavailable"},
            read_recording(supervisor.episode_dir)["issues"],
        )


class RecordingRetentionTests(unittest.TestCase):
    def test_final_test_does_not_authorize_disposal_after_teardown_failure(self):
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            supervisor = fixture.supervisor(
                recording=RecordingConfig(
                    candidates="metadata", lease_outputs="discard_after_finalize"
                )
            )
            supervisor.start_lease()
            candidate = supervisor.commit_lease("/output/candidate")
            supervisor.declare_incumbent(candidate["candidate_id"])
            supervisor.finish()
            fixture.worker.evaluation_exceptions["analysis"] = WorkerTeardownFailure(
                "cleanup failed"
            )
            fixture.finalize_after_agent_stop(supervisor)
            self.assertEqual(supervisor.state["status"], "finalized")
            self.assertEqual(len(list(supervisor.candidates_dir.iterdir())), 1)
            self.assertEqual(len(list(supervisor.runs_dir.glob("*/output"))), 1)
            self.assertTrue(
                any(
                    i["code"] == "artifact_retention_deferred"
                    for i in read_recording(supervisor.episode_dir)["issues"]
                )
            )
        finally:
            fixture.tearDown()

    def test_retention_runs_after_evaluation_and_keeps_metadata(self):
        for policy in ("all", "final", "metadata"):
            with self.subTest(policy=policy):
                fixture = test_supervisor.SupervisorTests()
                fixture.setUp()
                try:
                    supervisor = fixture.supervisor(
                        recording=RecordingConfig(
                            candidates=policy, lease_outputs="discard_after_finalize"
                        )
                    )
                    candidates = []
                    for index in range(2):
                        supervisor.start_lease()
                        lease = supervisor.state["active_lease"]["lease_id"]
                        (
                            supervisor.runs_dir / lease / "output/candidate/weights.bin"
                        ).write_bytes(bytes([index]))
                        candidates.append(
                            supervisor.commit_lease("/output/candidate")["candidate_id"]
                        )
                    supervisor.declare_incumbent(candidates[0])
                    supervisor.finish()
                    with self.assertRaisesRegex(RuntimeError, "finalized"):
                        supervisor.finalize_recording()
                    self.assertEqual(len(list(supervisor.candidates_dir.iterdir())), 2)
                    fixture.finalize_after_agent_stop(supervisor)
                    self.assertEqual(len(fixture.worker.evaluations), 3)
                    expected = 2 if policy == "all" else 1 if policy == "final" else 0
                    self.assertEqual(
                        len(list(supervisor.candidates_dir.iterdir())), expected
                    )
                    self.assertEqual(list(supervisor.runs_dir.glob("*/output")), [])
                    view = read_recording(supervisor.episode_dir)
                    bundles = [
                        a for a in view["artifacts"] if a["kind"] == "candidate_bundle"
                    ]
                    self.assertEqual(len(bundles), 2)
                    self.assertEqual(
                        sum(a["storage_status"] == "retained" for a in bundles),
                        expected,
                    )
                    self.assertEqual(view["issues"], [])
                    before = supervisor.events_path.read_bytes()
                    supervisor.finalize_recording()
                    self.assertEqual(before, supervisor.events_path.read_bytes())
                finally:
                    fixture.tearDown()

    def test_cleanup_failure_does_not_erase_a_final_result(self):
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            supervisor = fixture.supervisor(
                recording=RecordingConfig(candidates="metadata")
            )
            supervisor.start_lease()
            candidate = supervisor.commit_lease("/output/candidate")
            supervisor.declare_incumbent(candidate["candidate_id"])
            supervisor.finish()
            with patch.object(
                supervisor, "_remove_staging_tree", side_effect=OSError("read only")
            ):
                fixture.finalize_after_agent_stop(supervisor)
            self.assertEqual(supervisor.state["status"], "finalized")
            self.assertEqual(supervisor.state["outcome"], "success")
            view = read_recording(supervisor.episode_dir)
            self.assertEqual(view["artifacts"][0]["storage_status"], "retention_failed")
            supervisor.finalize_recording()
            self.assertEqual(
                read_recording(supervisor.episode_dir)["artifacts"][0][
                    "storage_status"
                ],
                "removed_by_policy",
            )
        finally:
            fixture.tearDown()
