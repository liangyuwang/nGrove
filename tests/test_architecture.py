import ast
import unittest
from dataclasses import fields
from pathlib import Path

from envcore.worker_protocol import (
    HiddenDatasetRequest,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    WorkloadRequest,
    WorkloadResolutionAcceptance,
)


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_environment_resolution_does_not_depend_on_benchmark_application(self):
        for filename in ("environment.py", "run_config.py"):
            with self.subTest(filename=filename):
                self.assertTrue(
                    {"benchmark", "envcore.benchmark"}.isdisjoint(
                        imported_modules(ROOT / "envcore" / filename)
                    )
                )

    def test_core_semantics_do_not_import_local_deployment_mechanisms(self):
        forbidden = {
            "local_mailbox",
            "local_agent_execution",
            "runtime",
            "envcore.local_mailbox",
            "envcore.local_agent_execution",
            "envcore.runtime",
        }
        for filename in (
            "conformance.py",
            "supervisor.py",
            "worker_protocol.py",
            "gateway_execution.py",
            "deadline_probe.py",
        ):
            with self.subTest(filename=filename):
                modules = imported_modules(ROOT / "envcore" / filename)
                self.assertTrue(forbidden.isdisjoint(modules))

    def test_task_evaluators_do_not_select_a_candidate_transport(self):
        for path in (
            ROOT / "dev_tasks/smoke_lm-pretrain/evaluator/evaluator.py",
            ROOT / "research_tasks/lm-pretrain/evaluator/evaluator.py",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                modules = imported_modules(path)
                self.assertNotIn("urllib.request", modules)
                self.assertNotIn("http", source.lower())
                self.assertNotIn("candidate-url", source)
                tree = ast.parse(source)
                functions = {
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                self.assertIn("evaluate", functions)

    def test_hidden_dataset_protocol_contains_no_deployment_location(self):
        for value_type in (HiddenDatasetRequest, ResolvedHiddenDataset):
            with self.subTest(value_type=value_type.__name__):
                names = {field.name for field in fields(value_type)}
                self.assertTrue(
                    names.isdisjoint(
                        {"path", "url", "credential", "mount", "cache_path"}
                    )
                )
        supervisor_source = (ROOT / "envcore" / "supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("config.analysis_path", supervisor_source)
        self.assertNotIn("config.test_path", supervisor_source)

    def test_workload_resolution_protocol_contains_no_platform_handle(self):
        forbidden = {
            "path",
            "url",
            "credential",
            "mount",
            "cache_path",
            "container_id",
            "job_id",
        }
        for value_type in (
            WorkloadRequest,
            ResolvedWorkload,
            WorkloadResolutionAcceptance,
        ):
            with self.subTest(value_type=value_type.__name__):
                names = {field.name for field in fields(value_type)}
                self.assertTrue(names.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
