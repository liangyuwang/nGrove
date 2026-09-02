import ast
import unittest
from pathlib import Path


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
    def test_core_semantics_do_not_import_local_deployment_mechanisms(self):
        forbidden = {
            "local_mailbox",
            "local_agent_execution",
            "runtime",
            "autotrain_bench.local_mailbox",
            "autotrain_bench.local_agent_execution",
            "autotrain_bench.runtime",
        }
        for filename in ("supervisor.py", "worker_protocol.py"):
            with self.subTest(filename=filename):
                modules = imported_modules(ROOT / "autotrain_bench" / filename)
                self.assertTrue(forbidden.isdisjoint(modules))


if __name__ == "__main__":
    unittest.main()
