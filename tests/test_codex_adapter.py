import unittest
from pathlib import Path

from autotrain_bench.agent_prompt import HOST_CLI_PROMPT
from autotrain_bench.cli_adapters.codex import build_codex_command


class CodexAdapterTests(unittest.TestCase):
    def test_command_is_non_interactive_logged_and_workspace_scoped(self):
        command = build_codex_command(
            codex_binary="/usr/local/bin/codex",
            model="test-model",
            reasoning_effort="high",
            workspace=Path("/episode/workspace"),
            control_dir=Path("/episode/control"),
            shell_environment={
                "AUTOTRAIN_BENCH_CONTROL_DIR": "/episode/control",
                "AUTOTRAIN_BENCH_FORWARDED_ENVIRONMENT": '["VISIBLE"]',
                "AUTOTRAIN_BENCH_TASK": "/episode/public/task.json",
                "PATH": "/usr/bin",
                "PYTHONPATH": "/package",
                "VISIBLE": "yes",
            },
        )
        self.assertEqual(command[:2], ["/usr/local/bin/codex", "exec"])
        for flag in (
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
        ):
            self.assertIn(flag, command)
        self.assertIn("/episode/control", command)
        self.assertIn("/episode/workspace", command)
        self.assertIn("test-model", command)
        self.assertIn('web_search="disabled"', command)
        self.assertIn('shell_environment_policy.inherit="none"', command)
        self.assertIn(
            'shell_environment_policy.set.HOME="/episode/workspace"',
            command,
        )
        self.assertFalse(any("CODEX_HOME" in value for value in command))
        self.assertIn(
            'shell_environment_policy.set.VISIBLE="yes"',
            command,
        )
        self.assertIn(
            "agent declare-incumbent <candidate-id>",
            HOST_CLI_PROMPT,
        )
        self.assertIn(
            "evaluation.candidate_boundary.prediction_manifest",
            HOST_CLI_PROMPT,
        )
        self.assertIn("evaluation.candidate_boundary.entrypoint", HOST_CLI_PROMPT)
        self.assertIn("maximize performance_score", HOST_CLI_PROMPT)
        self.assertIn("minimizing time_per_task", HOST_CLI_PROMPT)
        self.assertIn("Pareto-competitive", HOST_CLI_PROMPT)
        self.assertIn("evaluation.cost", HOST_CLI_PROMPT)
        self.assertIn("allowed public", HOST_CLI_PROMPT)


if __name__ == "__main__":
    unittest.main()
