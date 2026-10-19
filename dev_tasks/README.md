# Development tasks

Development tasks exercise the `envcore` runtime and Agent integration
surface. They are not leaderboard benchmarks and their results must not be
placed on a leaderboard. They use the same task-and-environment launch,
policy, recording, and trusted Evaluator contracts as research task packages.
Task-scale data and evaluation choices may be smaller.

The repository intentionally contains one such task:

- `smoke_lm-pretrain/` runs a small TinyStories byte-language-model episode through
  Training Leases, research sealing, isolated postmortem ANALYSIS/TEST
  evaluation, Candidate snapshots, and
  explicit incumbent selection. It ships no Research Agent or starter workspace;
  the user must select the Research Agent explicitly at launch.
