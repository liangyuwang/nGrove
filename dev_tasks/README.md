# Development tasks

Development tasks exercise the Envcore runtime and Agent integration
surface. They are not leaderboard challenges and their results must not be
placed on a leaderboard. They use the same profile-bound release launch,
Policy, and trusted Evaluator contract as challenge packages; task-scale data
and evaluation choices may be smaller.

The repository intentionally contains one such task:

- `smoke_lm-pretrain/` runs a small TinyStories byte-language-model episode through
  Training Leases, research sealing, isolated postmortem ANALYSIS/TEST
  evaluation, Candidate snapshots, and
  explicit incumbent selection. It ships no Tested Agent or starter workspace;
  the user must select the Tested Agent explicitly at launch.
