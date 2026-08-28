# Development tasks

Development tasks exercise the AutoTrain-Bench runtime and Agent integration
surface. They are not benchmark challenges and their results must not be placed
on a leaderboard.

The repository intentionally contains one such task:

- `smoke_lm/` runs a small TinyStories byte-language-model episode through
  Training Leases, isolated VAL/TEST evaluation, Candidate snapshots, and
  explicit incumbent selection. Its optional `reference_agent/` is a known-good
  development fixture, not a competitive baseline.
