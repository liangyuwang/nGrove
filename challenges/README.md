# Benchmark challenges

No official AutoTrain-Bench challenge has been released yet.

This directory is reserved for published benchmark challenge packages. A
challenge owns its dataset construction, hidden evaluation splits, public
metric and Candidate ABI, task images, budgets, artifact limits, and release
policy. Participant Agents, starter workspaces, and development fixtures do not
belong here.

Every challenge manifest must declare:

```json
{
  "task_kind": "benchmark_challenge",
  "leaderboard_eligible": false
}
```

`leaderboard_eligible` becomes `true` only when the corresponding submission
and comparison policy is published. Track A and Track B are execution modes
inside a challenge manifest, not directory categories.

For the lightweight end-to-end development task, use
`dev_tasks/smoke_lm/`.
