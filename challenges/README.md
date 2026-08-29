# Benchmark challenges

No frozen, leaderboard-eligible AutoTrain-Bench challenge has been released
yet. `pretrain_v1/` contains the first runnable preview package, but its data is
generated locally and its release settings are still provisional.

This directory is reserved for published benchmark challenge packages. A
challenge owns its dataset construction, hidden evaluation splits, public
metric and Candidate ABI, task images, budgets, artifact limits, and release
policy. Participant Agents, starter workspaces, and development fixtures do not
belong here.

Every runnable challenge task must declare:

```json
{
  "task_kind": "benchmark_challenge",
  "leaderboard_eligible": false
}
```

`leaderboard_eligible` becomes `true` only when the corresponding submission
and comparison policy is published. Track A and Track B are execution modes
inside a challenge manifest, not directory categories.

A challenge release also publishes finite `track_configs` and
`runtime_profiles` maps. The task remains hardware-independent; the runtime
profile selects episode resources and the leaderboard compares results only
when all four comparison-key fields match:

```text
challenge_id + track + track_config + runtime_profile
```

Preview profile maps can be inspected without preparing challenge data:

```bash
autotrain-bench challenge-info \
  --challenge challenges/pretrain_v1/challenge.json
```

For the lightweight end-to-end development task, use
`dev_tasks/smoke_lm/`.
