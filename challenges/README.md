# Benchmark challenges

No frozen, leaderboard-eligible AutoTrain-Bench challenge has been released
yet. `lm-pretrain-v1/` contains the first runnable preview package, but its data is
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

Core additionally requires an `official` release with no provisional track or
runtime profiles, a complete comparison key, digest-pinned task and Agent
images, policy-compliant execution, and certified conformant Research and
Evaluation Workers. Formal deployments must also supply a trusted
`ReleaseVerificationReceipt` from the organizer-controlled Coordinator. The
receipt binds the accepted Core revision and the challenge, task, policy,
evaluator, TRAIN-manifest, ANALYSIS, and TEST digests; hashes calculated only from
local files cannot self-promote a run. When a Worker executes a derived SIF, VM
image, or other artifact rather than the canonical digest-pinned OCI artifact,
the same trusted receipt must also contain the accepted workload-resolution
evidence. A Worker's own transformation claim is not organizer acceptance.
Missing evidence is recorded in
`leaderboard_eligibility_violations`; it never silently promotes a preview.

A challenge release also publishes finite `track_configs` and
`runtime_profiles` maps. The task remains hardware-independent; the runtime
profile selects episode resources and the leaderboard compares results only
when all four comparison-key fields match:

```text
challenge_id + track + track_config + runtime_profile
```

Runtime-profile preflight confirms that the execution environment satisfies
the hardware requirements declared by the selected profile, including host
count and accelerator vendor, model, count, and minimum memory.

Preview profile maps can be inspected without preparing challenge data:

```bash
autotrain-bench challenge-info \
  --challenge challenges/lm-pretrain-v1/challenge.json
```

For the lightweight end-to-end development task, use
`dev_tasks/smoke_lm/`. It remains permanently non-leaderboard, but uses the
same profile-bound start command, Policy boundary, and trusted Evaluator
implementation as the challenge preview.
