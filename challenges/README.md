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
when all four comparison-key fields match, within the same frozen release and
hidden-data identity. The names alone do not establish equal cost conditions:

```text
challenge_id + track + track_config + runtime_profile
```

Challenge schema v3 profiles may declare `evaluation_resource_contract`,
independently of the Training `resource_contract`. Its normative and observational
properties use the existing extensible `ResourceEnvelope`. Formal releases must
declare a nonempty evaluation contract requiring `enforced` evidence; the release
owner selects the relevant CPU, memory, I/O, cache, and contention conditions.
An accepted envelope covers only its declared properties. Schema v2 remains
readable for existing previews and rejects the new field to prevent silent
omission by older Core versions. See [evaluation resources](../docs/evaluation-resources.md).

Agent-visible release metadata contains only the public release and selected
profile identifiers, release stage, and eligibility setting. `challenge-info`
publishes the profile catalog without full task/challenge file hashes. Those
hashes bind trusted release verification and remain in the organizer's episode
audit records, not the participant-facing manifest. New private release fields
are not implicitly public. The public task manifest separately declares the
selected budget and Training resources.

`state.json` and the `report` command retain complete organizer audit evidence.
They are not participant-facing or publication-safe exports: exposing a digest
of private configuration can enable enumeration, including from an earlier
episode when that configuration is reused. Public release identifiers must be
independent of private configuration fingerprints; formal verification still
checks the complete trusted file identities.

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
