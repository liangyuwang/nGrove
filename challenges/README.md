# Benchmark challenges

No frozen, leaderboard-eligible AutoTrain-Bench challenge has been released
yet. `lm-pretrain/` contains the first runnable preview package, but its data is
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

## Reference environments and custom comparisons

For research on hardware scaling, Envcore also supports run-owned
[environment configurations](../docs/environments.md). Select `--environment`
at launch to reuse a task and budget with custom hardware without editing this
challenge release. Such runs have no benchmark comparison key and remain
leaderboard-ineligible. The published-lane rules below apply to benchmark mode.

Official benchmark comparisons will use a small set of designated reference
environments, published and frozen with each release. Organizers are responsible
for verifying those environments and the conditions under which official results
are measured. No official reference environment has been finalized for the
current preview.

A reference environment covers Training and Candidate evaluation separately.
Its definition must specify the conditions relevant to comparison, including
GPU model, count, memory and interconnect topology; CPU model and allocated
cores; host-memory allocation and NUMA placement; GPU power limits; storage,
cache and sharing/isolation policies; and the task-owned base images and software
requirements. These details define one environment, rather than independent
leaderboard categories for every hardware combination. The official comparison
set does not expand automatically when a runnable profile is added.

Users may adapt the benchmark to their own hardware through a distinct runtime
profile in a derived preview release, selected with `--runtime-profile`. Keep
`leaderboard_eligible: false` and preserve the original release provenance;
changed resource requirements must not reuse an official profile identity.
When comparing Agents, the evaluation organizer is responsible for:

- providing the same benchmark task and data identities, Track budgets,
  task-owned starting environment, and evaluation and policy rules;
- providing consistent effective Training and evaluation resources, including
  CPU, memory, GPU connectivity and contention conditions, not just matching GPU
  product names;
- publishing the actual configuration, relevant run evidence and any differences
  that limit the comparison.

The Agents' own identities and initial software remain participant-owned and
recorded; fixing the available resources does not restrict how Agents explore
within the benchmark rules. Custom results are welcome for independent research
and community sharing, but do not automatically qualify for official rankings
or direct comparison with results from another environment.

User responsibility does not replace Worker enforcement. Existing resource,
budget, isolation and receipt checks still apply to the selected profile.
Passing those checks establishes only its declared requirements, not that every
condition affecting fairness has been specified. Matching comparison keys and
hardware names alone therefore do not establish equivalent conditions; see
[evaluation resource comparability](../docs/evaluation-resources.md).

## Runtime profile contracts

Challenge schema v4 adds optional accelerator `model_match_mode: "exact"`.
The default remains case-sensitive substring matching (`"contains"`). Exact
matching compares the complete reported device name during selection and
verification, and projects to an `accelerator.model` resource requirement with
relation `exact` and one identical model string per allocated GPU. Use distinct
profiles for hardware variants; do not broaden a match to make a different
variant pass. Releases using exact matching require schema v4 so older Core
versions reject the manifest instead of silently applying substring matching.

Challenge schema v3 and later profiles may declare `evaluation_resource_contract`,
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
  --challenge challenges/lm-pretrain/challenge.json
```

For the lightweight end-to-end development task, use
`dev_tasks/smoke_lm-pretrain/`. It remains permanently non-leaderboard, but uses the
same profile-bound start command, Policy boundary, and trusted Evaluator
implementation as the challenge preview.
