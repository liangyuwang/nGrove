# Configurable research environments

Envcore provides run-owned compute and time configuration. A researcher can vary
accelerator type/count and research duration while controlling the other
experimental conditions. Better final scores alone do not establish
self-improvement; retain Agent
traces, artifact histories and resource evidence to study changes in strategy.

## Configuration

Install the checkout with `python3 -m pip install -e .`. Inspect a configuration
without preparing data, contacting Docker or allocating resources:

```bash
envcore environment-info --environment examples/environments/h100-x4.json
```

An environment file requires `schema_version`, `name` and `runtime_profile`.
The optional `budget` controls the current training protocol's time budget:

```json
{
  "schema_version": 1,
  "name": "h100-x4-2h",
  "runtime_profile": {
    "host_count": 1,
    "accelerator": {
      "vendor": "nvidia",
      "model": "NVIDIA H100 80GB HBM3",
      "model_match": "H100 80GB HBM3",
      "model_match_mode": "exact",
      "count": 4,
      "minimum_memory_mib": 80000
    }
  },
  "budget": {
    "track": "A",
    "total_seconds": 7200
  }
}
```

Copy the file into your trusted configuration directory and choose compute
resources and time before launch. Counts such as 1, 4 or 16 express separate
experiments; they do not provision hardware or prove it exists. Unknown fields
are rejected. The
Worker must admit the configuration and return matching allocation evidence.

Set `accelerator` to `null` for CPU-only execution. Optional `resource_contract`
and `evaluation_resource_contract` use the portable `ResourceEnvelope` schema.
The training contract must include requirements consistent with `host_count`
and `accelerator`; it may add Worker-supported properties. The
[CPU example](../examples/environments/cpu-2cores.json) adds a two-core CPU quota
and 1 GiB training memory cap. The local Worker applies exact `cpu.quota_cores`
and `memory.limit_bytes` limits and verifies runtime configuration before Lease
admission. These are caps, not exclusive CPU reservations; evidence remains
best-effort on this development Worker.

Training and evaluation contracts are independent. In the current adapter,
CUDA Candidate evaluation uses the selected training GPU set; CPU evaluation
does not. Set an evaluation contract separately to control serving CPU/memory
conditions. See [evaluation resource semantics](evaluation-resources.md).

## Time budgets and freezing

For Track A, `budget` requires `track: "A"` and a positive integer
`total_seconds`. This is the existing research clock: declared benchmark-owned
preparation is refunded; post-research TEST, ANALYSIS and cleanup use their own
clocks. It is not the resource allocator's reservation duration.

For Track B, use positive integer Lease count and duration:

```json
"budget": {
  "track": "B",
  "max_leases": 6,
  "lease_seconds": 1800
}
```

This permits six Training Leases of up to 30 minutes each. Time outside Leases
does not consume this budget. `--operator-max-research-seconds` remains a
separate private protection limit and does not become an Agent-visible budget.
The task's existing ANALYSIS capacity must cover `max_leases`; increasing the
Lease count does not silently change evaluation policy.

If `budget` is omitted, resolve it from the selected benchmark's `--track-config`
or the task's `--track`. An explicit budget overrides those numeric values for
the environment run, but its Track must match the selected Track. Source task
and benchmark files remain unchanged. Unknown fields, nonpositive/fractional
values and mixed Track A/B fields are errors before Worker setup.

The resolved environment always records the effective budget. It is fixed before
task reveal and governs deadline and Lease admission; editing the input JSON
afterward affects only later runs. The public task, status and trusted report
carry the same effective budget, and the environment fingerprint binds both
resources and time. `environment-info` inspects the input without a task; an
omitted budget remains unresolved there, so its input fingerprint is not the
final run fingerprint.

See the [four-H100, two-hour example](../examples/environments/h100-x4-2h.json)
and [four-H100, six-30-minute-Lease example](../examples/environments/h100-x4-6x30m.json).

## Environment runs

These commands are templates: supply your own Agent, prepared data, authorized
hardware and an external episode root. Existing preparation, operator protection
and Worker admission requirements still apply. No profile is chosen implicitly.

Use a task and its existing Track budget (the CPU example omits `budget`):

```bash
envcore start \
  --task dev_tasks/smoke_lm-pretrain/task.json --track A \
  --environment examples/environments/cpu-2cores.json \
  --episodes-root /srv/envcore-runs/episodes \
  --agent-runner container \
  --agent-image 'participant-agent@sha256:<actual-digest>' \
  --agent-command 'participant-agent run'
```

Or use a benchmark's task and Track with custom compute resources and two hours:

```bash
envcore start \
  --benchmark benchmarks/lm-pretrain/benchmark.json \
  --track-config preview-a-8h \
  --environment examples/environments/h100-x4-2h.json \
  --episodes-root /srv/envcore-runs/episodes \
  --agent-runner container \
  --agent-image 'participant-agent@sha256:<actual-digest>' \
  --agent-command 'participant-agent run'
```

`--environment` selects environment mode. To reuse a published hardware profile,
use `--mode environment --benchmark ... --track-config ... --runtime-profile ...`.
Do not combine `--environment` with `--runtime-profile` or manual GPU/device flags.

Environment mode does not edit release files, change data or images, relax task
policy, or expose hidden scores during research. The current task protocol still
uses Track A/B budgets, explicit incumbents, research sealing, TEST and
postmortem ANALYSIS. General task protocols and configurable feedback policies
remain future extensions; this implementation separates hardware configuration
from benchmark lanes.

## Benchmark runs and evidence

Existing benchmark launches without `--environment` retain benchmark semantics.
`--mode benchmark` makes this explicit and rejects `--environment`. The release
still owns its finite profiles and formal eligibility rules. Published A100,
H100 and H200 catalogs remain unchanged.

Environment runs record `leaderboard_eligible: false` and `comparison_key: null`.
Result schema v16 adds an `environment` record with the normalized configuration
and its SHA-256 identity. The public task, status and trusted report include it;
it contains compute configuration and resolved time budgets, not private
task/release fingerprints.
Allocation receipts retain evidence of resources actually supplied. Report
validation rejects a changed fingerprint, inconsistent resource/time contracts
or an environment result claiming a benchmark lane. Older v12–v15 benchmark reports
remain readable.

The environment fingerprint ignores display names and provisional labels, while
binding resource constraints, evaluation contracts and the resolved budget. It
is not a complete experiment comparison key. Scaling studies should report deliberately varied
compute/time dimensions and control task/data/image identities, Agent initial
configuration and the remaining execution conditions. Record contention and cache
differences; matching configuration hashes alone do not prove equal performance
conditions.

## Package and scope

Code imports `envcore`; `envcore.environment.RuntimeProfile` and
`EnvironmentConfig` are independent of benchmark catalogs. Benchmark resolution
reuses that profile type. Restricted Agent/Candidate helper bundles expose only
the public `envcore` modules.

The distribution and CLI are named `envcore`; `python -m envcore` also works.
Integrations use `ENVCORE_*` variables, `envcore.agent_metadata` and
`envcore.agent_usage` trace events, `envcore-tree-v2-sha256` Candidate identities
and `envcore-hidden-bundle-v1-sha256` hidden bundle identities. Worker ownership
labels use the `envcore.` prefix; recovery journals default to
`~/.envcore/workers/`.

The bundled deployment supports one host with CPU or full NVIDIA GPUs. This
change does not implement distributed execution, add other accelerator runtimes,
provide an Agent or certify a Worker. Platform mechanisms remain Worker-owned.
