# Configurable research environments

Envcore separates run-owned hardware configuration from benchmark lanes. A
researcher can vary hardware while keeping the task, Agent and research budget
fixed. Better final scores alone do not establish self-improvement; retain Agent
traces, artifact histories and resource evidence to study changes in strategy.

## Configuration

Install the checkout with `python3 -m pip install -e .`. Inspect a configuration
without preparing data, contacting Docker or allocating resources:

```bash
envcore environment-info --environment examples/environments/h100-x4.json
```

An environment file has three fields:

```json
{
  "schema_version": 1,
  "name": "my-hardware",
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
  }
}
```

Copy the file into your trusted configuration directory and choose hardware
before launch. Counts such as 1, 4 or 16 express separate experiments; they do
not provision hardware or prove it exists. Unknown fields are rejected. The
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

## Environment runs

These commands are templates: supply your own Agent, prepared data, authorized
hardware and an external episode root. Existing preparation, operator protection
and Worker admission requirements still apply. No profile is chosen implicitly.

Use a task and its existing Track budget:

```bash
envcore start \
  --task dev_tasks/smoke_lm-pretrain/task.json --track A \
  --environment examples/environments/cpu-2cores.json \
  --episodes-root /srv/autotrain-runs/episodes \
  --agent-runner container \
  --agent-image 'participant-agent@sha256:<actual-digest>' \
  --agent-command 'participant-agent run'
```

Or reuse a challenge's named budget with custom hardware:

```bash
envcore start \
  --challenge challenges/lm-pretrain/challenge.json \
  --track-config preview-a-8h \
  --environment examples/environments/h100-x4.json \
  --episodes-root /srv/autotrain-runs/episodes \
  --agent-runner container \
  --agent-image 'participant-agent@sha256:<actual-digest>' \
  --agent-command 'participant-agent run'
```

`--environment` selects environment mode. To reuse a published hardware profile,
use `--mode environment --challenge ... --track-config ... --runtime-profile ...`.
Do not combine `--environment` with `--runtime-profile` or manual GPU/device flags.

Environment mode does not edit release files, change data or images, relax task
policy, or expose hidden scores during research. The current task protocol still
uses Track A/B budgets, explicit incumbents, research sealing, TEST and
postmortem ANALYSIS. General task protocols and configurable feedback policies
remain future extensions; this implementation separates hardware configuration
from benchmark lanes.

## Benchmark runs and evidence

Existing challenge launches without `--environment` retain benchmark semantics.
`--mode benchmark` makes this explicit and rejects `--environment`. The release
still owns its finite profiles and formal eligibility rules. Published A100,
H100 and H200 catalogs remain unchanged.

Environment runs record `leaderboard_eligible: false` and `comparison_key: null`.
Result schema v16 adds an `environment` record with the normalized configuration
and its SHA-256 identity. The public task, status and trusted report include it;
it contains hardware configuration, not private task/release fingerprints.
Allocation receipts retain evidence of resources actually supplied. Report
validation rejects a changed fingerprint, inconsistent resource contracts or an
environment result claiming a benchmark lane. Older v12–v15 benchmark reports
remain readable.

The hardware fingerprint ignores display names and provisional labels, while
binding resource constraints and evaluation contracts. It is not a complete
experiment comparison key. Scaling studies should report deliberately varied
hardware and control task/data/image identities, Agent initial configuration,
research budgets and relevant execution conditions. Record contention and cache
differences; matching configuration hashes alone do not prove equal performance
conditions.

## Migration and scope

New code imports `envcore`; `envcore.environment.RuntimeProfile` and
`EnvironmentConfig` are independent of challenge catalogs. Benchmark resolution
reuses that profile type. Prior `benchcore` module imports forward to the same
Python objects, preserving protocol class identity. Both namespaces work in the
restricted Agent/Candidate helper bundles without exposing trusted modules.

`BENCHCORE_*` variables, existing trace event names, artifact hash algorithms,
Worker ownership labels and `~/.benchcore` recovery storage retain their existing
identities. Renaming must not invalidate artifacts or strand owned resources.
The distribution remains `autotrain-bench`; both `envcore` and `autotrain-bench`
CLI entry points are installed. `python -m envcore` also works.

The bundled deployment supports one host with CPU or full NVIDIA GPUs. This
change does not implement distributed execution, add other accelerator runtimes,
provide an Agent or certify a Worker. Platform mechanisms remain Worker-owned.
