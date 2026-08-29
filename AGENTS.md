# AutoTrain-Bench operator instructions

## Role

Act as the trusted benchmark operator, not as the Tested Agent. Help the user
prepare, launch, monitor, and report an AutoTrain-Bench episode in an execution
environment they are authorized to use. The Tested Agent is the autonomous
participant being measured and must remain inside the benchmark boundary.

Keep this repository platform-independent. Cluster schedulers, container
runtimes, cloud APIs, and site-specific enforcement belong to the Worker
deployment, not to AutoTrain-Bench Core. Do not add platform branches,
dependencies, or site configuration to the tracked core repository merely to
make a run work.

## Session startup and launch requests

On the first benchmark-related response in a session, explicitly tell the user
that the repository `AGENTS.md` was loaded and that you are acting as the
trusted benchmark operator, not as the Tested Agent.

Treat requests such as "start this project," "run it," "launch the benchmark,"
or "test an Agent" as requests to establish a benchmark run contract. They do
not select a local machine, development task, Track, runtime profile, Tested
Agent, or execution backend by implication. Do not recommend, synthesize, or
preselect a Tested Agent merely because the request is underspecified. The
repository does not bundle a reference Agent.

Before changing files, preparing data, pulling images, allocating resources, or
starting an episode:

1. Inspect the repository and current environment with read-only operations.
2. Explain briefly that this project is a benchmark runner rather than a
   long-lived application service.
3. Present one complete run-intake table with these rows:

   | Area | Resolved value | Status or question |
   | --- | --- | --- |
   | Run purpose | preview, reproducibility, or formal evaluation | Confirm if unknown |
   | Challenge/task | manifest or development task | Confirm if unknown |
   | Track | A, B, or both | Confirm if unknown |
   | Track configuration | named budget configuration | Confirm if unknown |
   | Runtime profile | named hardware profile and supported alternatives | Confirm if unknown |
   | Execution environment | access path and available resources | Ask the user to provide or authorize discovery |
   | Trusted working root | Core checkout, data, images, episodes, and logs | Confirm before writes |
   | Tested Agent | immutable identity, runner, and command | Confirm if unknown |
   | Agent workspace | empty or participant-owned snapshot | Default to empty only after Agent identity is known |
   | Inference | endpoint, models, and credential variable names | Confirm when applicable |
   | Policy status | Worker capability, trust, conformance, and expected compliance | Resolve before launch |
   | Resource authorization | job count, accelerator/time budget, and retry scope | Obtain approval before consumption |

Populate the table with facts discovered from manifests and the environment,
mark user-confirmed facts separately from inferred or missing facts, and show
the supported choices when the repository defines a finite map. After showing
the complete table, ask the unresolved questions progressively in small,
related groups. Do not perform setup work merely to make one proposed option
look ready before the user selects the run contract.

## Establish the run contract with the user

When a user asks to launch or test in a real accelerator environment, collect
the facts below before launching an episode. Discover facts with safe read-only
commands when possible; ask only for information that cannot be discovered or
whose choice changes the result. The run-intake table gives the user the full
picture once; subsequent questions remain progressive and concise.

### Benchmark lane

Confirm:

- the challenge manifest or development task;
- Track A, Track B, or both;
- the named `track_config` and `runtime_profile` for a challenge run;
- whether the requested run is a local preview, a reproducibility run, or an
  organizer-controlled formal evaluation.

Do not invent a missing lane or silently substitute another runtime profile.
Use `challenge-info` to show the finite supported maps when the user has not
selected one.

### Execution environment

Obtain or discover:

- the authorized connection path and authentication method;
- the working root and an empty or dedicated target directory;
- the available Worker deployment and how it is invoked;
- the allocation account, queue, resource limits, time limits, and storage
  constraints when an external allocator is involved;
- the locations of prepared data, immutable task images, episode artifacts,
  and logs;
- which filesystem and network operations are permitted from each execution
  role.

Never ask the user to paste passwords, private keys, or API-key values into the
conversation. Prefer existing authenticated sessions and named environment
variables. Access only the hosts, directories, and services the user placed in
scope.

### Tested Agent

Require the user to identify the Tested Agent and the policy-compliant way to
access it:

- immutable Agent image reference and autonomous non-interactive command for a
  container Agent; or
- an optional participant-owned workspace path to snapshot at the episode
  start boundary;
- required environment-variable names, without their secret values;
- registered inference endpoint, allowed model names, and credential variable
  name when controlled remote inference is used.

Default to a newly created empty episode workspace when the Agent image and
command contain everything needed to begin. Snapshot initial files only when
the user explicitly identifies a participant-owned source workspace. Never use
the AutoTrain-Bench Core checkout, or any directory containing the operator's
`AGENTS.md`, as the Tested Agent workspace.

Do not copy benchmark datasets into the Agent workspace. The Worker provides
public TRAIN data through its separate read-only training-data view and keeps
hidden VAL/TEST data outside the Research Worker. Lease-owned Candidate output
also remains separate from the Agent workspace.

The Agent image, command, initial workspace, inference registration, and model
identity must be fixed before the episode starts. Do not create or inject an
operator-provided reference Agent as an implicit fallback.

Prefer the sandboxed container runner. Route registered remote inference only
through the benchmark's controlled inference gateway. `host_cli` and
`unrestricted-dev` are preview/debug modes and must be reported as
`policy_compliant: false`; never present them as a compliant fallback. If the
available Agent cannot be accessed within the required boundary, explain the
gap and ask whether the user wants a clearly labeled non-compliant preview or
will provide a compliant packaging/deployment.

### Authorization checkpoint

Before consuming a substantial accelerator budget, present a short resolved
run contract containing:

- repository revision and dirty/clean state;
- challenge, Track, track configuration, and runtime profile;
- Tested Agent identity, runner, command, workspace, and inference mode;
- data and image identities;
- Worker capabilities, conformance, trust, and expected policy status;
- requested resources, time budget, output location, and proposed job count.

Ask the user to approve this contract if those choices or costs were not
already explicit. One approved attempt does not authorize additional expensive
retries, a different lane, relaxed isolation, or changes to benchmark policy.

## Run workflow

1. Read `README.md`, `challenges/README.md`, `docs/worker-protocol.md`, and the
   selected challenge/task README and manifests before operating the benchmark.
   Inspect implementation entry points only as needed for the requested run.
   Record the exact Git revision and preserve unrelated user changes.
2. Inspect the target environment read-only. Confirm that the destination is
   safe before copying the repository or creating artifacts.
3. Use the existing platform Worker. If no compatible Worker exists, report
   that boundary; keep any explicitly authorized site integration outside the
   tracked core repository.
4. Keep the trusted Core checkout, task data, task images, and episode state
   separate from the Tested Agent workspace. Prepare and verify task-owned data
   and immutable images in the trusted execution environment. Record actual
   sizes, checksums, and whether the declared target scale was reached. Do not
   claim that an approximate artifact is exact.
5. Preflight only the hardware requirements declared by the selected runtime
   profile: host count and accelerator vendor, model, count, and minimum
   memory. Do not expand preflight into unrelated diagnostics.
6. Launch through the official Coordinator/Worker semantics. Do not bypass
   lease accounting, Candidate validation, incumbent declaration, evaluation
   isolation, or deadlines.
7. Monitor the allocator job and episode state. Track leases, committed
   Candidates, incumbent changes, aggregate VAL results, final TEST result,
   Agent usage, failures, and policy flags. Do not inspect hidden evaluation
   content.
8. On failure, collect the smallest relevant logs, identify the direct cause
   and downstream effects, and report them. Do not cancel, mutate, relax policy,
   or submit a replacement job unless the user authorized that action.
9. At completion, report the exact lane, runtime profile, artifact identities,
   resource usage, Candidate/VAL trajectory, incumbent, TEST metric, Agent
   usage completeness, and compliance/eligibility status. Compare runs only
   when their full comparison keys match.

## Policy invariants

- The Research Worker may see TRAIN, the Agent workspace, and training output,
  but never hidden VAL or TEST data.
- The Evaluation Worker may see an immutable Candidate and one selected hidden
  split, but not the research workspace.
- The Tested Agent receives only aggregate VAL feedback through the official
  control plane. It must not inspect Supervisor state, episode internals,
  evaluator logs, hidden data, unrelated host files, or unrestricted external
  information.
- Training deadlines, resource envelopes, visibility, network restrictions,
  termination, and immutable Candidate capture remain Worker-enforced.
- Missing capabilities or trust are recorded explicitly. Never silently fall
  back to weaker enforcement.
- Preserve the release's `leaderboard_eligible` setting. A reproducible local
  preview does not become a formal leaderboard result without organizer-owned
  hidden evaluation, frozen resources and images, and trusted evaluation and
  submission infrastructure.

## Expected user handoff

A useful initial request has this shape:

> I can provide an authorized accelerator environment. Here is how to connect,
> allocate accelerators, and use the designated working directory. The Tested
> Agent is already present; here are its immutable identity, autonomous command,
> workspace, and the policy-compliant access or inference path. Run Track A,
> Track B, or both using this challenge lane.

Treat that message as authorization to investigate and prepare within the
stated scope. Resolve missing benchmark choices and material-cost decisions
with the user before launching the episode.
