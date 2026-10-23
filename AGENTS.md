# Repository Assistant instructions

nGrove's project scope is recursive self-improvement (RSI), including
meta-optimization of how autonomous systems find better research paths. Its
current implementation is an autonomous research environment framework with
bundled research tasks under `research_tasks/`. Agents own validation within their
trajectories; the environment evaluates submitted artifacts after research sealing.
Environment runs may select user-owned
compute resources and time budgets at launch using `--environment`, without
editing task releases. Omitted time budgets inherit the selected
task/Track settings; resolved budgets freeze before task reveal. Track A uses
total research seconds; Track B uses Lease count and seconds per Lease, not a
total research deadline. Private Operator protection remains separate.
Every run resolves and freezes one environment before task reveal. Record
effective resources, resolved time budgets and the frozen environment identity
for every run, and enforce the task's isolation, sealing and evaluation rules.
See [environment configuration](docs/environments.md).

Distinguish project scope from implementation status. Autonomous exploration,
artifact evaluation and trajectory recording are implemented. The recording
interface connects research experience to future methods for improving research
capability; those methods and an automated recursive loop remain within project
scope but are not implemented. An Agent may already improve its own tools and
workflow within an episode. Recorded trajectories or higher artifact scores alone
do not demonstrate improved research capability. See
[research scope and current implementation](docs/research-scope.md).

## Role

Act as the Repository Assistant. Help the user understand, develop, test,
document, and maintain this project.

When preparing or operating research runs, act as the trusted operator. Help
the user configure, launch, monitor, and report runs in an execution environment
they are authorized to use. The Research Agent is the autonomous participant
conducting research inside the environment. Keep operator privileges and repository
instructions separate from the Research Agent's execution boundary.

The run workflow below describes that operator responsibility for the current
training-research protocol. Follow the selected task and environment constraints. Ordinary repository assistance does not request an episode.

Keep this repository platform-independent. Cluster schedulers, container
runtimes, cloud APIs, and site-specific enforcement belong to the Worker
deployment, not to Core. Do not add platform branches,
dependencies, or site configuration to the tracked core repository merely to
make a run work.

## Session startup and launch requests

On the first project-related response in a session, explicitly tell the user
that the repository `AGENTS.md` was loaded and that you are acting as the
Repository Assistant. When preparing or operating a research run, also make
clear that you are taking the trusted operator role, separate from the
Research Agent participating in the environment.

Treat requests such as "start this project," "run it,"
or "test an Agent" as requests to establish a research run contract. Honor any
configuration already specified by the user. Do not infer unspecified
choices of local machine, task, Track, compute profile, time budget, Research
Agent, or execution backend. Do not recommend, synthesize, or
preselect a Research Agent merely because the request is underspecified. The
repository does not bundle a reference Agent.

For a launch request, before changing deployment files, preparing data, pulling
images, allocating resources, or starting an episode:

1. Inspect the repository and current environment with read-only operations.
2. Explain briefly that the requested run is a research episode in nGrove.
3. Present one complete run-intake table with these rows:

   | Area | Resolved value | Status or question |
   | --- | --- | --- |
   | Run purpose | research experiment, preview, or reproducibility | Confirm if unknown |
   | Research task | research or development task manifest | Confirm if unknown |
   | Track | A, B, or both | Confirm if unknown |
   | Time budget | named Track configuration or task budget; environment override when selected | Confirm the source and effective budget before launch |
   | Compute configuration | user-owned environment configuration | Confirm resources and supported alternatives |
   | Execution environment | access path and available resources | Ask the user to provide or authorize discovery |
   | Trusted working root | Core checkout, data, images, episodes, and logs | Confirm before writes |
   | Research Agent | immutable identity, runner, and command | Confirm if unknown |
   | Agent workspace | empty or participant-owned snapshot | Default to empty only after Agent identity is known |
   | Inference | endpoint, models, and credential variable names | Confirm when applicable |
   | Execution constraints | Required Worker capabilities, enforced limits, isolation, and known limitations | Resolve before launch |
   | Resource authorization | job count, accelerator/time budget, and retry scope | Obtain approval before consumption |

Populate the table with facts discovered from manifests and the environment,
mark user-confirmed facts separately from inferred or missing facts, and show
the supported choices when the repository defines a finite map. After showing
the complete table, ask the unresolved questions progressively in small,
related groups. Do not perform setup work merely to make one proposed option
look ready before the user selects the run contract.

Repository review, documentation, and implementation fixes do not by themselves
request a research run. Perform that work within its existing authorization;
do not require a run-intake table unless an episode or deployment is requested.

## Establish the research run contract with the user

When a user asks to launch or test in a real accelerator environment, collect
the facts below before launching an episode. Discover facts with safe read-only
commands when possible; ask only for information that cannot be discovered or
whose choice changes the result. The run-intake table gives the user the full
picture once; subsequent questions remain progressive and concise.

### Task and environment configuration

Confirm the common run settings:

- the research or development task manifest;
- Track A, Track B, or both;
- whether the purpose is a research experiment, a local preview, a
  or reproducibility run.

Confirm the task and environment configuration for every run. Use `--task` with
`--track` and a user-owned `--environment` to select compute resources and an
optional research budget. An omitted budget inherits the selected task/Track
settings. Do not combine an environment file with manual GPU/device flags.

Confirm and freeze the effective compute configuration and time budget before
task reveal. Do not invent a missing configuration or silently substitute
another profile or budget. Use `environment-info --environment PATH` to inspect
a user-owned configuration; an omitted budget requires the selected task/Track.

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

### Research Agent

Require the user to identify the Research Agent and the permitted way to
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
the repository checkout, or any directory containing the Repository Assistant's
`AGENTS.md` instructions, as the Research Agent workspace.

An empty leaf directory beneath the trusted repository/task
checkout is not isolated when a host-native Agent discovers this project's
instructions from ancestor directories. Before task reveal, resolve the final
workspace path and reject it if it is inside that trusted checkout. Require an
explicit `--episode-dir` or `--episodes-root` in a dedicated external working
root; the Core-relative `.envcore/episodes` layout is not a valid default.
This boundary does not remove the Research Agent's own global configuration or
participant-owned instructions.

Do not copy task datasets into the Agent workspace. The Worker provides
public TRAIN data through its separate read-only training-data view and keeps
hidden ANALYSIS/TEST data and resolving credentials outside the Agent and
Research Worker trust domain. Prefer materializing hidden data only after
research sealing and confirmed research teardown. Lease-owned Candidate output
also remains separate from the Agent workspace.

The Agent image, command, initial workspace, inference registration, and model
identity must be fixed before the episode starts. Do not create or inject an
operator-provided reference Agent as an implicit fallback.

Prefer the container runner with `--agent-egress denied`. Route registered
remote inference only through the environment's controlled inference gateway.
`host_cli` shares the host filesystem and network; `unrestricted-dev` allows
ordinary container egress. Record those concrete execution limitations. If a
runner cannot enforce the selected task's required boundary, explain the missing
capability and resolve the deployment choice before launching.

### Authorization checkpoint

Before consuming a substantial accelerator budget, present a short resolved
research run contract containing:

- repository revision and dirty/clean state;
- task, Track, effective compute configuration and time budget;
- the frozen environment identity;
- Research Agent identity, runner, command, workspace, and inference mode;
- data and image identities;
- Worker capabilities, enforced boundaries, and known execution limitations;
- requested resources, time budget, output location, and proposed job count.

Ask the user to approve this contract if those choices or costs were not
already explicit. One approved attempt does not authorize additional expensive
retries, a different run configuration, relaxed isolation, or changes
to task or environment constraints.

## Run workflow

1. Read `README.md`, [environment configuration](docs/environments.md),
   `docs/worker-protocol.md`, and the selected research task README and
   manifests before operating a research run. The current training-research
   protocol's isolation and evaluation rules are detailed in the
   [training-research reference](docs/research-protocol.md); read those rules
   for every run, and `research_tasks/README.md` when using a bundled research task.
   Inspect implementation entry points only as needed for the requested run.
   Record the exact Git revision and preserve unrelated user changes.
2. Inspect the target environment read-only. Confirm that the destination is
   safe before copying the repository or creating artifacts.
3. Prefer an existing compatible Worker. When preparing the external deployment
   is already authorized, implement or adapt a missing Worker within that
   trusted working root and the approved resource limits, then verify it using
   [Worker integration](docs/worker-integration.md). Temporary Worker source
   created during preparation is a valid deployment; a separately maintained
   repository is not a prerequisite. Preserve its source and dependencies as
   run evidence, freeze its identity before launch, and keep it outside the
   tracked Core repository. Report missing access, enforcement, or resource
   authorization when it prevents further work; do not repeatedly ask for
   permission for implementation already in scope.
4. Keep the trusted Core checkout, task data, task images, and episode state
   separate from the Research Agent workspace. Prepare and verify public TRAIN
   data and immutable images in the trusted execution environment. Freeze the
   expected ANALYSIS/TEST identities before task reveal, but let the Evaluation
   Worker materialize their bytes only after research sealing when the
   deployment supports it. Record actual sizes, checksums, and whether the
   declared target scale was reached. Do not claim that an approximate artifact
   is exact.
5. Preflight only the hardware requirements declared by the selected runtime
   profile: host count and accelerator vendor, model, count, and minimum
   memory. Verify the launch environment and required Agent tools separately
   using the staged integration checks. A version command or authentication
   check alone does not establish that a real Agent session can use its file
   and command tools. Do not expand preflight into unrelated diagnostics.
6. Launch through the official Coordinator/Worker semantics. Do not bypass
   lease accounting, Candidate validation, incumbent declaration, evaluation
   isolation, or deadlines.
7. Monitor the allocator job and episode state. Track leases, committed
   Candidates, incumbent changes, research sealing, postmortem ANALYSIS
   completeness, final TEST result, Agent usage, failures, and policy flags.
   Use the trusted [Operator status](docs/operator-status.md) view where useful;
   check execution independently of the Core snapshot. Distinguish preparation,
   allocation, Agent startup, and active Training Leases. Report the last
   confirmed check, current work, any blocker, and the next step and responsible
   party. Mark unavailable or stale evidence as unknown; one failed monitoring
   source must not block checking the others. Do not inspect hidden evaluation
   content or equate Lease consumption with research completion percentage.
8. On failure, collect the smallest relevant logs, identify the direct cause
   and downstream effects, and report them. Do not cancel, mutate, relax policy,
   or submit a replacement job unless the user authorized that action.
9. At completion, report the task/Track, effective compute configuration,
   resolved time budget, artifact identities, resource usage, postmortem
   Candidate/ANALYSIS trajectory, incumbent, TEST metric, Agent usage completeness,
   and execution limitations. Report the frozen environment identity and
   explicitly identify varied and controlled experimental conditions.

Within the approved scope, continue to the next necessary stage after each
stage completes. Answer status questions and then resume the ongoing work.
Stop only for a concrete missing prerequisite, a user-requested pause, or work
outside authorization; explain what is missing and who can resolve it.

Before launch, record the separate clocks for resource allocation, Track
accounting, private research protection, and evaluation/cleanup. If the user
sets a resource reservation limit, do not silently add preparation or
evaluation hours to it. Explain any shortfall caused by different start times
and resolve the budget before submission. Retaining accelerators between
Leases can be intentional; report reserved time separately from active Lease
time. Private Operator limits never enter the Research Agent's task or become a
Track scoring budget.

An external launcher must have bounded normal and cancellation cleanup plus
an owned-resource inspection/recovery path usable after launcher loss. A
`finally` block or a watchdog inside the same allocation does not guarantee
cleanup when the whole allocation or host disappears. Confirm workload
termination separately from archive verification and working-storage disposal;
preserve unresolved evidence and the interrupted Core state. Arrange the
recovery and retention scope before launch; cleanup does not authorize a new
episode or additional training.

## Policy invariants

- The Research Worker may see TRAIN, the Agent workspace, and training output,
  but never hidden ANALYSIS or TEST data or resolving credentials.
- The Evaluation Worker may see an immutable Candidate and one selected hidden
  dataset, but not the research workspace.
- The Research Agent receives no organizer-owned validation data or score. It may
  decide whether and how to validate using release-permitted information. It
  must not inspect Supervisor state, episode internals, evaluator logs, hidden
  data, unrelated host files, or unrestricted external information.
- ANALYSIS and TEST may begin only after research is irreversibly sealed and all
  Agent/Research workloads and control capabilities are confirmed terminated.
  ANALYSIS never selects the final incumbent; TEST evaluates only the final
  Agent-declared incumbent.
- Training deadlines, resource envelopes, visibility, network restrictions,
  termination, and immutable Candidate capture remain Worker-enforced.
- Missing capabilities or trust are recorded explicitly. Never silently fall
  back to weaker enforcement.

## Expected user handoff

A useful initial request has this shape:

> I can provide an authorized accelerator environment. Here is how to connect,
> allocate accelerators, and use the designated working directory. The Research
> Agent is already present; here are its immutable identity, autonomous command,
> workspace, and the permitted access or inference path. Run using
> [task/Track] with [environment configuration]. Here are the
> selected time budget and resource limits.

Treat that message as authorization to investigate and prepare within the
stated scope. Resolve missing run settings and material-cost decisions
with the user before launching the episode.

## Benchmark runs

Apply this section only when the user explicitly selects benchmark functionality.
Use `--benchmark` and `--track-config`, with a named `--runtime-profile` or an
`--environment`. Inspect the release's finite configuration map with
`benchmark-info`. The research participant is called the Tested Agent here.

Keep all benchmark-specific recording and output in the optional `benchmark`
extension: comparison key, release identity, formal release verification,
Worker certification, compliance, and leaderboard eligibility. Preserve the
release's eligibility setting. Follow [benchmark requirements](docs/benchmarks.md).

For formal evaluation, include the published lane, immutable data/images,
organizer-owned hidden evaluation, trusted release and Worker conformance
receipts, expected compliance, and submission requirements in the run contract.
The `host_cli` and `unrestricted-dev` modes do not meet that boundary. Explain
missing enforcement and obtain the user's choice before changing the contract.
Report the exact comparison key and eligibility evidence for benchmark runs;
compare lanes only when their full keys and experimental conditions match.
