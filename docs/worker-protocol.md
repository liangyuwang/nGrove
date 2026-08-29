# Worker Protocol boundary

AutoTrain-Bench Core owns benchmark decisions.  A Worker deployment owns
execution and enforcement.  The boundary between them is expressed in
benchmark terms and must not expose a scheduler, container runtime, or cloud
API.

```text
AutoTrain-Bench Coordinator
  authoritative benchmark state
              │
              │ Worker Protocol
              ▼
Research Worker      Evaluation Worker
  enforcement          hidden evaluation
       │                       │
       └──── immutable artifact┘
```

## Repository boundary

The core repository owns:

- Track A/B state and scoring rules;
- Agent and Worker protocol schemas;
- semantic capability and trust requirements;
- canonical Candidate validation and artifact identity;
- conformance specifications;
- leaderboard eligibility decisions.

Platform packages own:

- resource allocation and placement;
- image preparation;
- process, container, Pod, or job lifecycle;
- GPU, filesystem, and network enforcement mechanisms;
- platform logs and environment receipts;
- site-specific security hardening.

Platform packages are distributed separately.  Examples of possible package
names are `autotrain-worker-docker`, `autotrain-worker-orix`, and
`autotrain-worker-kubernetes`; these names are examples, not core backends.
Core must never grow platform selection branches or import their SDKs.

## Policy and enforcement

The Coordinator decides:

- the selected track and budget;
- whether another lease is allowed;
- whether a VAL result is valid;
- which Candidate is the declared incumbent;
- when TEST is requested;
- whether the result is leaderboard eligible.

The Worker enforces:

- local lease and episode deadlines;
- the maximum resource envelope;
- TRAIN/workspace/output visibility;
- hidden-dataset isolation;
- network restrictions;
- termination and immutable artifact capture.

A Worker may retain a small local enforcement envelope so a Coordinator outage
cannot extend a deadline.  It must not duplicate incumbent selection, scoring,
or other benchmark policy.

## Capabilities, conformance, and trust

`autotrain_bench.worker_protocol` describes capabilities by semantic name and
an enforcement level:

```text
UNSUPPORTED < UNVERIFIED < BEST_EFFORT < ENFORCED
```

Mechanisms may be included as evidence, but requirements never name a platform
mechanism.  For example, Core requires `training_network_isolation`; it does
not require a particular network command or policy object.

Three independent checks govern formal eligibility:

```text
required capabilities satisfied
AND conformance receipt passed
AND deployment trust is certified
```

Passing conformance does not certify a user-controlled deployment.  Likewise,
an organizer-controlled deployment is not eligible when required technical
capabilities are missing.  Missing or insufficient capabilities are recorded
explicitly and never trigger a fallback.

## Research and evaluation roles

The protocol models ResearchWorker and EvaluationWorker separately from the
first version because their trust domains differ:

- ResearchWorker sees TRAIN, the Agent workspace, and training outputs, but
  never hidden VAL/TEST;
- EvaluationWorker sees an immutable Candidate and the selected hidden split,
  but does not need the research workspace.

A local development Worker may implement both roles in one process.  A formal
deployment may place them on separate services connected only through an
immutable artifact reference.

## Current migration stage

The phase-one interface is an in-process Python protocol.  The Supervisor uses
semantic lease operations (`start_lease`, `exec_lease`, `lease_state`, and
`terminate_lease`) and semantic evaluation (`evaluate_candidate`).  It neither
stores container identifiers nor asks how those effects are implemented.

The pre-existing Docker runtime remains temporarily as the bundled v0 local
reference implementation so the runnable preview does not regress.  It is a
compatibility implementation, not the extension point for more platforms.  A
later extraction moves it to a separately distributed Worker package without
changing Supervisor semantics.  Its deadline guard currently shares the
Coordinator process, so it reports that capability as `BEST_EFFORT`; this is an
explicit development limitation rather than a silent fallback.

Deliberately deferred work includes:

1. canonical remote ArtifactRef transport;
2. an atomic `activate_episode` operation with Worker-local monotonic timing;
3. asynchronous operations, idempotent request IDs, and event streams;
4. Unix-socket and remote mTLS transports;
5. deployment certification.

Those features should be added to the Worker protocol and external Worker SDK,
not by adding platform code to AutoTrain-Bench Core.
