# Research scope and current implementation

nGrove is a project for **recursive self-improvement (RSI)** of autonomous
research systems. Its scope includes **meta-optimization**: improving how a system
finds better research paths, learns from experiments, and develops its research
methods, tools, and workflows.

Model training is the first concrete setting. The ambition is to discover paths
that outperform established human-designed training procedures under comparable
conditions, and to develop the ability to keep finding such paths. This is a
research objective; the current preview results do not establish that it has
been achieved.

## Scope, implementation and interface

| Level | Responsibility and status |
| --- | --- |
| Project scope | Develop and study ways to improve the research system's ability to discover better paths, including recursive use of research experience. |
| Current implementation | Provide a configurable environment for autonomous research, evaluate submitted artifacts after research sealing, and record observable trajectories. |
| Implemented connection point | Expose research evidence through the recording interface for future analysis, learning and system-update methods. |
| Future work within scope | Develop those methods, test whether they improve research capability, and connect successive rounds into a recursive improvement loop. |

Within an episode, an Agent can already use permitted experience to change its
code, tools, memory and research workflow autonomously. The current framework
supplies the environment and records observable evidence of that exploration.
It does not yet supply a method that consumes trajectories to improve the
researcher, or an automated loop that feeds those improved systems into new runs.

The implementation boundary is a stage of the project. Future methods may be
developed as modules or separate packages; their placement does not narrow the
project's RSI scope.

## From a better model to a better research method

In the LM task, model parameters are optimized during training. The Agent explores
training procedures: data choices, architectures, optimization methods and serving
implementations. The meta-optimization question concerns the research strategy
that discovers those procedures: which hypotheses to pursue, how to design
experiments, how to use failures, and when to change direction.

The research process is itself an object of study. We want to understand what
helps a system repeatedly find useful improvements within its constraints, how
it spends its research budget, and which methods remain effective as the problem
changes. Evaluated artifacts provide evidence of outcomes; available trajectories
provide evidence for studying the paths that produced them.

A trajectory preserves evidence of the path taken, including unsuccessful
attempts and corrections where captured. These records can support analysis of
research behavior, the design of tools and workflows, and future methods for
improving the researcher. Their value is not limited to training a model or
comparing final scores.

## Independent exploration and accumulated experience

Independent episodes provide separate research paths. Repeated runs under the
same declared conditions can help investigate variation and recurring approaches;
explicitly varied conditions can help investigate where an approach remains
useful. Distinct sessions alone do not guarantee different strategies or establish
that any observed improvement will transfer.

Independence applies to the episodes in a given round: each starts from its agreed
initial state and keeps its research state and results within its own boundary.
A future outer method may use permitted experience from completed episodes to
construct the next declared research system. Subsequent episodes would start from
that system under a new frozen run contract. This allows experience to accumulate
across rounds while preserving independent exploration within each round.

No particular update mechanism is selected yet. Changes may involve model
parameters, memory, tools, harnesses or research strategies, under an explicit
information policy. The intended recursive loop is:

```text
Research system → autonomous exploration → recorded trajectory
       ↑                                      │
       └──── improved research capability ←───┘
             future update methods and orchestration
```

Exploration and recording are implemented. Methods that turn this experience into
an updated research system, and orchestration of successive rounds, remain future
work. The central open question is how experience from independent exploration
can accumulate into transferable research capability. Which experience carries
over, how it should change the system, and whether the benefits persist require
investigation.

## Testing improvements in research capability

An update produces a new research system whose capability must be tested.
Controlled comparisons across repeated episodes can examine whether it finds
useful improvements more reliably or at lower research cost under comparable
conditions. Tests on new problems, budgets or starting points can examine whether
those gains transfer. Record the changed system components, available information,
resource conditions, failures and variation so the basis of each claim is clear.

A higher Candidate score in one episode establishes an artifact outcome. It does
not by itself establish that the researcher became better at finding improvements.
Evidence that an updated system can repeatedly discover further improvements,
and adapt its methods to new conditions, would support progress toward the RSI
objective. The current preview results do not establish that this loop works.

## The recording interface

The existing [research recording](recording.md) interface is the connection point
between the current environment and future meta-optimization methods:

- The event journal links actions, returned responses, transport publication,
  command executions, Candidate commits and incumbent declarations.
- Bounded payload capture, artifact identities and retention records preserve
  available evidence and make omissions explicit.
- `Supervisor.record_observation()` accepts optional trusted runner observations
  of tools, feedback or code changes, with source identities and explicit links.
- `ngrove trajectory --episode PATH` exports recorded evidence, resource and
  usage information, and a separate postmortem section for trusted consumers.

Coverage depends on the configured capture and runner. A trajectory is not a
complete Agent-state snapshot, a causal explanation or a replayable session.
Response publication does not prove consumption. Future consumers must preserve
these distinctions and the declared gaps when using the data.

The complete export is private Operator evidence. Agent-visible research
feedback and hidden postmortem measurements retain separate information
boundaries. A future learning-data export needs an explicit policy for selecting
permitted information; adding a consumer does not authorize exposing hidden
ANALYSIS/TEST data or scores to the research Agent.

## Research tasks and diagnostics

Research tasks supply concrete problems in which to study these questions.
LM pretraining is the current task; LM Open World is a proposed broader task.
These are settings for the project's RSI objective.

The [research-dynamics proposal](research-dynamics.md) describes a possible
read-only analysis consumer of the evidence. Its proposed diagnostics are not
implemented and do not themselves update the research system. The existing
recording interface, proposed diagnostics and future improvement methods have
distinct implementation statuses within the same project scope.
