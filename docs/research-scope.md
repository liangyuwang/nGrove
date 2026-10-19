# Research scope and current implementation

Envcore is a project for **recursive self-improvement (RSI)** of autonomous
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
project's RSI scope. No particular update mechanism is selected yet: changes may
involve model parameters, memory, tools, harnesses or research strategies.

## From a better model to a better research method

In the LM task, model parameters are optimized during training. The Agent explores
training procedures: data choices, architectures, optimization methods and serving
implementations. The meta-optimization question concerns the research strategy
that discovers those procedures: which hypotheses to pursue, how to design
experiments, how to use failures, and when to change direction.

A trajectory preserves evidence of the path taken, including unsuccessful
attempts and corrections where captured. Future methods can use this experience
to improve the ability to find useful paths in subsequent research. The intended
recursive loop is:

```text
Research system → autonomous exploration → recorded trajectory
       ↑                                      │
       └──── improved research capability ←───┘
             future update methods and orchestration
```

Exploration and recording are implemented. The return path is future work.
Whether an update improves research capability must be tested. Controlled
comparisons and transfer to other tasks, budgets or starting points can help
distinguish a reusable research method from a successful individual solution.
Higher Candidate scores alone do not establish that the researcher became better
at finding improvements.

## The recording interface

The existing [research recording](recording.md) interface is the connection point
between the current environment and future meta-optimization methods:

- The event journal links actions, returned responses, transport publication,
  command executions, Candidate commits and incumbent declarations.
- Bounded payload capture, artifact identities and retention records preserve
  available evidence and make omissions explicit.
- `Supervisor.record_observation()` accepts optional trusted runner observations
  of tools, feedback or code changes, with source identities and explicit links.
- `envcore trajectory --episode PATH` exports recorded evidence, resource and
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
