# LM Open World

> **Early vision, open for community design.** This directory currently contains
> only this README. The task and evaluation contract remain to be implemented.

This proposed task would provide a broader experimental setting for the project's
[RSI and meta-optimization scope](../../docs/research-scope.md). Research
trajectories and artifact measurements would support that work.

## Vision

**Constrain resources, not ideas. Define the evaluator, not the solution.**

LM Open World is a proposed research task for autonomous research on language tasks.
The Agent is free to choose how to achieve the task objective within the
available world and resource budget. Its central question is:

**How effectively can an autonomous research system use the resources available
to it to produce a capable, efficient solution to a language task?**

The participant supplies a research Agent. During an episode, that Agent builds
or selects the system to submit for independent evaluation. The research Agent
and its final submission may share components, but need not be identical.

The research task should leave the largest practical solution space open, defining
only what makes that freedom meaningful and evaluable:

| Component | What the task and environment should define |
| --- | --- |
| **Evaluator** | The task objective and how success and execution cost are measured |
| **Sandbox and policy** | The world the Agent may act in, including data, tools, services, and network access |
| **Resource budget** | The time, compute, storage, and external-service resources available to the system |
| **Minimal interface** | How a submitted system can be invoked and evaluated independently |

Within those boundaries, whether to train, which models and data to use, and how
to build infrastructure or improve the research process are the Agent's
decisions. The language focus defines the tasks we evaluate without prescribing
the system's architecture. A successful strategy the designers did not
anticipate is an intended outcome.

Every restriction should have a clear reason in the task objective, resource
budget, or enforceable boundary. Early versions may expose a smaller world;
their limitations should be explicit and open to revision.

## Beyond the LM pretrain preview

The current [LM pretrain preview](../lm-pretrain/README.md) already leaves model
architecture, training methods, data processing, and autonomous research choices
open within its [release policy](../lm-pretrain/POLICY.md). LM Open World aims to
extend that freedom to the inputs, capabilities, and system forms available to
the Agent. The following table and scenarios illustrate some possible directions:

| Area | Current LM pretrain preview | Direction to explore in LM Open World |
| --- | --- | --- |
| **Data** | Provided TRAIN data and policy-permitted synthetic data; extra training corpora are excluded | Discover and download permitted datasets; select, combine, or generate data |
| **Model weights** | Pretrained target-model checkpoints are excluded | Download and use existing weights, adapt or combine models, or train from scratch |
| **External capabilities** | No open network retrieval; remote researcher inference uses registered access | Controlled access to software packages, technical documentation, retrieval, and model services |
| **Evaluated system** | A Candidate implementing the next-token logits interface | A task-appropriate interface for systems that may include models, harnesses, tools, and memory |

For the same future objective, one Agent might train from scratch. Another might
download permitted weights and fine-tune them on newly acquired data. A third
might combine an existing model with retrieval and tools without updating any
weights. An Agent might also improve its own experiment scheduler, memory, or
research tools to use the remaining budget more effectively.

These examples are not exhaustive and do not prescribe a workflow. The task,
resource budget, and sandbox define the boundary of the solution space.

These freedoms remain proposals. This README does not enable new capabilities
or change the current runner's interfaces or policies. Each future release must
specify permitted access, account for resource use, and protect hidden
evaluation. Research and evaluation may have different access rules.

## From research to independent evaluation

The handoff should let an evaluator invoke the submitted system independently
under the task's resource and access rules. One direction is an executable
system with a declared entrypoint, dependencies, permitted capabilities, and
resource requirements. The interface should follow the evaluation objective
and accommodate different internal designs.

A submission could be a single model, a retrieval pipeline, or an Agent system
with its own harness, tools, and memory. It could combine several models or use
registered services. What must be packaged locally, and what can be supplied
through controlled services, remains a design question involving reproducibility,
availability, isolation, and accounting.

The initial research system, external access rules, and capability boundaries
should be registered before task reveal, with room for autonomous changes
during the episode. Those rules may permit the Agent to discover and acquire
resources, such as datasets or model weights, during research; the identities
of acquired artifacts should be recorded as they are obtained.

The submitted artifacts and declared initial evaluation state are frozen when
research ends. If the evaluation task permits interaction, memory updates, or
adaptation during execution, its contract must define the allowed changes,
state resets, and resource accounting.

## Freedom within a reliable boundary

Research should remain autonomous after task reveal. Hidden evaluation content
and results must stay outside the research environment. Submitted artifacts and
declared initial evaluation state must be frozen, and research workloads and
control capabilities confirmed terminated, before hidden evaluation begins.

The sandbox should define the available world, including any permitted data,
tools, and network access. Workers should enforce that boundary and its resource
limits. Research and evaluation semantics should remain platform-independent.

Opening access to existing models and services also makes the interpretation of
results central to the design. Final quality and execution cost describe the
submitted system under its evaluation conditions. Claims about research
effectiveness also need evidence of the Agent's starting point, the resources it
could access, and how it used its budget. Choosing an effective existing system
is a valid research decision; comparisons should make those conditions visible.

## Help shape the research task

The first priority is to identify a language task that makes this broader
freedom useful and measurable. We invite proposals around four connected
questions:

- **Task and success:** What should the first task measure, and which
  substantially different approaches should it admit? How should quality and
  execution cost be measured?
- **Resources and comparison:** How should research and evaluation budgets cover
  local compute, model services, storage, and time? Which conditions must match
  for a comparison, and how should differences in initial systems and access to
  existing models or services be reported?
- **The available world:** Which data, models, tools, and services should be
  accessible? How can downloads and service calls remain reproducible and
  accountable without exposing hidden evaluation content?
- **Handoff and state:** What is the smallest interface that lets an evaluator
  run these systems independently? What may change during research or
  evaluation, and when must state be frozen or reset?

A useful task proposal describes the objective, observable inputs and outputs,
permitted resources, and at least two different approaches it should admit.
Examples of systems an interface would exclude, and small evaluation experiments,
are also welcome. Please open an issue or propose a README change.

No reference Agent, fixed harness, or model implementation is provided here.
