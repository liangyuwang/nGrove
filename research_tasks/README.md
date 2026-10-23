# Research tasks

This directory contains nGrove's research tasks.
`lm-pretrain/` is the first runnable preview task; `lm-open-world/` is a design
proposal. These tasks provide experimental settings for the project's
[RSI and meta-optimization scope](../docs/research-scope.md). The pretraining
preview generates its data locally and its release settings remain provisional.

A research task owns its dataset construction, evaluation splits, public metric
and Candidate ABI, task images, default budgets, artifact limits, and task
policy. Run-owned environment configurations can select compute resources and
override budgets before research begins. Participant Agents, starter workspaces,
and development fixtures do not belong here.

During research, the Agent decides whether and how to validate using permitted
information. Evaluation takes place after research sealing and
confirmed teardown. TEST measures the final declared Candidate; the existing
postmortem ANALYSIS measures committed Candidates. These are artifact measurements,
not an evaluation of how good the research trajectory is.

## Research workflow

Choose a task, a Track and a user-owned environment configuration; the environment
freezes the effective compute resources and time budget. The
[LM pretraining run example](lm-pretrain/README.md#run-a-research-episode) uses
`--task`, `--track` and `--environment` directly. An explicitly selected
[development smoke task](../dev_tasks/smoke_lm-pretrain/README.md) provides a smaller
integration check.

Every run records available research evidence. Use `--recording` to configure
capture and artifact retention, and `ngrove trajectory --episode PATH` to
inspect the private export. See [research recording](../docs/recording.md) for
capture limitations and the interface for future meta-optimization methods.

## Development checks

Use `dev_tasks/smoke_lm-pretrain/` for lightweight end-to-end integration. It uses
the same research lifecycle, recording and trusted Evaluator implementation with
smaller data and budgets.
