# Training-research preview policy

The research protocol freezes the submitted autonomous system and its external boundary
at task reveal. Within that boundary, autonomous task-local improvement is
allowed: the Agent may change its research process, code, tools, memory, model,
training stack, data policy, and serving implementation.

Synthetic data produced autonomously during the episode by the frozen initial
system through registered capabilities is allowed. Human intervention after
task reveal, unregistered network access, additional mounted data, and an
initial workspace or image containing extra training corpora or a pretrained
target-model checkpoint are outside policy.

The **user/operator** prepares and launches the run and may itself be a human or
another Agent; it is part of the trusted host boundary. The **Research Agent** is
the submitted autonomous system conducting research. The runtime isolates the
Research Agent from prohibited network and evaluator resources where practical.
It records immutable inputs and execution provenance for reproducibility and
audit, but does not try to prevent a user/operator or administrator from
modifying its own host. The run records its actual execution boundary and enforcement limitations.
