# TinyStories single-GPU smoke benchmark

This task reuses the pinned byte-level TinyStories sample from `mini_lm`, but
replaces the CPU n-gram starter with a small PyTorch causal Transformer. It is a
low-cost rehearsal of a real single-GPU release rather than a competitive model
quality benchmark.

Fixed resources:

- one CUDA-compatible NVIDIA GPU supplied by the user;
- pinned PyTorch 2.7.1 / CUDA 12.8 training and evaluator image;
- 257-token UTF-8 byte vocabulary and task-defined context length 128;
- batched hidden VAL/TEST evaluation on the supplied GPU;
- Track A: 600 seconds total wall clock;
- Track B: three Training Leases of at most 180 seconds each.

Prepare the dataset and fixed Docker images:

```bash
python3 examples/mini_lm/prepare_data.py
autotrain-bench prepare --task examples/gpu_lm/task.json
```

Then run either track through the same Docker reference runtime as every other
task:

```bash
autotrain-bench start --task examples/gpu_lm/task.json --track B
```

Docker and the NVIDIA Container Toolkit must already expose exactly one GPU to
the benchmark host. AutoTrain-Bench does not request a Slurm allocation, create
a Kubernetes workload, or provision a cloud instance. Those platform concerns
belong outside the core benchmark; once the machine is available, the same
Docker isolation contract applies.
