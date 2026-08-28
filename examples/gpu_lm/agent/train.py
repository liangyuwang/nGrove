"""Train the starter byte-level Transformer inside a GPU Training Lease."""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import ByteTransformer


def load_rows(path, context_length):
    sequences = json.loads(Path(path).read_text(encoding="utf-8"))["sequences"]
    rows = [row[: context_length + 1] for row in sequences if len(row) > context_length]
    if not rows:
        raise RuntimeError("training split has no full context windows")
    return torch.tensor(rows, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")
    if args.seconds <= 15:
        raise ValueError("seconds must leave enough time to save a candidate")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows = load_rows("/benchmark/train/train.json", args.context_length).to(device)
    config = {
        "vocab_size": 257,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "dropout": args.dropout,
    }
    model = ByteTransformer(config).to(device)
    train_model = torch.compile(model) if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    deadline = time.monotonic() + args.seconds - 10
    step = 0
    tokens = 0
    last_loss = float("nan")
    started = time.monotonic()
    while time.monotonic() < deadline:
        indices = torch.randint(rows.shape[0], (args.batch_size,), device=device)
        batch = rows[indices]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = train_model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        tokens += inputs.numel()
        last_loss = float(loss.detach())
        if step == 1 or step % 25 == 0:
            elapsed = time.monotonic() - started
            print(
                f"step={step} loss={last_loss:.6f} "
                f"tokens_per_second={tokens / max(elapsed, 1e-6):.0f}",
                flush=True,
            )

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(Path("/workspace/model.py"), args.output / "model.py")
    shutil.copy2(Path("/workspace/candidate.py"), args.output / "candidate.py")
    elapsed = time.monotonic() - started
    print(
        f"saved={args.output} steps={step} tokens={tokens} "
        f"elapsed={elapsed:.2f} final_loss={last_loss:.6f} "
        f"tokens_per_second={tokens / max(elapsed, 1e-6):.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
