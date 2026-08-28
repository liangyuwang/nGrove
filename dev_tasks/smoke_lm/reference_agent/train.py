"""Train the optional reference participant inside one Training Lease."""

import argparse
import contextlib
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
    if len(rows) < 40:
        raise RuntimeError("training split has too few full context windows")
    return torch.tensor(rows, dtype=torch.long)


@torch.inference_mode()
def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def local_validation_loss(model, rows, device):
    model.eval()
    losses = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        with autocast_context(device):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        losses.append(loss.float())
    model.train()
    return float(torch.stack(losses).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=55)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")
    if args.seconds <= 15:
        raise ValueError("seconds must leave enough time to save a candidate")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows("/benchmark/train/train.json", args.context_length)
    local_val = rows[::20].to(device)
    train_rows = torch.cat([rows[offset::20] for offset in range(1, 20)]).to(device)
    config = {
        "vocab_size": 257,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "dropout": args.dropout,
    }
    model = ByteTransformer(config).to(device)
    optimizer_options = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
    deadline = time.monotonic() + args.seconds - 8
    best_loss = local_validation_loss(model, local_val, device)
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    step = 0
    tokens = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        indices = torch.randint(
            train_rows.shape[0], (args.batch_size,), device=device
        )
        batch = train_rows[indices]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        tokens += inputs.numel()
        if step % 50 == 0:
            validation_loss = local_validation_loss(model, local_val, device)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
            if step == 50 or step % 250 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"step={step} train_loss={float(loss.detach()):.6f} "
                    f"local_val_nll={validation_loss:.6f} "
                    f"tokens_per_second={tokens / max(elapsed, 1e-6):.0f}",
                    flush=True,
                )

    model.load_state_dict(best_state)
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
        f"saved={args.output} steps={step} tokens={tokens} elapsed={elapsed:.2f} "
        f"best_local_val_nll={best_loss:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
