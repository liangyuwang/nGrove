"""Deterministic fallback Agent for the single-GPU smoke task."""

import json
import os
from pathlib import Path

from autotrain_bench.client import Client


def main():
    client = Client()
    public_task = json.loads(
        Path(os.environ["AUTOTRAIN_BENCH_TASK"]).read_text(encoding="utf-8")
    )
    context_length = int(public_task["evaluation"]["context_length"])
    candidates = []
    for index, (width, layers, learning_rate) in enumerate(
        ((192, 3, 4e-4), (256, 4, 3e-4), (320, 5, 2e-4)), start=1
    ):
        status = client.status()
        if status["status"] != "active":
            break
        if status["track"] == "A":
            remaining = status["remaining_wall_clock_seconds"]
            if remaining < 75:
                break
            requested = int(min(180, remaining - 45))
        else:
            if status["leases_remaining"] <= 0:
                break
            requested = min(180, int(status.get("remaining_wall_clock_seconds", 180)))
        lease = client.start_lease(requested)
        output = f"candidate-{index}"
        result = client.exec(
            [
                "python",
                "/workspace/train.py",
                "--output",
                f"/workspace/{output}",
                "--seconds",
                str(max(30, int(lease["duration_seconds"]) - 15)),
                "--context-length",
                str(context_length),
                "--d-model",
                str(width),
                "--layers",
                str(layers),
                "--heads",
                "4",
                "--learning-rate",
                str(learning_rate),
            ],
            timeout_seconds=max(30, lease["duration_seconds"] - 5),
        )
        if result["exit_code"] != 0:
            client.stop_lease()
            continue
        candidates.append(client.commit(output))

    valid = [candidate for candidate in candidates if candidate["val_nll"] is not None]
    best = min(valid, key=lambda candidate: candidate["val_nll"]) if valid else None
    client.finish(best["candidate_id"] if best else None)


if __name__ == "__main__":
    main()
