"""Deterministic baseline Agent for the comparable mini benchmark."""

import json
import os
from pathlib import Path

from autotrain_bench.client import Client


def main():
    client = Client()
    public_task = json.loads(
        Path(os.environ["AUTOTRAIN_BENCH_TASK"]).read_text(encoding="utf-8")
    )
    evaluation = public_task["evaluation"]
    incumbent = None
    for index, (width, layers, learning_rate) in enumerate(
        ((192, 3, 4e-4), (256, 4, 3e-4), (320, 5, 2e-4)), start=1
    ):
        status = client.status()
        if status["status"] != "active":
            break
        if status["track"] == "A":
            remaining = status["remaining_wall_clock_seconds"]
            if remaining < 55:
                break
            requested = int(min(75, remaining - 30))
        else:
            if status["leases_remaining"] <= 0:
                break
            requested = 75

        lease = client.start_lease(requested)
        output = f"/output/candidate-{index}"
        result = client.exec(
            [
                "python",
                "/workspace/train.py",
                "--output",
                output,
                "--seconds",
                str(max(25, int(lease["duration_seconds"]) - 20)),
                "--context-length",
                str(evaluation["context_length"]),
                "--d-model",
                str(width),
                "--layers",
                str(layers),
                "--heads",
                "4",
                "--learning-rate",
                str(learning_rate),
            ],
            timeout_seconds=max(30, lease["duration_seconds"] - 12),
        )
        if result["exit_code"] != 0:
            client.stop_lease()
            continue

        preflight = client.exec(
            [
                "python",
                "-m",
                "autotrain_bench.candidate_preflight",
                "--candidate",
                output,
                "--device",
                evaluation["device"],
                "--vocab-size",
                str(evaluation["vocab_size"]),
                "--context-length",
                str(evaluation["context_length"]),
            ],
            timeout_seconds=10,
        )
        if preflight["exit_code"] != 0:
            client.stop_lease()
            continue

        candidate = client.commit(output)
        if candidate["val_nll"] is not None and (
            incumbent is None or candidate["val_nll"] < incumbent["val_nll"]
        ):
            client.declare_incumbent(candidate["candidate_id"])
            incumbent = candidate

    client.finish()


if __name__ == "__main__":
    main()
