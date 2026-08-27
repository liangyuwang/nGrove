"""A deterministic example AutoTrain agent that uses two GPU-lease rounds."""

from autotrain_bench.client import Client


def main():
    client = Client()
    candidates = []
    for index, smoothing in enumerate((1.0, 0.1), start=1):
        client.start_lease()
        candidate_path = f"candidate-{index}"
        result = client.exec(
            [
                "python",
                "/workspace/train.py",
                "--smoothing",
                str(smoothing),
                "--output",
                f"/workspace/{candidate_path}",
            ]
        )
        if result["exit_code"] != 0:
            client.stop_lease()
            raise RuntimeError(result["stderr"] or "toy training failed")
        candidates.append(client.commit(candidate_path))

    valid = [candidate for candidate in candidates if candidate["val_nll"] is not None]
    if not valid:
        client.finish()
        return
    best = min(valid, key=lambda candidate: candidate["val_nll"])
    final = client.finish(best["candidate_id"])
    print(f"final TEST NLL: {final['test_nll']}")


if __name__ == "__main__":
    main()

