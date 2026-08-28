"""A deterministic example AutoTrain agent that uses two GPU-lease rounds."""

from autotrain_bench.client import Client


def main():
    client = Client()
    candidates = []
    incumbent = None
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
        candidate = client.commit(candidate_path)
        candidates.append(candidate)
        if candidate["val_nll"] is not None and (
            incumbent is None or candidate["val_nll"] < incumbent["val_nll"]
        ):
            client.declare_incumbent(candidate["candidate_id"])
            incumbent = candidate

    valid = [candidate for candidate in candidates if candidate["val_nll"] is not None]
    if not valid:
        client.finish()
        return
    final = client.finish()
    print(f"final TEST NLL: {final['test_nll']}")


if __name__ == "__main__":
    main()
