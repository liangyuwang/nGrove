"""Low-cost autonomous baseline for the real-text mini benchmark."""

from autotrain_bench.client import Client


def run_experiment(client, index, *, order, smoothing, max_contexts):
    client.start_lease(seconds=60)
    candidate_path = f"candidate-{index}"
    result = client.exec(
        [
            "python",
            "/workspace/train.py",
            "--order",
            str(order),
            "--smoothing",
            str(smoothing),
            "--max-contexts",
            str(max_contexts),
            "--output",
            f"/workspace/{candidate_path}",
        ]
    )
    if result["exit_code"] != 0:
        client.stop_lease()
        raise RuntimeError(result["stderr"] or "mini LM training failed")
    committed = client.commit(candidate_path)
    return {
        **committed,
        "order": order,
        "smoothing": smoothing,
        "max_contexts": max_contexts,
    }


def main():
    client = Client()
    candidates = []
    incumbent = None

    def record(candidate):
        nonlocal incumbent
        candidates.append(candidate)
        if candidate["val_nll"] is not None and (
            incumbent is None or candidate["val_nll"] < incumbent["val_nll"]
        ):
            client.declare_incumbent(candidate["candidate_id"])
            incumbent = candidate

    record(run_experiment(client, 1, order=2, smoothing=1.0, max_contexts=20_000))
    record(
        run_experiment(
            client, 2, order=3, smoothing=0.25, max_contexts=30_000
        )
    )

    valid = [candidate for candidate in candidates if candidate["val_nll"] is not None]
    if valid and min(valid, key=lambda candidate: candidate["val_nll"])["order"] >= 3:
        revision = {"order": 4, "smoothing": 0.1, "max_contexts": 40_000}
    else:
        revision = {"order": 2, "smoothing": 0.1, "max_contexts": 30_000}
    record(run_experiment(client, 3, **revision))

    valid = [candidate for candidate in candidates if candidate["val_nll"] is not None]
    if not valid:
        client.finish()
        return
    best = min(valid, key=lambda candidate: candidate["val_nll"])
    final = client.finish()
    print(
        "selected "
        f"order={best['order']} smoothing={best['smoothing']} "
        f"VAL_NLL={best['val_nll']:.6f} TEST_NLL={final['test_nll']:.6f}"
    )


if __name__ == "__main__":
    main()
