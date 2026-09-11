from benchcore.candidate_protocol import (
    load_context_request,
    write_prediction_artifact,
)


def run_offline(candidate_dir, contexts_path, output_dir, device):
    del candidate_dir, device
    request = load_context_request(contexts_path)
    write_prediction_artifact(
        output_dir,
        dataset_id=request["dataset_id"],
        logits=([0.0] * request["vocab_size"] for _ in request["examples"]),
        examples=len(request["examples"]),
        vocab_size=request["vocab_size"],
    )
