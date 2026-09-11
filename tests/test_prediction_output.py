import tempfile
import unittest
from pathlib import Path

from benchcore.candidate_protocol import (
    CANDIDATE_PROTOCOL,
    CandidateProtocolError,
    write_prediction_artifact,
)
from benchcore.prediction_output import prediction_output_contract


class PredictionOutputContractTests(unittest.TestCase):
    def test_unknown_protocol_never_falls_back_to_next_token(self):
        for protocol in ("", "variable_text_future_v1", "offline_next_token_logits_v2"):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    prediction_output_contract(protocol)

    def test_next_token_budget_uses_trusted_shape_and_allows_runtime_headroom(self):
        contract = prediction_output_contract(CANDIDATE_PROTOCOL)
        limits = contract.storage_limits(
            request_metadata={"examples": 512},
            task_metadata={"vocab_size": 257},
        )
        self.assertEqual(limits["logits_bytes"], 526336)
        self.assertEqual(limits["artifact_max_bytes"], 591872)
        self.assertGreater(limits["max_bytes"], limits["artifact_max_bytes"])
        self.assertGreater(limits["max_inodes"], limits["artifact_max_entries"])
        self.assertEqual(
            contract.artifact_manifest()["logits_artifact"]["filename"], "logits.f32"
        )
        for examples in (True, 0, -1, "512"):
            with self.subTest(examples=examples):
                with self.assertRaises(ValueError):
                    contract.storage_limits(
                        request_metadata={"examples": examples},
                        task_metadata={"vocab_size": 257},
                    )

    def test_final_validation_binds_request_and_rejects_extra_files_and_bad_shape(self):
        contract = prediction_output_contract(CANDIDATE_PROTOCOL)
        metadata = {"dataset_id": "request-1", "examples": 2}
        task = {"vocab_size": 3}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_prediction_artifact(
                output,
                dataset_id=metadata["dataset_id"],
                logits=[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]],
                examples=2,
                vocab_size=3,
            )
            self.assertEqual(
                contract.validate_artifact(
                    output, request_metadata=metadata, task_metadata=task
                ),
                sum(path.stat().st_size for path in output.iterdir()),
            )
            with self.assertRaises(CandidateProtocolError):
                contract.validate_artifact(
                    output,
                    request_metadata={**metadata, "dataset_id": "other-request"},
                    task_metadata=task,
                )
            extra = output / "leftover.tmp"
            extra.touch()
            with self.assertRaises(CandidateProtocolError):
                contract.validate_artifact(
                    output, request_metadata=metadata, task_metadata=task
                )
            extra.unlink()
            (output / "logits.f32").write_bytes(b"short")
            with self.assertRaises(CandidateProtocolError):
                contract.validate_artifact(
                    output, request_metadata=metadata, task_metadata=task
                )


if __name__ == "__main__":
    unittest.main()
