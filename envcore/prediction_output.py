"""Trusted task adapters for prediction artifacts and their storage budgets.

The interface does not assume tokens, a fixed output shape, or two files.
Adapters are selected by trusted protocol identity, never from Candidate code.
Adding an adapter does not by itself enable a new evaluation protocol.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Dict, Mapping, Protocol, Union

from . import candidate_protocol as next_token


class PredictionOutputContract(Protocol):
    """Output semantics owned by an evaluation protocol, not a container runtime.

    Metadata comes from trusted input validation and the frozen task. It must
    not contain hidden targets. Variable-length protocols must derive a finite
    storage bound before Candidate startup; actual output can be smaller.
    """

    def artifact_manifest(self) -> Dict[str, Any]:
        """Describe final output structure without requiring a fixed byte size."""
        ...

    def storage_manifest(self) -> Dict[str, Any]:
        """Describe the protocol's rule for deriving its runtime storage budget."""
        ...

    def storage_limits(
        self,
        *,
        request_metadata: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
    ) -> Mapping[str, int]:
        """Return positive max_bytes/max_inodes, optionally with artifact bounds.

        Storage backends consume only max_bytes and max_inodes. Additional keys
        describe this adapter's artifact and must not drive backend decisions.
        """
        ...

    def validate_artifact(
        self,
        output_dir: Union[str, PathLike],
        *,
        request_metadata: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
    ) -> int:
        """Validate frozen output without targets; return its file-data bytes.

        Call only after confirmed Candidate termination. Scoring and metric
        computation remain the trusted evaluator's separate responsibility.
        """
        ...


class _NextTokenLogitsOutput:
    """Compatibility adapter for the existing offline next-token ABI."""

    def artifact_manifest(self) -> Dict[str, Any]:
        return next_token.prediction_manifest_contract()

    def storage_manifest(self) -> Dict[str, Any]:
        return next_token.prediction_storage_manifest()

    def storage_limits(
        self,
        *,
        request_metadata: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
    ) -> Mapping[str, int]:
        return next_token.prediction_storage_limit(
            examples=request_metadata["examples"],
            vocab_size=task_metadata["vocab_size"],
        )

    def validate_artifact(
        self,
        output_dir: Union[str, PathLike],
        *,
        request_metadata: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
    ) -> int:
        return next_token.validate_prediction_artifact(
            output_dir,
            dataset_id=request_metadata["dataset_id"],
            examples=request_metadata["examples"],
            vocab_size=task_metadata["vocab_size"],
        )


def prediction_output_contract(protocol: str) -> PredictionOutputContract:
    """Resolve only implemented, trusted adapters; unknown protocols fail closed."""

    if protocol == next_token.CANDIDATE_PROTOCOL:
        return _NextTokenLogitsOutput()
    raise ValueError(f"unsupported prediction output protocol: {protocol!r}")
