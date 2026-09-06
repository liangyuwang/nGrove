# Prediction output extension boundary

Training artifacts (Candidate code and model files) and prediction artifacts
produced during evaluation are separate contracts. The next-token release uses
two prediction files with a fixed logits shape; this is not a requirement for
every future evaluation task.

`autotrain_bench.prediction_output.PredictionOutputContract` is the trusted
extension point for prediction artifacts. It separates:

- `artifact_manifest()`: the final output format, which can be variable-length;
- `storage_manifest()`: the rule for reserving bounded runtime output storage;
- `storage_limits(request_metadata=..., task_metadata=...)`: concrete positive
  `max_bytes` and `max_inodes`, derived before Candidate execution;
- `validate_artifact(...)`: target-free final validation after confirmed
  Candidate termination, returning the actual number of output file-data bytes.

The Worker storage backend consumes only the generic byte/inode limits. It must
not infer capacity from vocabulary size, assume one prediction per context, or
assume two output files. Those choices belong to an individual adapter. Runtime
capacity and final artifact validity remain separate: bounded temporary files
can be allowed even when the final format is strict.

For variable-length generation, a future adapter can derive an upper bound from
the number of requests, a release-defined generation limit, and its encoding.
It need not know the exact resulting file size. An output contract without a
finite pre-execution bound cannot claim bounded storage protection. Limits are
trusted task declarations, not values chosen by an untrusted Candidate.

The only implemented adapter is `offline_next_token_logits_v3`. It delegates to
the existing helpers, preserving current formats and storage rules. The Docker
evaluation path resolves this adapter for storage sizing. Unknown protocol IDs
are rejected; task manifests cannot import arbitrary modules or register code.

This is an extension point, not support for a second evaluation protocol. A new
task must also implement its input schema, Candidate invocation, trusted scorer,
metric/cost semantics, and public declarations, and update the currently
next-token-specific TaskConfig/channel validation. Protocol/release identities
must change when externally observable semantics change. Existing releases
must keep their original meaning. Implementing an adapter alone must never
silently route a different task through the next-token evaluator.

The interface is Worker-side and target-free. Hidden targets remain exclusively
with trusted evaluation; research sealing, role isolation, termination, and
artifact cleanup requirements continue to apply to every output protocol.
