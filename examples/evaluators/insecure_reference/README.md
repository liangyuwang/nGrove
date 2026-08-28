# Insecure reference evaluator

This evaluator exists only for local examples, unit tests, and Candidate ABI
development. It imports participant-supplied `candidate.py` in the same process
that can read the mounted evaluation split. A malicious candidate can therefore
inspect hidden VAL or TEST content directly.

Tasks must select it explicitly with:

```json
{
  "evaluation": {
    "backend": "insecure_reference",
    "entrypoint": "../evaluators/insecure_reference/evaluator.py"
  }
}
```

Every episode using this backend is reported as `policy_compliant: false` with
the violation `insecure_reference_evaluator`. It must never be used for an
official benchmark result or leaderboard submission.

A compliant evaluator must keep hidden data in a trusted process and execute
the untrusted candidate behind a separate, ephemeral serving boundary.
