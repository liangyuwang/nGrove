# Inference Gateway execution evidence

Core can record a trusted launcher's observation of the actual inference
Gateway before task reveal. This addresses the Core interface in original
issue 12. It does not certify a deployment, add a launcher backend, or change
Worker Protocol v16. Gateway lifecycle remains owned by the trusted composition.

## Release identity and actual execution

For formal registered inference, the organizer supplies
`ReleaseVerificationReceipt.inference_gateway_artifact` as a
`GatewayArtifactIdentity(format, sha256)`. This is independent verification of
the complete executable artifact selected for that release. The format defines
the hashing procedure; Core compares exact format and digest values, without
loading a platform artifact or interpreting deployment handles.

The identity must cover every executable input: the runtime environment,
dependencies, entrypoint and any injected implementation code. A deployment
may define a canonical composite manifest when these are separate artifacts.
The organizer must verify that manifest and its components independently of
the launcher's claim. A configured image reference, a base-image digest, or a
label naming a Core revision is insufficient when other code is injected.
Changes to the executable bundle require a newly approved artifact identity.

After construction, `Supervisor.gateway_execution_request` is a frozen
`GatewayExecutionRequest`, or `None` when inference is disabled. It binds:

- The fresh semantic episode ID.
- The task-selected Gateway workload reference and observed Core revision.
- A SHA-256 fingerprint of the registered protocol, upstream origin, credential
  variable **name**, allowed models, allowed paths, and request timeout.
- A separate fingerprint of the reportable protocol/model/path policy so the
  result reader can check those fields without storing the private origin.

The fingerprint contains no credential value and is private Operator evidence.
Models and paths are sorted before hashing; JSON attribute order does not
change request identity. The launcher receives the actual frozen inference
configuration separately through its trusted composition, never from the Agent.

## Launcher sequence

1. Construct the Supervisor with the verified release and other frozen inputs.
2. Obtain `supervisor.gateway_execution_request`. Register ownership and bounded
   cleanup before attempting to create the Gateway.
3. Start the Gateway with that request's workload and inference registration.
   Confirm readiness and inspect the actual running artifact. Preserve the
   underlying deployment observations with the receipt's unique ID.
4. Return a `GatewayExecutionReceipt` containing `receipt_id`, the request's
   `sha256`, `source`, observed `artifact`, `enforcement`, `trust_level`, and
   explicit `limitations`. Call `supervisor.record_gateway_execution(receipt)`.
5. Call `supervisor.begin()` immediately before starting the Tested Agent.
   Enforce the fixed Gateway artifact and registration for the execution
   lifetime, and include it in normal, failed-start and interrupted cleanup.

`record_gateway_execution` is a trusted Python composition interface, not an
Agent request or a CLI option for uploading an arbitrary JSON assertion. The
receipt must describe a Gateway that has actually started; resolving an image
or passing a preflight is insufficient. The launcher authenticates and retains
the underlying evidence. Core does not provide signatures, remote attestation,
or a certificate authority.

Registration is allowed once while the episode is `created`. A receipt for
another request is rejected, including reuse across episodes or a change of
inference registration. Duplicate registration and registration after `begin()`
are rejected; missing evidence cannot be repaired retroactively to upgrade an
episode's eligibility. Startup failure still requires deployment cleanup.

## Eligibility and reporting

Formal eligibility requires the observed artifact to match the organizer's
release artifact, `enforcement=ENFORCED`, `trust_level=CERTIFIED`, and no
limitations. Enforcement means the selected artifact and inference registration
remain enforced throughout execution. Certification must come from the
organizer's trusted launcher boundary; a mechanism name or self-description
does not establish it. Worker conformance and all other formal checks remain
independent requirements.

Missing or insufficient evidence keeps the run ineligible using these codes:

| Code | Meaning |
| --- | --- |
| `inference_gateway_workload_evidence_missing` | No actual execution receipt |
| `inference_gateway_release_artifact_missing` | No organizer-verified complete artifact |
| `inference_gateway_artifact_mismatch` | Observed format or digest differs from the approved artifact |
| `inference_gateway_execution_not_certified` | Insufficient enforcement, trust, or unresolved limitations |

These are eligibility findings. They do not blame a Candidate, erase an
otherwise accepted TEST measurement, or stand in for deployment teardown
failures. A malformed or misbound receipt is rejected at registration.

Result schema v15 stores `gateway_execution` in the private state/report:
`null` when disabled, otherwise a request and a nullable receipt. The reader
revalidates request binding and checks the evidence and release artifact before
accepting an eligible result. The Agent's public task/status contains neither
the receipt nor the private registration fingerprint. Historical v12/v13/v14
reports remain readable under their original semantics and cannot contain the
new evidence fields.

The bundled local launcher remains a development composition and does not
issue this receipt. In particular, its Gateway uses a base image plus a mounted
Core package; the image alone cannot certify the executable bundle or enforce
immutability of the host source. Formal inference therefore continues to fail
closed in that composition. Unit and fake-launcher integration tests establish
the Core contract; deployment certification and real execution acceptance
remain separate work.
