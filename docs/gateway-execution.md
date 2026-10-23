# Inference Gateway execution evidence

Core can record a trusted launcher's observation of the actual inference
Gateway before task reveal. Gateway lifecycle remains owned by the trusted
composition.

## Actual execution

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

1. Construct the Supervisor with the frozen task and environment inputs.
2. Obtain `supervisor.gateway_execution_request`. Register ownership and bounded
   cleanup before attempting to create the Gateway.
3. Start the Gateway with that request's workload and inference registration.
   Confirm readiness and inspect the actual running artifact. Preserve the
   underlying deployment observations with the receipt's unique ID.
4. Return a `GatewayExecutionReceipt` containing `receipt_id`, the request's
   `sha256`, `source`, observed `artifact`, `enforcement`, `trust_level`, and
   explicit `limitations`. Call `supervisor.record_gateway_execution(receipt)`.
5. Call `supervisor.begin()` immediately before starting the Research Agent.
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
are rejected. Startup failure still requires deployment cleanup.

Result schema v15 stores `gateway_execution` in the private state/report:
`null` when disabled, otherwise a request and a nullable receipt. The reader
revalidates request binding. The Agent's public task/status contains neither
the receipt nor the private registration fingerprint. Historical v12/v13/v14
reports remain readable under their original semantics and cannot contain the
new evidence fields.
