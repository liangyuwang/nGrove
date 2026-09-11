# Worker acceptance evidence and remaining scenarios

This is the evidence matrix promised by step 4 of the preview repair plan:
minimum acceptance for original issues **2, 3, 4, 5, 6 and 8**. It connects
current implementation checks to external Worker observations and identifies the
remaining deployment tests. It does not issue conformance receipts or change
capability, trust, policy or leaderboard eligibility.

Reviewed code baseline: `70b8585df503375e15c0c6132abb8e06ab1365bd`;
Worker Protocol v16, Task v7, Challenge v3, Result v14. The last recorded local
full run at this code baseline passed 564 tests. Subsequent documentation-only
changes do not supply new runtime evidence. Reassess affected rows when code,
Worker source, runtime, launch configuration or fixtures change.

Use this matrix on the user's own authorized execution environment. Core defines
the common Worker contract; the deployment supplies its platform integration.
No access to a particular test site is required, and Core does not bundle an
adapter for every platform. Historical runs are evidence for their recorded
configurations; this matrix does not require repeating them on the same sites.

## Reading the evidence

| Evidence category | What it establishes | What it does not establish |
| --- | --- | --- |
| Local logic/command test | Core decisions, generated runtime arguments and handling of injected responses | Actual container limits, device isolation or physical resource cleanup |
| Local process test | Observed pipes, signals, process-group fencing or filesystem/lock behavior in the local fixture | Real Docker/scheduler behavior; several process fixtures still use a fake backend |
| Historical deployment observation | The stated scenario on the recorded Core/Worker/runtime combination | Acceptance of a later implementation or another deployment |
| Current deployment acceptance | A completed, scoped real-runtime test with identities, observations and cleanup evidence | Unexercised scenarios, other releases, or formal certification |

**Current deployment acceptance is recorded as pending for every remaining
scenario below.** The repository tests and supplied historical records do not
contain a real Worker v16 acceptance run for these changes. This is an evidence
gap, not a claim that all implementations are absent or all historical tests
must repeat.

Status is per scenario and deployment: `pending`, `passed`, `failed`, or
`not_applicable` with a reason. A missing, stale, skipped or unconfirmed check is
not a pass. A CPU-only diagnostic can mark a GPU case inapplicable to that
diagnostic; it cannot close the GPU guarantee for a CUDA release.

## Historical evidence catalog

These observations come from independent development deployments using Worker
v15. They identify useful regression cases without requiring users to know or
access those environments. Site names, job IDs, source snapshots, archive hashes
and access details stay in the Operator's external records. Consult the exact
implementation identity and scope there before reusing historical evidence;
observations from different deployments must not be combined into one deployment's
acceptance claim.

| ID | Observed case | Recorded evidence and limit |
| --- | --- | --- |
| H-FAULT | Targeted resource and failure diagnostics | Actual byte/file-entry exhaustion, open-but-unlinked storage accounting, paused Worker deadline enforcement, independent cleanup after forced termination, separate prediction storage/final validation and accelerator identity checks. These cover the tested implementation and cases only. |
| H-COMPLETE | Complete preview episodes | Successful Track A and Track B runs across separate deployments, including Candidate/incumbent, sealing, research/control teardown, TEST then ANALYSIS, publication, archive verification and working-storage disposal. Normal completion did not exercise private-limit expiry or every failure path. |
| H-ADMISSION | Execution identity and tool admission | One deployment's final identity/path/tool checks and subsequent normal episode passed. This does not certify another wrapper or all storage/termination/isolation cases. |
| H-INTERRUPTED | External cancellation | An episode was cancelled during research without Candidate/TEST or a cancellation-time Worker teardown receipt. Later archive and working-storage disposal completed. Storage disposal does not by itself prove workload absence or full recovery. |
| H-MON | Logs, monitoring and tool compatibility | Transient-heartbeat fixes have targeted regressions; short-log delay and stale final summaries were observed. A file-tool/path issue remained unresolved in one wrapper. A fix in another wrapper does not establish its resolution. |

## Coverage matrix

Test links refer to the reviewed baseline. Each row's remaining scenario is
defined in the next section; the IDs are local documentation labels, not a new
Worker protocol or a second list of Core defects.

| Scenario / original issue | Current implementation and local evidence | Historical deployment evidence | Remaining acceptance |
| --- | --- | --- | --- |
| R02 / 2: private research protection | Private protection, independent clocks and late-ack rejection. [Supervisor tests](../tests/test_supervisor.py): `test_operator_limit_aborts_at_boundary_without_hidden_evaluation`, `test_operator_limit_includes_refunded_startup_and_rejects_late_ack`; [watchdog tests](../tests/test_local_watchdog.py) exercise independent deadlines with real local processes and fixture cleanup. | H-COMPLETE completed before expiry; H-INTERRUPTED was cancelled externally. Their allocation/protection clocks were recorded separately. | Exercise expiry while idle and while a Worker call is blocked, including failed termination. Demonstrate the actual deployment's reservation/evaluation/cleanup budget handling. |
| R03 / 3: bounded writable storage | Workspace/storage admission and independent prediction budget/final checks. [Storage tests](../tests/test_local_storage.py) simulate mount/capacity admission; [prediction tests](../tests/test_prediction_output.py) check artifact rules; [runtime tests](../tests/test_runtime.py) capture runtime boundaries. | H-FAULT records actual byte/inode ENOSPC and open-unlinked accounting. H-ADMISSION records bounded storage and normal use, not all exhaustion paths. | Exercise physical byte/inode exhaustion on each applicable writable allocation, isolation of protected state, and recovery. |
| R04 / 4: independent deadlines and cancellation | [Watchdog tests](../tests/test_local_watchdog.py): `test_paused_or_killed_worker_is_fenced_and_recovered`, `test_parent_signals_do_not_interrupt_fencing_or_recovery`. Real local processes/signals; runtime cleanup remains a fixture. | H-FAULT records SIGSTOP/SIGKILL behavior. H-MON records separate external heartbeat fixes. H-INTERRUPTED lacks the cancellation-time teardown receipt. | Test the selected current launcher's operation/Lease/protection deadlines and cancellation during launch, execution and cleanup; test recovery after launcher loss separately. |
| R05 / 5: owned-resource recovery | Durable journal, shared allowance, workload-first cleanup, durable removal progress and supervised I/O. [Recovery tests](../tests/test_docker_recovery.py) include `test_same_budget_recovery_retries_progress_through_workloads_and_storage`; [process tests](../tests/test_local_recovery.py) include `test_filesystem_stalls_are_fenced_and_recovery_can_be_retried`. Backend is simulated, with real local process/lock checks. | H-FAULT has independent cleanup evidence; H-COMPLETE have normal teardown and disposal. H-INTERRUPTED has later archive/disposal only. | Prove current runtime ownership/fencing, cleanup ordering, bounded failure and same-budget retry progress; separately record workloads, private storage and retained-artifact disposal. |
| R06 / 6: exact GPU grant | Frozen UUID selection, workload validation for Training and configuration checks for CUDA Candidate. [GPU tests](../tests/test_docker_gpu.py) simulate inventories/grants; [evaluation tests](../tests/test_evaluation_resources.py) check actual-config response handling. Evidence remains `best_effort`. | H-FAULT records actual CUDA/device identity. H-ADMISSION records its allocation/visible devices; neither proves all heterogeneous or negative-access guarantees for current code. | Verify exact granted/visible devices and controlled negative access on the selected hardware, plus rejection of disappeared/mismatched devices. |
| R08-E / 8, bridge to 7: evaluation resources | v16 operation receipts, preflight and [evaluation probe](worker-evaluation-probe.md). [Resource](../tests/test_evaluation_resources.py), [preflight](../tests/test_evaluation_preflight.py) and [probe tests](../tests/test_evaluation_probe.py) use mocks/fixture Workers. | H-COMPLETE measured cost under v15 without the new evaluation receipt. | Run actual v16 supported/rejected contracts, inspect the workload allocation, and exercise Supervisor classification/report validation. The probe alone cannot establish all three. |
| R08-L / 8: lifecycle, identity and publication | [Supervisor tests](../tests/test_supervisor.py) cover delayed hidden materialization, teardown gates, TEST-first decisions, failed cleanup and atomic publication. These use fixture Workers and local files. | H-COMPLETE have complete normal sequences. H-INTERRUPTED supplies interruption evidence only. | Exercise failed research teardown, hidden-resolution boundaries, ANALYSIS-only errors and failed final cleanup/publication in a disposable integration harness. |
| R08-O / 8 and new observations: logs, status and tools | [Host log tests](../tests/test_host_agent.py) use real child pipes plus injected storage errors. [Operator status tests](../tests/test_operator_status.py) use local records; [integration guidance](worker-integration.md) covers external tool/monitor wiring. | H-MON records the log defect and stale summaries; one wrapper passed tool admission, while another retained an unresolved file-tool issue. | Verify the current deployment publishes live logs/capture loss and correct final/stale status; retest affected native tools and heartbeat handling in their actual view. |

## Scenarios and passing criteria

The trusted Operator selects the applicable cases and the deployment maintainer
provides the runtime mechanisms. A temporary Worker created during preparation
is valid: freeze its source/dependencies and launcher configuration before the
test. Use public synthetic fixtures, dedicated owned resources, independent
observation and a bounded outer watchdog/recovery path. Define each timeout,
termination grace, recovery allowance and retry cap before dispatch; their sum
must fit the authorized reservation. Platform-specific commands stay in that
deployment, not in Core or this matrix.

For an injected failure, correctly reporting an unknown boundary can pass that
decision subcheck. The attempt's required cleanup remains pending or failed
until absence is confirmed; do not mark the whole case passed while leaving
owned workloads or private storage unconfirmed.

### R02: private research limit and resource clocks

Use a short explicitly configured diagnostic limit; cover idle research,
blocked Worker execution and a late startup acknowledgement. Also let a fixture
finish normally before expiry and delay its postmortem work past that instant.
Pass when expiry seals research, prevents further admission and hidden
evaluation, and produces `operator_stopped / aborted` with
`operator_time_limit_reached` when Core can still record the transition. After
Core loss, preserve the interrupted snapshot and external stop evidence instead.
Confirm termination independently; a cleanup error must retain the primary
cause and the unresolved handles. Normal pre-expiry finish must retain its TEST
path under the separate evaluation/cleanup allowance.

Record allocation start/end, private research start/deadline, Track accounting,
operation deadlines and cleanup grace separately. This implementation is not a
universal whole-episode wall-clock cap: enforcing the total reservation belongs
to the deployment. The check must neither expose the private limit to the Agent
nor change Track A refunds or Track B Lease counts/durations.

### R03: bytes, inodes and independent prediction storage

In the disposable allocation, separately exhaust physical space and file
entries; retain an open file after unlinking it to verify it still consumes
capacity. Cover workspace/control/scratch/Lease output and runtime state where
the deployment admits writes, plus the separate prediction allocation. Verify
that the workload and trusted observer see the same admitted boundary and that
protected journal/other-run storage remains usable.

Prediction runtime temporary files are allowed within the current v15/v16
budget. Final output must still satisfy the exact file/shape/size/identity
contract; do not restore the obsolete two-files-only runtime rule. Keep workload
over-quota/output errors distinct from Worker storage failures and unconfirmed
teardown, using the applicable Core failure path. Pass only with bounded usage,
the expected disposition and confirmed cleanup or correctly retained failed
handles followed by a successful authorized recovery check.

### R04: stalled calls, signals and loss of the launcher

Exercise an operation that exceeds its allowance and a stopped/nonresponsive
Worker; independently check Lease and private research deadlines. Deliver a
handled cancellation during launch, active work and cleanup. Verify that a
fresh heartbeat cannot extend an expired deadline and transient unreadable
heartbeats neither reset the deadline nor immediately erase the last valid
observation. Pass when the independent controller stops admission and fences
owned work within the configured allowances, preserves the trigger and reports
cleanup confirmation or uncertainty accurately.

Whole-allocation or host loss requires an outside observer/recovery mechanism;
a parent in the same lost allocation cannot prove this case. Test those faults
only where explicitly in scope. A deployment without that guarantee records the
limitation. Full remote operation IDs, cancellation/reconciliation and atomic
activation remain separately deferred capabilities.

### R05: durable recovery, ownership and retry progress

Use only disposable resources created through the deployment's ownership
mechanism. Cover Worker loss after creation, an uncertain create response, lost
removal acknowledgement, failed workload removal and interrupted storage cleanup.
Use owned decoys to test wrong backend/identity and reused names; recovery must
not remove a foreign replacement or proceed while creation remains unfenced.

Include removed history before live resources and retry with the **same** small
allowance and a fixed retry cap. Record actual backend operations and durable
resource states on every pass. Pass when confirmed progress survives restart,
historical records cannot starve live work, and workloads stop before storage
is reclaimed. Each pass must return within its budget plus the declared fencing
grace. Confirmed removals remain done; uncertain acknowledgements remain subject
to ownership verification. Do not edit production journals to manufacture this
fixture.

For the bundled Worker, also inject stalls at journal open/save, private
deletion/close and recovery-report writing in a controlled diagnostic environment.
The supervising process must remain bounded, preserve the previous report where
applicable and retain locks/uncertainty when helper termination is unconfirmed.
Use [the documented recovery entry point](worker-recovery.md), not the
cooperative primitive as proof of independent supervision. External Workers
need their own equivalent evidence; Docker journal tests do not certify them.
Recovery must leave sealed research, saved TEST decisions, Candidate identities
and publication history unchanged. Archive verification and authorized working
storage disposal remain separate observations.

### R06: granted and workload-visible devices

Within an authorized dedicated device pool, request the selected profile and
record host inventory, frozen grant and Training/Candidate-visible devices as
different observations. Exercise a mismatching/heterogeneous inventory and the
loss of a selected device between preflight and execution. A runtime-visible
workload must receive the exact selected set; no count-only replacement is
allowed. CPU evaluation must receive no GPU grant even if Training uses GPUs.

Use a controlled device outside the fixture's grant to check negative access;
do not inspect other users' workloads or devices outside the authorized test
scope. A host inventory or container configuration alone cannot pass that
isolation check. The bundled Worker rejects unsupported MIG inventories. Keep
its `best_effort` status unless the separate capability/conformance/trust process
actually establishes a stronger claim.

### R08-E: preflight, receipts and infrastructure attribution

First check that an unsupported normative contract is rejected before episode
setup/task reveal; unavailable observational fields remain permitted. Change a
controlled backend condition after successful preflight and verify that the
actual evaluation rechecks it. An absent preview envelope produces no fabricated
allocation evidence.

Run [the v16 evaluation probe](worker-evaluation-probe.md) with one supported
and one deliberately unsupported envelope. It attempts at most four calls:
supported TEST/ANALYSIS, then rejected TEST/ANALYSIS. Retain the private report
and independently inspect actual CPU/memory/device settings and teardown. A
returned `EvaluationResourceFailure` establishes the observed error type; it
does not by itself prove the rejection cause or absence of a launched workload.

In a separate trusted Supervisor integration harness using public fixtures,
inject missing/wrong/reused receipts and a resource-query failure for TEST, then
for ANALYSIS after an accepted TEST. Pass when TEST reports
`infrastructure_failure` with `evaluation_resource_contract_failure`, no score
and no subsequent ANALYSIS. An ANALYSIS-only resource failure must make the
diagnostics incomplete without erasing TEST or inventing a teardown violation;
its prior policy/eligibility disposition is preserved. Actual
`WorkerTeardownFailure` remains a distinct cleanup failure. Result v14
must revalidate binding and reject edited/replayed evidence. Equivalent reordered
contracts must be accepted, genuinely changed requirements rejected.

Delay workload/data resolution and evaluation/inspection to check the shared
operation deadline; late success must not become an accepted measurement. Inspect
the Candidate invocation clock separately from trusted preparation, resource
inspection and scoring. The probe does not test Supervisor publication or a
wedged call's independent termination; those require R08-L and R04.

### R08-L: teardown gates and result persistence

Using public fixtures, prevent research/control teardown confirmation and verify
that hidden-data resolution and evaluation never begin. Once sealed and stopped,
check that the Candidate receives one target-free input and no research workspace,
while scoring retains its separate target view. Bind actual materialized fixture
bytes to the declared identity; reject changed identities. Log access attempts
without retaining real organizer hidden content.

If role images or UID mappings changed, also verify permitted reads/writes and
post-stop cleanup across Agent, Training, Candidate and scorer under the final
identities, including restrictive fixture modes and image `USER` settings where
applicable. Required inputs must remain readable without widening hidden-data
access. This is a changed-boundary check, not a reopening of issue 1's already
demonstrated main path.

After a successful TEST, independently inject an ANALYSIS evidence/error path,
a genuine cleanup failure and a result-publication write failure. Pass when
TEST remains the primary accepted decision, ANALYSIS cannot select an incumbent,
and cleanup attempts/dispositions precede final publication. Diagnostic or
post-TEST teardown failure must not erase the TEST measurement; genuine cleanup
failure must remain recorded and revoke compliance/eligibility. A finalized
report must not claim confirmed cleanup when absence is unconfirmed. A prepared
event alone is not a finalized result. An in-process publication retry must not
duplicate the decision; cleanup recovery after process loss must not resume or publish the
episode. Validate the resulting authoritative report and retain external
evidence when Core cannot write one.

### R08-O: observability and external tool compatibility

Verify a short flushed child log is visible while the process lives; fill or
fail its log sink and confirm pipes still drain and lost/incomplete usage is
marked correctly. Check final state against stale progress, expired observations,
one unavailable monitoring source and an externally terminated job. Pass when
the trusted view distinguishes current execution, the historical snapshot,
accepted result, workload cleanup and storage disposal without rewriting Core.

For an affected Agent wrapper, exercise actual required file/command tools in
the final UID/path/temp view on a separate diagnostic workspace. A version or
shell-only check does not establish native tool compatibility. Follow the
already selected Agent identity and diagnostic authorization; this matrix does
not select or supply a Tested Agent. An affected wrapper remains unverified until
its own evidence exists; a fix in another deployment does not establish its pass.

## Execution order and completion record

The next deployment work is R08-E, with R08-L attribution/publication checks and
R08-O checks for the affected wrapper. Begin with a supported public CPU fixture
where that deployment supports one; a CPU result covers only that fixture. Use
the selected CUDA profile for the GPU-specific portion rather than substituting
its hardware. Then exercise R02/R03/R04/R05 in fresh disposable fault scopes and
R06 on an authorized device pool. Stop a sequence when ownership or teardown is
unconfirmed; recover that scope before admitting more work.

This order is a proposed diagnostic sequence, not a new resource authorization.
Before execution, resolve the deployment, fixtures, case list, job/call counts,
resource limits and retry scope under [the integration workflow](worker-integration.md).
No particular host, backend or GPU allocation is selected here. Reuse valid,
unchanged evidence when its full identity and scope apply; do not impose the
whole adversarial matrix on every ordinary preview run.

For each case, retain a trusted record containing:

| Field | Required content |
| --- | --- |
| Case and disposition | Scenario ID/subcase, `pending`/`passed`/`failed`/`not_applicable`, reviewer, date and scope/limitation |
| Exact implementation | Core commit and dirty diff identity; Worker source/dependency hashes, launcher configuration and runtime identity; protocol/schema versions |
| Frozen inputs | Fixture task/Candidate/data identities, selected resource contract and applicable Agent identity; no secret values or real hidden content |
| Authorization and clocks | Approved working root, resource/job/call/retry limits; start/deadline/finish observations and termination/recovery grace |
| Injection and expectation | Controlled failure, expected error/state, observability method and criteria fixed before execution |
| Actual evidence | Bounded logs, Worker receipts, actual allocation/workload observations, authoritative state/report where available; artifact locations and SHA-256 manifest |
| Cleanup and follow-up | Owned workload absence, private-storage removal, archive verification/disposal separately; unresolved handles, next responsible party and retest scope |

These are Operator record fields, not a new signed receipt or Core schema. Keep
prior failed records; attach a new attempt when retesting. A row closes for a
deployment only when all selected applicable subcases have evidence and the
remaining limitations are explicit. Missing guarantees remain missing and can
still prevent the intended run's admission. A passing diagnostic never upgrades
formal eligibility.

Completing this document closes the **matrix deliverable**, not the pending
real-runtime acceptance. Step 5 still requires release owners to freeze and
Workers to enforce the comparison conditions in
[evaluation resources](evaluation-resources.md). Step 6 covers original issues
10/12/13: formal conformance, Gateway execution identity and release/publication.
Original issue 11 and full remote operations remain demand-driven work.
