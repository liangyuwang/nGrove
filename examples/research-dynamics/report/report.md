# AutoTrain Analysis — Worked Demo

**SYNTHETIC EXAMPLE. Every number and process note below is invented.** This demonstrates the output of the `autotrain-analysis` skill; it is not a benchmark result, an actual Agent trace, or a production analyzer.

## What the analysis reveals

- Track A submits a loss-3.1 Candidate at hour 4 but never declares it. Delivered quality remains at 3.6. The 0.5 NLL gap is a loss-only comparison: the unselected Candidate is also slower (12 versus 6 ms/task in the synthetic ANALYSIS measurements).
- Re-declaring A1 at hour 5 worsens delivered NLL from 3.6 to 4.0. The observed oracle remains 3.1. This is why a best-so-far curve cannot represent actual delivery decisions.
- Track B first shows a target-meeting incumbent at Lease 3. Because B2 is unscored, the report can say **reached by Lease 3**, but cannot claim the target was first reached at Lease 3.
- The process notes illustrate tool reuse. They do not establish a dependency on a later research mechanism, a causal gain, or an RSI capability score.

![Synthetic research curves](curves.png)

## Frozen demonstration settings

Both cases use mean NLL, lower being better. A uses recorded elapsed seconds, displayed as hours. B uses consumed Leases, with each checkpoint cut off before the next Lease is consumed; the final checkpoint uses the synthetic seal event. The target 3.5 and the domains/weights are explicitly selected teaching settings, not recommended defaults.

Input schema: `autotrain-synthetic-demo-fixture-v1`. Real-episode admission is **not applicable**: the fixture contains no actual workload receipts. Nothing was trained, evaluated, resumed, or published. A production analysis would require the skill's sealing, termination, cleanup, publication, identity, and consistency checks.

## Computed summary

| Metric | Track A demo | Track B demo |
| --- | --- | --- |
| Requested domain | 0–8 hours | Leases 1–4, equal weights |
| Candidate quality coverage | 5/5 unique Candidates | 2/3 unique Candidates |
| Incumbent coverage | 87.5% of requested duration | 75% of checkpoint weight |
| Mean incumbent NLL | Unavailable: no incumbent in hour 0–1 | Unavailable: Lease 2 is unscored |
| Known-interval integral | 25.5 NLL·hours; not a full-domain mean | Not a continuous-time statistic |
| Target ≤ 3.5 | Exact first attainment: hour 6 | First observed: Lease 3; exact first attainment unknown |
| Final ANALYSIS NLL | 3.00 | 3.20 |
| Illustrative final TEST NLL | 3.04 | 3.25 |
| Illustrative final TEST serving cost | 5.2 ms/task | 5.1 ms/task |

The TEST numbers are separate invented endpoint values, never a historical TEST curve. No official 0–100 score is produced because no release score mapping is defined. Tracks are independent demonstrations and are not ranked against each other. Agent token usage is unknown because this fixture does not model usage traces.

## Track A: commitments and declarations are different

| Hour | Actual incumbent | Incumbent NLL | Observed oracle NLL |
| --- | --- | --- | --- |
| 0 | None | Unknown | Unknown |
| 1 | A1 | 4.0 | 4.0 |
| 2 | A1 | 4.0 | 3.6 |
| 3 | A2 | 3.6 | 3.6 |
| 4 | A2 | 3.6 | 3.1 |
| 5 | A1 | 4.0 | 3.1 |
| 6 | A4 | 3.3 | 3.1 |
| 7 | A5 | 3.0 | 3.0 |
| 8 | A5 | 3.0 | 3.0 |

Sources: [synthetic input](../fixtures/demo.json), `/episodes/0/events`; commit event 5 introduces A3, while event 6 re-declares A1. The duplicate A5 declaration at hour 7.5 does not change quality or attainment. All Candidate scores are invented postmortem observations, not feedback seen by a researching Agent.

## Track B: a gap stays a gap

| Consumed Leases | Actual incumbent | Incumbent NLL | Observed oracle NLL |
| --- | --- | --- | --- |
| 1 | B1 | 4.0 | 4.0 |
| 2 | B2 | Unknown | 4.0 |
| 3 | B3 | 3.2 | 3.2 |
| 4 | B3 | 3.2 | 3.2 |

Sources: [synthetic input](../fixtures/demo.json), `/episodes/1/events` and `/episodes/1/candidates/1`. At checkpoint 2, B2 is the actual incumbent; B1's known score cannot replace it. Event 10 consumes the fourth Lease before event 11 records its failure, so that Lease still counts. No new declaration occurs, and B3 remains the final incumbent. The figure shows fixed checkpoint values without interpolation; orange squares mark the observed oracle.

## Process evidence: reuse without a recursive claim

| Record | Illustrative evidence | Supported annotation |
| --- | --- | --- |
| note-1 | A validation helper is changed to compare Candidates on the same participant-owned split. | implemented_change, within the fictional fixture only |
| note-2 | A later selection step calls that helper before declaring A2. | reused_change; used_to_select relation |
| note-3 | No record links the helper to a subsequent research mechanism. | recursive reuse remains unestablished |

Sources: [synthetic process notes](../fixtures/demo.json), `/process_notes/0` through `/process_notes/2`; [derived annotations](evidence.jsonl). These are invented narrative records, not verified code changes or actual execution evidence. Their purpose is to show how claims would be separated from stronger evidence in a real report. No causal benefit is inferred from timing or quality improvement.

## Reproduce and inspect

- [Numerical results and provenance](analysis.json)
- [Frozen configuration](analysis-config.json)
- [Raw and reconstructed curves](curves.csv)
- [Evidence annotations](evidence.jsonl)
- [Source fixture](../fixtures/demo.json)
- [Figure as SVG](curves.svg)

Rebuild with Python 3 and matplotlib using `python build_demo.py` from the demo directory. The script only accepts the synthetic demo schema and writes into `report/`. It checks the step-integral example, undeclared-candidate handling, regression, repeated declarations, missing quality, target ambiguity, and failed-Lease accounting. Inputs are hashed before and after execution and must remain unchanged. Identical input/configuration gives identical numerical results.

Source fixture SHA-256: `af248ce8faddf4110f9397c0999861046a343bc61784ea484ff0f8800fcd1444`.

Configuration SHA-256: `bf4bc81cc8a3ddcc8860f6599f947335fd9c0a5a55955ce818fcd6d95ec2ad52`.
