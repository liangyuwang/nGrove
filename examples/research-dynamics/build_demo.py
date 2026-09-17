"""Rebuild a synthetic teaching report; this is not an official episode loader.

Run with Python 3 and matplotlib installed. All inputs are local invented data.
"""

from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
SOURCE = ROOT / "fixtures" / "demo.json"
CONFIG = ROOT / "analysis-config.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def snapshot(episode, cutoff):
    candidates = {c["candidate_id"]: c for c in episode["candidates"]}
    commits, declaration = set(), None
    for event in episode["events"]:
        if event["seq"] > cutoff:
            break
        if event["type"] == "CANDIDATE_COMMITTED":
            commits.add(event["candidate_id"])
        elif event["type"] == "INCUMBENT_DECLARED":
            assert event["candidate_id"] in commits
            declaration = event
    current = candidates[declaration["candidate_id"]] if declaration else None
    scored = [
        candidates[cid]
        for cid in commits
        if candidates[cid]["analysis_raw_metrics"]["mean_nll"] is not None
    ]
    best = (
        min(
            scored,
            key=lambda c: (c["analysis_raw_metrics"]["mean_nll"], c["candidate_id"]),
        )
        if scored
        else None
    )
    loss = current["analysis_raw_metrics"]["mean_nll"] if current else None
    oracle = best["analysis_raw_metrics"]["mean_nll"] if best else None
    return {
        "event_cutoff": cutoff,
        "candidate_id": current["candidate_id"] if current else None,
        "source_event_seq": declaration["seq"] if declaration else None,
        "loss": loss,
        "time_per_task": current["time_per_task_seconds"] if current else None,
        "status": "known" if loss is not None else "unknown",
        "reason": None
        if loss is not None
        else (current["reason"] if current else "no_incumbent"),
        "oracle_loss": oracle,
        "oracle_candidate_id": best["candidate_id"] if best else None,
        "loss_only_selection_gap": loss - oracle
        if loss is not None and oracle is not None
        else None,
    }


def at_time(episode, seconds):
    cutoff = max(
        (e["seq"] for e in episode["events"] if e["elapsed_seconds"] <= seconds),
        default=0,
    )
    return {"budget": seconds, **snapshot(episode, cutoff)}


def at_lease(episode, n):
    next_lease = next(
        (
            e
            for e in episode["events"]
            if e["type"] == "LEASE_CONSUMED" and e["leases_used"] == n + 1
        ),
        None,
    )
    cutoff = (
        next_lease["seq"] - 1
        if next_lease
        else next(e["seq"] for e in episode["events"] if e["type"] == "RESEARCH_SEALED")
    )
    return {"budget": n, **snapshot(episode, cutoff)}


def attainment(points, target):
    hidden_success_possible = False
    for point in points:
        if point["candidate_id"] is not None and point["loss"] is None:
            hidden_success_possible = True
        if point["loss"] is not None and point["loss"] <= target:
            return {
                "target": target,
                "first_observed_budget": point["budget"],
                "exact_budget": None if hidden_success_possible else point["budget"],
                "status": "observed_upper_bound"
                if hidden_success_possible
                else "exact",
                "reason": "Earlier unknown incumbent quality may hide an earlier success."
                if hidden_success_possible
                else None,
            }
    return {
        "target": target,
        "first_observed_budget": None,
        "exact_budget": None,
        "status": "unknown" if hidden_success_possible else "not_reached",
        "reason": "Earlier unknown incumbent quality may hide success."
        if hidden_success_possible
        else "No attainment in the observation domain.",
    }


def integrate(points):
    total, known, area = points[-1]["budget"] - points[0]["budget"], 0, 0.0
    for left, right in zip(points, points[1:]):
        span = right["budget"] - left["budget"]
        if left["loss"] is not None:
            known += span
            area += span * left["loss"]
    return {
        "value": area / total if total > 0 and known == total else None,
        "status": "known" if total > 0 and known == total else "insufficient_coverage",
        "reason": None
        if total > 0 and known == total
        else "Requested interval contains unknown incumbent quality.",
        "known_integral": area,
        "known_duration": known,
        "requested_duration": total,
        "coverage": known / total if total else None,
    }


def main():
    before = {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, CONFIG)}
    fixture = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (
        fixture["schema"] == "autotrain-synthetic-demo-fixture-v1"
        and fixture["synthetic"] is True
    )
    assert config["synthetic"] is True
    A, B = fixture["episodes"]
    for episode in (A, B):
        assert episode["synthetic"] is True
        assert [e["seq"] for e in episode["events"]] == sorted(
            {e["seq"] for e in episode["events"]}
        )
    OUT.mkdir(exist_ok=True)
    a_times = sorted(
        set(
            config["A"]["domain_seconds"]
            + config["A"]["checkpoints_seconds"]
            + [e["elapsed_seconds"] for e in A["events"]]
        )
    )
    a = [at_time(A, t) for t in a_times]
    b = [at_lease(B, n) for n in config["B"]["checkpoints"]]
    a_mean = integrate(a)
    a_target = attainment(a, config["A"]["target"])
    b_target = attainment(b, config["B"]["target"])
    weights = config["B"]["weights"]
    assert len(weights) == len(b) and all(w > 0 for w in weights)
    known_weight = sum(w for p, w in zip(b, weights) if p["loss"] is not None)
    b_mean = {
        "value": sum(w * p["loss"] for p, w in zip(b, weights)) / sum(weights)
        if known_weight == sum(weights)
        else None,
        "status": "known" if known_weight == sum(weights) else "insufficient_coverage",
        "reason": None
        if known_weight == sum(weights)
        else "Checkpoint 2 has unknown incumbent quality.",
        "coverage": known_weight / sum(weights),
        "weights": weights,
    }

    # Numeric and behavioral checks exercise the core teaching cases.
    hand = [
        {"budget": 1, "loss": 5},
        {"budget": 3, "loss": 3},
        {"budget": 4, "loss": 5},
        {"budget": 5, "loss": 5},
    ]
    assert integrate(hand)["value"] == 4.5
    assert at_time(A, 2 * 3600)["candidate_id"] == "A1"
    assert at_time(A, 4 * 3600)["candidate_id"] == "A2"
    assert at_time(A, 4 * 3600)["oracle_candidate_id"] == "A3"
    assert at_time(A, 5 * 3600)["candidate_id"] == "A1"
    assert at_time(A, 7 * 3600)["loss"] == at_time(A, int(7.5 * 3600))["loss"]
    assert a_mean["value"] is None and a_mean["coverage"] == 7 / 8
    assert math.isclose(a_mean["known_integral"] / 3600, 25.5)
    assert a_target["exact_budget"] == 6 * 3600
    assert b[1]["loss"] is None and b[1]["candidate_id"] == "B2"
    assert b_target["exact_budget"] is None and b_target["first_observed_budget"] == 3
    assert b_mean["coverage"] == 0.75 and b_mean["value"] is None
    assert b[-1]["candidate_id"] == "B3" and b[-1]["budget"] == 4

    metrics = {
        "schema": "autotrain-analysis-synthetic-demo-report-v1",
        "synthetic": True,
        "notice": fixture["notice"],
        "admission": {
            "status": "synthetic_demonstration_only",
            "real_episode_admission": "not_applicable",
            "reason": "No real episodes or workload lifecycle receipts are claimed; this is not an official-result validator.",
        },
        "provenance": {
            "input_hashes": before,
            "tool_sha256": sha(Path(__file__)),
            "original_result_version": None,
            "identity_algorithms": None,
            "compatibility_adapter": None,
            "reason": "Demo-only schema; not official Benchcore state.",
            "configuration": config,
        },
        "comparison": {
            "combined_ranking": False,
            "reason": "Independent illustrative Tracks; no cross-Track ranking.",
        },
        "episodes": {
            "DEMO-A": {
                "mean": a_mean,
                "attainment": a_target,
                "points": a,
                "candidate_coverage": {"evaluated": 5, "unique_candidates": 5},
                "illustrative_test": A["illustrative_test"],
            },
            "DEMO-B": {
                "mean": b_mean,
                "attainment": b_target,
                "points": b,
                "candidate_coverage": {"evaluated": 2, "unique_candidates": 3},
                "illustrative_test": B["illustrative_test"],
            },
        },
        "warnings": [
            "All numbers and process notes are invented.",
            "No official performance-score mapping is defined.",
            "No usage traces, causal controls, or independent repeat runs are modeled.",
            "Curvature and significance are intentionally not computed.",
        ],
    }
    save_json(OUT / "analysis.json", metrics)
    save_json(OUT / "analysis-config.json", config)
    rows = []
    for episode, points in ((A, a), (B, b)):
        for p in points:
            for kind in ("incumbent", "observed_oracle"):
                loss = p["loss"] if kind == "incumbent" else p["oracle_loss"]
                rows.append(
                    {
                        "synthetic": True,
                        "episode_id": episode["episode_id"],
                        "track": episode["track"],
                        "axis": "elapsed_seconds"
                        if episode["track"] == "A"
                        else "consumed_leases",
                        "budget": p["budget"],
                        "event_cutoff": p["event_cutoff"],
                        "curve_type": kind,
                        "candidate_id": p["candidate_id"]
                        if kind == "incumbent"
                        else p["oracle_candidate_id"],
                        "source_event_seq": p["source_event_seq"]
                        if kind == "incumbent"
                        else None,
                        "loss": loss,
                        "time_per_task": p["time_per_task"]
                        if kind == "incumbent"
                        else None,
                        "status": "known" if loss is not None else "unknown",
                        "reason": (
                            p["reason"]
                            if kind == "incumbent"
                            else "no_evaluated_candidate"
                        )
                        if loss is None
                        else None,
                        "held_after_finish": False,
                    }
                )
        for e in episode["events"]:
            if e["type"] != "CANDIDATE_COMMITTED":
                continue
            c = next(
                c
                for c in episode["candidates"]
                if c["candidate_id"] == e["candidate_id"]
            )
            rows.append(
                {
                    "synthetic": True,
                    "episode_id": episode["episode_id"],
                    "track": episode["track"],
                    "axis": "elapsed_seconds"
                    if episode["track"] == "A"
                    else "consumed_leases",
                    "budget": e.get("elapsed_seconds", e.get("leases_used")),
                    "event_cutoff": e["seq"],
                    "curve_type": "submitted",
                    "candidate_id": c["candidate_id"],
                    "source_event_seq": e["seq"],
                    "loss": c["analysis_raw_metrics"]["mean_nll"],
                    "time_per_task": c["time_per_task_seconds"],
                    "status": c["status"],
                    "reason": c["reason"],
                    "held_after_finish": False,
                }
            )
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    notes = fixture["process_notes"]
    evidence = [
        {
            "record_type": "observation",
            "id": "e1",
            "synthetic": True,
            "episode_id": "DEMO-A",
            "sources": [
                {"file": "../fixtures/demo.json", "json_pointer": "/process_notes/0"}
            ],
            "change": notes[0]["text"],
            "tags": ["implemented_change"],
            "certainty": "observed",
            "scope": "Observation within the invented fixture only; no actual implementation is claimed.",
        },
        {
            "record_type": "observation",
            "id": "e2",
            "synthetic": True,
            "episode_id": "DEMO-A",
            "sources": [
                {"file": "../fixtures/demo.json", "json_pointer": "/process_notes/1"}
            ],
            "change": notes[1]["text"],
            "tags": ["reused_change"],
            "certainty": "observed",
            "scope": "Illustrative fixture behavior only.",
        },
        {
            "record_type": "relation",
            "id": "r1",
            "synthetic": True,
            "source_id": "e1",
            "target_id": "e2",
            "relation": "used_to_select",
            "certainty": "observed",
            "sources": [
                {"file": "../fixtures/demo.json", "json_pointer": "/process_notes/1"}
            ],
            "limitations": [
                "Selects a Candidate, not a later research mechanism; recursive_reuse is not established.",
                "No real execution or causal effect is claimed.",
            ],
        },
    ]
    (OUT / "evidence.jsonl").write_text(
        "".join(json.dumps(e, allow_nan=False) + "\n" for e in evidence),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "text.color": "#172b4d",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "figure.facecolor": "#f7f9fc",
            "axes.facecolor": "white",
        }
    )
    teal, amber, violet = "#007f86", "#d88a12", "#785ac8"
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.4, 1]}
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.09, right=0.96, hspace=0.60)
    fig.text(
        0.09,
        0.955,
        "How research decisions become measurable",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.09,
        0.918,
        "AUTOTRAIN ANALYSIS DEMO  /  ALL DATA ARE SYNTHETIC",
        fontsize=11,
        color="#a15c00",
        weight="bold",
    )
    ax = axes[0]
    ax.axvspan(0, 1, color="#e9edf3", alpha=0.9)
    xs = [p["budget"] / 3600 for p in a]
    ax.step(
        xs,
        [p["loss"] if p["loss"] is not None else float("nan") for p in a],
        where="post",
        color=teal,
        lw=3,
        label="Declared incumbent",
    )
    ax.step(
        xs,
        [p["oracle_loss"] if p["oracle_loss"] is not None else float("nan") for p in a],
        where="post",
        color=amber,
        lw=2,
        ls="--",
        label="Observed oracle (postmortem)",
    )
    commits = [e for e in A["events"] if e["type"] == "CANDIDATE_COMMITTED"]
    losses = {
        c["candidate_id"]: c["analysis_raw_metrics"]["mean_nll"]
        for c in A["candidates"]
    }
    ax.scatter(
        [e["elapsed_seconds"] / 3600 for e in commits],
        [losses[e["candidate_id"]] for e in commits],
        s=65,
        color=violet,
        edgecolors="white",
        zorder=5,
        label="Submitted Candidate",
    )
    ax.annotate(
        "A3 is better, but never declared",
        xy=(4, 3.1),
        xytext=(2.1, 2.77),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        color="#475569",
    )
    ax.annotate(
        "Re-declaring A1 causes a regression",
        xy=(5, 4),
        xytext=(4.0, 4.16),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        color="#475569",
    )
    ax.text(0.12, 3.42, "No\nincumbent", fontsize=9, color="#64748b")
    ax.set(
        title="Track A  |  Submitted quality and delivered quality can differ",
        xlabel="Recorded research time (hours)",
        ylabel="ANALYSIS mean NLL  ↓",
        xlim=(0, 8),
        ylim=(2.65, 4.42),
    )
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, 1.00), frameon=False, fontsize=9, ncol=3
    )
    ax = axes[1]
    x = [p["budget"] for p in b]
    y = [p["loss"] if p["loss"] is not None else float("nan") for p in b]
    ax.axvspan(1.86, 2.14, color="#fff0d8")
    ax.plot(x, y, "o", ms=9, color=teal, label="Incumbent at fixed Lease checkpoints")
    ax.scatter(
        x,
        [p["oracle_loss"] for p in b],
        s=190,
        marker="s",
        facecolors="none",
        edgecolors=amber,
        linewidths=1.8,
        label="Observed oracle (postmortem)",
    )
    ax.scatter(
        [1, 3],
        [4, 3.2],
        s=170,
        facecolors="none",
        edgecolors=violet,
        linewidths=1.8,
        label="Scored submission",
    )
    ax.axhline(3.5, color="#94a3b8", ls=":", lw=1.5)
    ax.text(3.47, 3.53, "Demo target 3.5", fontsize=9, color="#64748b")
    ax.text(
        2,
        3.70,
        "B2 is unscored\nquality is unknown",
        ha="center",
        fontsize=10,
        color="#a15c00",
    )
    ax.annotate(
        "Target observed by Lease 3;\nfirst attainment remains unknown",
        xy=(3, 3.2),
        xytext=(1.25, 2.85),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        color="#475569",
    )
    ax.set(
        title="Track B  |  Missing evaluations change what can be concluded",
        xlabel="Consumed Leases (includes a failed fourth Lease)",
        ylabel="ANALYSIS mean NLL  ↓",
        xlim=(0.6, 4.4),
        ylim=(2.72, 4.25),
        xticks=x,
    )
    ax.grid(axis="y", alpha=0.15)
    fig.text(
        0.09,
        0.025,
        "Illustrative postmortem measurements only. No real Agent ranking or RSI claim is made.",
        fontsize=10,
        color="#64748b",
    )
    fig.savefig(OUT / "curves.png", dpi=160)
    fig.savefig(OUT / "curves.svg")
    svg_path = OUT / "curves.svg"
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)

    a_lines = "\n".join(
        f"| {p['budget']/3600:g} | {p['candidate_id'] or 'None'} | {p['loss'] if p['loss'] is not None else 'Unknown'} | {p['oracle_loss'] if p['oracle_loss'] is not None else 'Unknown'} |"
        for p in a
        if p["budget"] in config["A"]["checkpoints_seconds"]
    )
    b_lines = "\n".join(
        f"| {p['budget']} | {p['candidate_id']} | {p['loss'] if p['loss'] is not None else 'Unknown'} | {p['oracle_loss']} |"
        for p in b
    )
    report = f"""# AutoTrain Analysis — Worked Demo

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
| Incumbent coverage | {a_mean['coverage']:.1%} of requested duration | {b_mean['coverage']:.0%} of checkpoint weight |
| Mean incumbent NLL | Unavailable: no incumbent in hour 0–1 | Unavailable: Lease 2 is unscored |
| Known-interval integral | {a_mean['known_integral']/3600:.1f} NLL·hours; not a full-domain mean | Not a continuous-time statistic |
| Target ≤ 3.5 | Exact first attainment: hour {a_target['exact_budget']/3600:g} | First observed: Lease 3; exact first attainment unknown |
| Final ANALYSIS NLL | {a[-1]['loss']:.2f} | {b[-1]['loss']:.2f} |
| Illustrative final TEST NLL | {A['illustrative_test']['mean_nll']:.2f} | {B['illustrative_test']['mean_nll']:.2f} |
| Illustrative final TEST serving cost | {A['illustrative_test']['time_per_task_seconds']*1000:.1f} ms/task | {B['illustrative_test']['time_per_task_seconds']*1000:.1f} ms/task |

The TEST numbers are separate invented endpoint values, never a historical TEST curve. No official 0–100 score is produced because no release score mapping is defined. Tracks are independent demonstrations and are not ranked against each other. Agent token usage is unknown because this fixture does not model usage traces.

## Track A: commitments and declarations are different

| Hour | Actual incumbent | Incumbent NLL | Observed oracle NLL |
| --- | --- | --- | --- |
{a_lines}

Sources: [synthetic input](../fixtures/demo.json), `/episodes/0/events`; commit event 5 introduces A3, while event 6 re-declares A1. The duplicate A5 declaration at hour 7.5 does not change quality or attainment. All Candidate scores are invented postmortem observations, not feedback seen by a researching Agent.

## Track B: a gap stays a gap

| Consumed Leases | Actual incumbent | Incumbent NLL | Observed oracle NLL |
| --- | --- | --- | --- |
{b_lines}

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

Source fixture SHA-256: `{before['fixtures/demo.json']}`.

Configuration SHA-256: `{before['analysis-config.json']}`.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, CONFIG)}
    print(
        json.dumps(
            {
                "report": str(OUT / "report.md"),
                "figure": str(OUT / "curves.png"),
                "A_coverage": a_mean["coverage"],
                "A_exact_target_hours": a_target["exact_budget"] / 3600,
                "B_coverage": b_mean["coverage"],
                "B_exact_target": b_target["exact_budget"],
                "B_first_observed_target": b_target["first_observed_budget"],
                "validation": "Passed; source inputs unchanged.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
