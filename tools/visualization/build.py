"""Render saved LM result reports or a synthetic example as an offline dashboard.

This is a display/export tool, not a research-dynamics admission validator.
It never evaluates Candidates, loads hidden data, or executes episode commands.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def perplexity(loss):
    if loss is None or not finite(loss) or loss > 700:
        return None
    return math.exp(loss)


def cost_fields(cost):
    cost = cost or {}
    seconds = cost.get("time_per_task")
    return {
        "seconds_per_task": seconds,
        "ms": None if seconds is None else seconds * 1000,
        "evaluation_seconds": cost.get("candidate_wall_seconds"),
        "task_count": cost.get("task_count"),
        "tasks_per_second": cost.get("tasks_per_second"),
        "input_tokens_per_second": cost.get("input_tokens_per_second"),
    }


def base_row(label, track):
    return {
        "model": label,
        "track": track,
        "budget_hours": None,
        "lease_hours": None,
        "research_hours": None,
        "reasoning_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "agent_turns": None,
        "reasoning_effort": None,
    }


def demo_data(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if (
        fixture.get("schema") != "autotrain-synthetic-demo-fixture-v1"
        or fixture.get("synthetic") is not True
    ):
        raise ValueError("Expected the explicitly synthetic demo fixture.")
    rows = []
    for episode in fixture["episodes"]:
        candidates = {c["candidate_id"]: c for c in episode["candidates"]}
        commits = [e for e in episode["events"] if e["type"] == "CANDIDATE_COMMITTED"]
        test = episode["illustrative_test"]
        points = []
        for index, event in enumerate(commits, 1):
            candidate = candidates[event["candidate_id"]]
            loss = candidate["analysis_raw_metrics"]["mean_nll"]
            points.append(
                {
                    "index": index,
                    "candidate_id": candidate["candidate_id"],
                    "research_hours": event.get("elapsed_seconds", 0) / 3600
                    if "elapsed_seconds" in event
                    else None,
                    "hours": None,
                    "leases_used": event.get("leases_used"),
                    "nll": loss,
                    "ppl": perplexity(loss),
                    "score": None,
                    "analysis_status": candidate["status"],
                    "final": candidate["candidate_id"] == test["candidate_id"],
                    **cost_fields(
                        {"time_per_task": candidate["time_per_task_seconds"]}
                    ),
                }
            )
        row = base_row("Synthetic Agent", episode["track"])
        row.update(
            status="synthetic",
            outcome="synthetic",
            policy_compliant=None,
            leaderboard_eligible=False,
            test_nll=test["mean_nll"],
            score=None,
            ppl=perplexity(test["mean_nll"]),
            leases=max((e.get("leases_used", 0) for e in episode["events"]), default=0)
            if episode["track"] == "B"
            else None,
            candidates=len(candidates),
            first_analysis=points[0]["nll"] if points else None,
            final_analysis=candidates[test["candidate_id"]]["analysis_raw_metrics"][
                "mean_nll"
            ],
            points=points,
            **cost_fields({"time_per_task": test["time_per_task_seconds"]}),
        )
        rows.append(row)
    return {
        "synthetic": True,
        "task": "Synthetic LM illustration",
        "runtime": "No workloads executed",
        "track_budgets": {"A": "8 hours (synthetic)", "B": "4 Leases (synthetic)"},
        "generated_at": "Synthetic fixture",
        "rows": rows,
    }


def report_data(paths, labels):
    # Bootstrap only the repository's pure result validator, never a Supervisor.
    sys.path.insert(0, str(ROOT))
    from benchcore.results import normalize_result_report

    rows, groups, budgets, tasks, runtimes, seals = [], {}, {}, set(), set(), []
    for index, path in enumerate(paths):
        result = normalize_result_report(json.loads(path.read_text(encoding="utf-8")))
        if result["status"] != "finalized":
            raise ValueError("Only saved finalized reports are accepted for display.")
        if result.get("active_lease") or result.get("pending_lease_teardowns"):
            raise ValueError("Report contains active or unresolved Lease state.")
        if result.get("agent_control", {}).get("cleanup_complete") is not True:
            raise ValueError("Report must record confirmed Agent-control cleanup.")
        if not result.get("agent_system", {}).get("termination_reason"):
            raise ValueError("Report must record Agent termination.")
        if result.get("deployment_teardown_failures"):
            raise ValueError("Report contains unresolved deployment teardown failures.")
        track = result.get("track")
        if track not in {"A", "B"}:
            raise ValueError("Report must specify Track A or B.")
        if result["performance_score_contract"]["raw_metrics"] != ["mean_nll"]:
            raise ValueError("This viewer currently supports LM mean_nll reports only.")
        key = result.get("comparison_key") or {}
        required_key = {"challenge_id", "track", "track_config", "runtime_profile"}
        complete_key = all(
            isinstance(key.get(name), str) and key[name] for name in required_key
        )
        if key and key.get("track") != track:
            raise ValueError("Comparison-key Track does not match the result.")
        contract = {
            field: result.get(field)
            for field in (
                "comparison_key",
                "artifact_identities",
                "artifact_identity_algorithms",
                "performance_score_contract",
                "cost_contract",
                "resource_envelope",
                "evaluation_resource_envelope",
            )
        }
        if track in groups and (
            not complete_key
            or not result.get("artifact_identities")
            or groups[track] != contract
        ):
            raise ValueError(
                "Different or missing comparison conditions within a Track; render separately."
            )
        groups[track] = contract
        budgets[track] = key.get("track_config", "Not recorded")
        tasks.add(result.get("task_id", "Not recorded"))
        runtimes.add(key.get("runtime_profile", "Not recorded"))
        seals.append(result["research_sealed_at"])
        points = []
        for point in sorted(
            result["analysis_trajectory"], key=lambda p: p["event_seq"]
        ):
            loss = (point.get("analysis_raw_metrics") or {}).get("mean_nll")
            points.append(
                {
                    "index": point["commit_index"],
                    "candidate_id": point["candidate_id"],
                    "research_hours": point["elapsed_seconds"] / 3600,
                    "hours": None,
                    "leases_used": None,
                    "nll": loss,
                    "ppl": perplexity(loss),
                    "score": point.get("analysis_performance_score"),
                    "analysis_status": point["analysis_status"],
                    "final": point["candidate_id"] == result.get("final_candidate_id"),
                    **cost_fields(point.get("analysis_cost")),
                }
            )
        finals = [p for p in points if p["final"]]
        loss = (result.get("test_raw_metrics") or {}).get("mean_nll")
        row = base_row(labels[index] if labels else f"Run {index + 1}", track)
        timing = result.get("timing", {})
        row.update(
            status=result["status"],
            outcome=result["outcome"],
            policy_compliant=result.get("policy_compliant"),
            leaderboard_eligible=result.get("leaderboard_eligible"),
            test_nll=loss,
            score=result.get("test_performance_score"),
            ppl=perplexity(loss),
            research_hours=result.get("elapsed_seconds", 0) / 3600
            if "elapsed_seconds" in result
            else None,
            lease_hours=timing["training_lease_seconds"] / 3600
            if "training_lease_seconds" in timing
            else None,
            leases=result.get("leases_used"),
            candidates=len({p["candidate_id"] for p in result["candidate_commits"]}),
            first_analysis=points[0]["nll"] if points else None,
            final_analysis=finals[-1]["nll"] if finals else None,
            points=points,
            **cost_fields(result.get("test_cost")),
        )
        rows.append(row)
    if len({(r["model"], r["track"]) for r in rows}) != len(rows):
        raise ValueError("Each display label/Track pair must be unique.")
    return {
        "synthetic": False,
        "task": "; ".join(sorted(tasks)),
        "runtime": "; ".join(sorted(runtimes)),
        "track_budgets": budgets,
        "generated_at": max(seals),
        "rows": rows,
    }


def build(data, output, inputs):
    output = output.resolve()
    for source in inputs:
        if source.resolve().is_relative_to(output):
            raise ValueError("Output directory must not contain source inputs.")
    data = dict(data)
    data["schema_version"] = "autotrain-visualization-v1"
    data["source_sha256"] = hashlib.sha256(
        b"".join(hashlib.sha256(path.read_bytes()).digest() for path in inputs)
    ).hexdigest()
    data["core_revision"] = "See original report provenance"
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    translations = json.loads(
        (ASSETS / "translations.json").read_text(encoding="utf-8")
    )
    for marker in ("__RSI_DATA__", "__I18N_DATA__", "__DATA_URL__"):
        if template.count(marker) != 1:
            raise ValueError(f"Template must contain exactly one {marker} marker.")
    page = template.replace("__RSI_DATA__", payload.replace("<", "\\u003c"))
    page = page.replace(
        "__I18N_DATA__",
        json.dumps(translations, ensure_ascii=False).replace("<", "\\u003c"),
    )
    page = page.replace("__DATA_URL__", "data.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "data.json").write_text(payload + "\n", encoding="utf-8")
    (output / ".nojekyll").touch()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo", action="store_true", help="Use invented teaching data."
    )
    source.add_argument(
        "--report",
        type=Path,
        action="append",
        help="Saved finalized Benchcore report JSON; repeat for more runs.",
    )
    parser.add_argument(
        "--label", action="append", help="Display label per report, in input order."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".autotrain" / "visualization"
    )
    args = parser.parse_args()
    if args.label and (not args.report or len(args.label) != len(args.report)):
        parser.error("Supply one --label per --report, or omit all labels.")
    inputs = (
        [ROOT / "examples/research-dynamics/fixtures/demo.json"]
        if args.demo
        else args.report
    )
    before = [path.read_bytes() for path in inputs]
    try:
        data = demo_data(inputs[0]) if args.demo else report_data(inputs, args.label)
        output = build(data, args.output_dir, inputs)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if before != [path.read_bytes() for path in inputs]:
        raise RuntimeError("Inputs changed during rendering; discard this output.")
    print(f"Built {len(data['rows'])} run(s) in {output}")


if __name__ == "__main__":
    main()
