# Result visualization

This standalone, standard-library tool turns saved LM results into an offline
HTML dashboard. It is a trusted Operator tool and has no dependency from
Benchcore, Workers, or Evaluators. It does not launch workloads or evaluate
Candidates. Python 3.9+ is sufficient.

## Local demonstration

From the repository root:

```sh
python3 tools/visualization/build.py --demo
python3 -m http.server 8000 --bind 127.0.0.1 --directory .autotrain/visualization
```

Open `http://127.0.0.1:8000/?lang=en`. The default input is explicitly invented
teaching data from `examples/research-dynamics/fixtures/demo.json`. No actual
Agent results, hardware allocations, or official performance scores are implied.
The generated `index.html` also opens offline. Keep `data.json` alongside it for
the download link. Stop the local server with Ctrl+C.

## Saved Benchcore reports

After the episode is finalized and research termination/cleanup is confirmed,
save the existing Operator report outside the output directory:

```sh
autotrain-bench report --episode /path/to/finalized-episode > /path/to/report.json
python3 tools/visualization/build.py \
  --report /path/to/report.json --label "Agent A" \
  --output-dir /path/to/private-dashboard
```

The renderer also accepts a compatible finalized `state.json`. Repeat `--report`
and, optionally, supply one `--label` per file. Labels may identify arbitrary
Agents; there is no fixed model list. A label/Track pair must be unique.

The current Benchcore result normalizer validates schema, identity algorithms,
metric contracts, and Candidate bindings. Non-finalized reports and recorded
unresolved cleanup are rejected. Historical identities incompatible with the
current normalizer are rejected rather than relabeled. Multiple results within
one Track require matching recorded comparison keys, artifact identities, metric
contracts, and resource envelopes. Render incompatible groups separately.

These checks do not independently verify snapshot consistency, external workload
termination, hidden-data identity, actual resource equality, or formal eligibility.
Use the [analysis workflow](../../skills/autotrain-analysis/SKILL.md) for the
broader evidence review. The renderer preserves official outcome/compliance/
eligibility values; it cannot certify them. It currently supports the LM
`mean_nll` raw metric only.

## What the page shows

- Final TEST score/NLL and Candidate serving cost, separately from research cost.
- Submitted-Candidate ANALYSIS observations, including gaps for unscored commits.
- Configurable axes, linear/log scales, tooltips, filters, and English/Chinese UI.
- Recorded task/runtime/Track configuration, rather than a fixed hardware campaign.

Candidate history is not the declared-incumbent curve or an oracle curve. Those
are described in the analysis skill and demonstrated in the worked analysis.
No Candidate Lease coordinate is inferred from a Lease-ID suffix. Without
supporting lifecycle events, per-Candidate consumed Leases and training time
remain unavailable. Usage is also left unknown instead of trusting an aggregate
that may represent a missing trace as zero calls.

The consumed-Lease axis is omitted when no Candidate has a recorded coordinate.
Incumbent declarations record Lease consumption at declaration time, which can
follow further training after a Candidate commit; their counts cannot be used as
that commit's coordinate. The synthetic demo includes explicit commit-time Lease
counts. Time presets use recorded research elapsed time for Candidate history.
Missing or null task/runtime metadata is displayed as unavailable, and recorded
Core revisions are preserved in the exported provenance.

For multiple input files, source SHA-256 hashes the concatenation of their
individual SHA-256 digests in command-line order. Output includes selected
metrics and display labels, not Agent commands, transcripts, workspace files,
or full Operator reports. Even aggregate ANALYSIS may have organizer-only or
delayed-publication restrictions; local rendering is not publication approval.

## Optional GitHub hosting

Local use needs no GitHub account or token. Any static server can host the output.

To host the synthetic demo on your own GitHub repository, copy
`github-pages.yml.example` to `.github/workflows/visualization-pages.yml`, commit
it, select **GitHub Actions** in **Settings → Pages**, and manually run the
workflow. The template is inactive until installed and has no push trigger.
It uses your repository's automatic `GITHUB_TOKEN` with Pages/OIDC permissions,
not a token from this project's author. Your account must be able to configure
Pages and Actions in that repository. For real data, deliberately change the
build input only after reviewing its publication policy.

See GitHub's [custom-workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
Enabling this example is optional and is not part of installing AutoTrain-Bench.
