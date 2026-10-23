# Public TRAIN integrity

The bundled data preparation writes a root `manifest.json` in the TRAIN directory.
Task builders own the data formats and source provenance. The shared local verifier checks
the complete file set, byte lengths, and SHA-256 identities. It uses the Python
standard library and adds no scheduler, container, or sandbox dependency.

The manifest may include task-owned **public** metadata. Its `integrity` field
has this format:

```json
{
  "integrity": {
    "schema_version": 1,
    "algorithm": "sha256",
    "files": [
      {"file": "train.json", "bytes": 123, "sha256": "<64 lowercase hex digits>"}
    ]
  }
}
```

File paths are relative to the TRAIN root and use `/`. Every regular file,
including document indexes and nested manifests, must appear exactly once.
Only the root manifest itself is excluded, avoiding a self-referential hash.
Missing or additional files, unsafe paths, symlinks, hard links, and special
files are rejected. Empty directories are not part of the file identity.

Builders can call `envcore.training_data.write_training_manifest()`
after writing all public files. Hidden evaluation data and metadata must stay
outside this directory. Source inventories used to claim a particular upstream
revision must be independently pinned by the task; generating an inventory of
arbitrary local bytes does not establish their provenance.

The local `ngrove prepare` and `start` launchers verify TRAIN before
Worker setup or task reveal. Each `start` performs its own check instead of
reusing an earlier preparation result. This local policy reads every TRAIN file
in full; plan for the disk I/O of both checks. A missing or legacy manifest
without an integrity inventory requires data preparation again in this path.

The trusted launcher passes `training_data_verification` to `Supervisor`.
Core requires a verified status and a digest matching the current root manifest,
copies the evidence into trusted state and binds the run to that identity.
It does not interpret the file inventory or scan the dataset.
Other trusted launchers may obtain evidence from their Worker or immutable data
view without using this local scanning implementation. The evidence is a trusted
composition input, never an Agent-supplied claim.

The trusted result records `training_data_verification`: verification status,
inventory version, manifest SHA-256, file count, and total file bytes excluding
the root manifest. `artifact_identities.train_manifest_sha256` binds that same
verified manifest. Existing version 12 results may lack the new audit field;
absence is not evidence of verification.

Verification checks changes while reading, but it does not enforce isolation or
authenticate a manifest that an operator replaced together with its files. The
Worker deployment must expose the verified TRAIN view read-only and keep it
immutable throughout research. Core still reads the root manifest through the
current deployment-provided TRAIN `Path` to bind the supplied evidence; this
adjustment does not introduce a remote data protocol. Hidden ANALYSIS/TEST bytes
are not read by this check and
remain subject to post-sealing materialization and evaluation rules.
