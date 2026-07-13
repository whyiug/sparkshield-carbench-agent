# SparkShield Track 1 Release Provenance

This repository is a parentless, sanitized source snapshot prepared for public
review. Internal experiment logs, handoff material, local paths, and release
credentials are not part of the snapshot.

## Frozen Artifact

- Frozen build commit: `396abf709c57d47421771fab5af22d12b69d01ac`
- Published image:
  `ghcr.io/whyiug/sparkshield-carbench-agent@sha256:c290a2873d5e5b00badddf8314568150073c9c66812fd39fcf39387aca2d26e3`
- Image platform: `linux/amd64`
- Starter base: `CAR-bench/car-bench-ijcai` at
  `1d1ea6ee7bb68461bfaed54234a1c6734703cbad`

The published image was built from the frozen build commit, not from the public
snapshot commit. The public snapshot preserves all 62 Track 1 Docker build
definition and copied runtime files byte-for-byte. The manifest is sorted by
path, uses LF endings, and formats each line as
`MODE<two spaces>SIZE<two spaces>CONTENT_SHA256<two spaces>PATH`. Its SHA-256 is:

```text
TRACK1_DOCKER_INPUT_COUNT=62
TRACK1_DOCKER_INPUT_MANIFEST_SHA256=f0be6a4199e008cf66d80f7fe51bd4c5cbbe81688f5aa39340e2f88262aa65db
```

The sanitization changes only files excluded by the Track 1 Dockerfile-specific
ignore rules. Rebuilding from this snapshot should reproduce the Agent runtime
contents, but registry metadata, SBOM, or build provenance can still produce a
different image manifest digest. The published digest above remains the
submission artifact of record.

The repository retains the starter kit's MIT license. Public development tests
are included for reproducibility; no hidden test data or evaluator modification
is included.
