# SparkShield Track 1 Release Provenance

This repository's history begins with a parentless, sanitized source snapshot
prepared for public review. Internal experiment logs, handoff material, local
paths, and release credentials are not part of the repository.

## Current Release Artifact

- Runtime build commit: `f5320cbf77e9aa33893c3869d0d9b0be56d0bebb`
- Immutable release tag: `final-20260719-optionc1`
- Published image:
  `ghcr.io/whyiug/sparkshield-carbench-agent@sha256:4f16a23a086c0d70867ba15036ebb0a5897d241acd49cb487f52e412c6fe70b4`
- Image platform: `linux/amd64`
- Starter base: `CAR-bench/car-bench-ijcai` at
  `1d1ea6ee7bb68461bfaed54234a1c6734703cbad`

The published image was built directly from the public runtime build commit.
Metadata-only descendants preserve all 62 Track 1 Docker build-definition and
copied runtime files byte-for-byte. The manifest is sorted by path, uses LF
endings, and formats each line as
`MODE<two spaces>SIZE<two spaces>CONTENT_SHA256<two spaces>PATH`. Its SHA-256 is:

```text
TRACK1_DOCKER_INPUT_COUNT=62
TRACK1_DOCKER_INPUT_MANIFEST_SHA256=e47394e6be69092f55f07f1a78df2c0c65044a6c02be6120b9329e607f549a78
```

The release adds `curl` to the final runtime stage because the unmodified starter
Compose flow executes its Agent Card healthcheck with container-local `curl`.
This packaging-only correction does not change Agent decision logic. Registry
metadata, SBOM, or build provenance can still produce a different manifest digest
on a later rebuild; the published digest above is the submission artifact of
record.

The repository retains the starter kit's MIT license. Public development tests
are included for reproducibility; no hidden test data or evaluator modification
is included.
