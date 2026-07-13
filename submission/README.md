# Track 1 Submission Material

[`scenario.toml`](scenario.toml) is the resolved organizer scenario for the
digest-pinned SparkShield Track 1 image. It contains environment-variable
references only and no credential values.

The scenario intentionally selects the hidden split for organizer evaluation.
Do not pass it to a local runner. Local review is limited to syntax and contract
validation:

```bash
python - <<'PY'
import tomllib
from pathlib import Path

tomllib.loads(Path("submission/scenario.toml").read_text(encoding="utf-8"))
print("TOML_PARSE=PASS")
PY

python scripts/validate_submission.py \
  --require-resolved-image submission/scenario.toml
```

Selected non-secret runtime values:

- Model: `gemini/gemini-3.5-flash`
- Provider route: native Gemini
- Temperature: `0`
- Structured output: `json_schema`
- Critic: disabled
- Timeout: 90 seconds per model call
- Retries: 2
- Soft Agent step limit: 49
- Agent user-turn limit: 24

Artifact and source-equivalence details are recorded in
[`RELEASE_PROVENANCE.md`](../RELEASE_PROVENANCE.md).
