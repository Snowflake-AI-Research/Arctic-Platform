# Agent Instructions for `cortex-training`

Always run the tests before and after changing any code in `cortex_training/`:

```bash
cd /home/repo/Arctic-Platform/cortex_training
python3 -m pytest tests/ -v
```

All tests must pass.

## Context

`cortex_training/client.py` calls into the **Cortex Training SNOWAPI** exposed by
Global Services. When changing the wire format, cross-check these:

- SNOWAPI spec (source of truth for the REST schema):
  `/home/repo/snowflake/GlobalServices/modules/snowapi/snowapi-codegen/src/main/openapi/specs/neutrino.yaml`
- Mock SNOWAPI server (for local end-to-end testing):
  `/home/repo/cortex/neutrino/cmd/mock-snowapi/` (see its `README.md`)
