# Agent Instructions for `dss-client`

Always run the tests before and after changing any code in `dss_client/`:

```bash
cd /home/repo/dss-client
python3 -m pytest tests/ -v
```

All tests must pass.

## Context

`dss_client/neutrino_client.py` calls into the **Neutrino SNOWAPI** exposed by
Global Services. When changing the wire format, cross-check these:

- SNOWAPI spec (source of truth for the REST schema):
  `/home/repo/snowflake/GlobalServices/modules/snowapi/snowapi-codegen/src/main/openapi/specs/neutrino.yaml`
- Mock SNOWAPI server (for local end-to-end testing):
  `/home/repo/cortex/neutrino/cmd/mock-snowapi/` (see its `README.md`)
