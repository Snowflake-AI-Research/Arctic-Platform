# Prerequisites

Before running a training recipe, you need:

- Python 3.8 or later
- Access to a Snowflake account with the Cortex Training endpoint enabled
- A programmatic access token
- A database and schema containing the endpoint
- Sufficient reserved GPU capacity for the selected recipe

Check current capacity after installing and authenticating:

```bash
cortex-training capacity
```

Model-specific GPU requirements are not yet fully validated. Treat recipe
defaults as starting configurations, not guaranteed minimums.
