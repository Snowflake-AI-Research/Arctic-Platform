# Set Up the Client

Install the client from a local checkout:

```bash
pip install -e .
```

Create a connection file from the template:

```bash
cp examples/config/connection.json.template /path/to/config.json
```

Fill in the account host, programmatic access token, database, and schema. Do
not commit the resulting file.

Validate and store the config path:

```bash
cortex-training login --config /path/to/config.json
cortex-training capacity
```

See the [CLI reference](../reference/cli.md) for environment variables, local
mock configuration, and one-command overrides.
