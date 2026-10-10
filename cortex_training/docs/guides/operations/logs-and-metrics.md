# Logs and Metrics

Download complete execution logs:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/logs
```

Tail a running job in the terminal:

```bash
cortex-training tui JOB_ID
```

Recipe-level metrics are written under each recipe's `log_path`. GPU
utilization, memory, throughput, tokens per second, and MFU are roadmap items
and are not yet available as a complete documented workflow.
