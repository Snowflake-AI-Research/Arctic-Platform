# Recipe Template

Each recipe directory should contain:

```text
README.md
recipe.yaml
train.py, evaluate.py, or another clear entry point
configs/
optional notebook.ipynb
```

The README should cover:

1. Outcome and training method
2. Prerequisites and access
3. Dataset and model
4. Hardware requirements
5. Exact run command
6. Expected outputs and metrics
7. Configuration variations
8. Checkpoint and evaluation workflow
9. Troubleshooting
10. Provenance and last validation date

The metadata should be machine-readable so model compatibility and recipe
catalog pages can eventually be generated.
