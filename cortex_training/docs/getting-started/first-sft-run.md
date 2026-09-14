# Run Your First SFT Job

The conversational SFT recipe is the current shortest end-to-end training path.
It creates a training job, loads a chat dataset, submits forward/backward
batches, applies optimizer steps, and cancels the job when complete.

Install the recipe dependency:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Start with a short run:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  max_steps=2
```

The default requests eight GPUs. Check capacity first and adjust `n_gpus`,
`batch_size`, and `micro_batch_size` together when needed.

For a longer run, dataset changes, dense training, and MoE configuration, see
the [conversational SFT recipe](../../recipes/sft/conversational/README.md).

Evaluation before and after training is still planned. Until that workflow is
implemented, confirm that training loss is recorded and decreases during a
longer run.
