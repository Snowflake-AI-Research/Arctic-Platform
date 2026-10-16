This figure records one completed comparison on one H200 GPU with Qwen3-0.6B,
rank-4 LoRA and BF16, using fixed synthetic pairs and two training steps. The
reference graph construction was reversed to match the observed candidate
backward accumulation order. All captured parameter gradients, updates and
Adam moments matched exactly in that diagnostic. This does not establish
four-GPU optimizer-update parity or close failures in the original schedules.

The adjacent JSON contains the plotted samples, complete comparison counts and
hashes binding the measured result, provenance and tensor artifacts. The plot
script renders those saved measurements; it does not run models or training.

Regenerate from this directory with Python 3.10.9, Matplotlib 3.10.9, NumPy 2.2.6
and Pillow 12.3.0:

```sh
python plot-pr-parity.py
```

The script writes PNG, SVG and PDF files alongside the saved data. The checked-in
PNG SHA-256 is `07a4f36d3868e60db49316dbf557f17c61ac4aa7292517e02887128080ff40ef`.

The measurement used Arctic-Platform base commit
`4f50c5bbea3f1ba102daba675bd480796c6dfea5` plus the uncommitted DPO source
snapshot `47da219d86deee98e6d857ec7ff8b7ad44862181b86a97e7328c38595c55ae7a`.
It predates the response-only metrics callback in this publication candidate.
The JSON identifies each tested source file, the digest-only runtime identity,
package versions, optimizer configuration and observed backward window order.
Full result/tensor hashes identify retained historical measurements; the complete
result and tensor files are not bundled here.

`historical_provenance_json` preserves the exact public-safe provenance text;
its UTF-8 SHA-256 equals `provenance_sha256`. The six computational function AST
hashes are equal for the tested and reviewed modules, excluding registration
decorators: reporting changed, while loss/config/mask/packed arithmetic did not.
To check the reviewed hashes with Python 3.12 from the repository root:

```python
import ast
import hashlib
import json
from pathlib import Path

data = json.loads(Path("docs/evidence/dpo-gradient-parity/pr-parity-chart-data.json").read_text())
assert hashlib.sha256(data["historical_provenance_json"].encode("utf-8")).hexdigest() == data["provenance_sha256"]
nodes = ast.parse(Path(data["computational_equivalence"]["module"]).read_text())
for node in nodes.body:
    if isinstance(node, ast.FunctionDef) and node.name in data["computational_equivalence"]["reviewed_function_AST_sha256"]:
        node.decorator_list = []
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
        assert digest == data["computational_equivalence"]["reviewed_function_AST_sha256"][node.name]
```
