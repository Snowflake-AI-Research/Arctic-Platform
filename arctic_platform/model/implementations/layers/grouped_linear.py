import torch
from torch import nn


class GroupedLinear(nn.Linear):
    """Block-diagonal linear that projects each of `n_groups` input groups independently to `out_features / n_groups` channels.

    Input is `(..., n_groups, in_features_per_group)`, output `(..., n_groups, out_features / n_groups)`.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class BlockFp8GroupedLinear(nn.Module):
    """GroupedLinear whose weight stays block-scaled FP8.

    ``o_a_proj`` is this shape in the official DeepSeek-V4-Flash checkpoint.
    It is not a LoRA target: a PEFT linear wrapper would replace the grouped
    ``bmm`` with ``F.linear``.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, *, block_size: int):
        super().__init__()
        from arctic_platform.model.implementations.fp8 import fp8_scale_shape
        from arctic_platform.model.implementations.fp8 import mark_keep_fp32

        self.in_features_per_group = in_features_per_group
        self.out_features = out_features
        self.n_groups = n_groups
        self.block_size = block_size
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features_per_group, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale_inv = mark_keep_fp32(
            nn.Parameter(
                torch.empty(fp8_scale_shape(self.weight.shape, block_size), dtype=torch.float32),
                requires_grad=False,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from arctic_platform.model.implementations.fp8 import fp8_linear

        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        grouped = x.reshape(-1, self.n_groups, hidden_dim)
        weight = self.weight.view(self.n_groups, -1, hidden_dim)
        rows_per_group = weight.shape[1]
        scale_rows = (rows_per_group + self.block_size - 1) // self.block_size
        scale = self.weight_scale_inv.view(self.n_groups, scale_rows, -1)
        outputs = [
            fp8_linear(grouped[:, group], weight[group], scale[group], self.block_size)
            for group in range(self.n_groups)
        ]
        return torch.stack(outputs, dim=1).reshape(*input_shape, self.n_groups, -1)
