from typing import List, Optional, Tuple

import torch


def esimd_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, flag: int, len: int
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.esimd_add(a, b, c, flag, len)


def esimd_mul_lgrf(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, flag: int, len: int
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.esimd_mul_lgrf(a, b, c, flag, len)