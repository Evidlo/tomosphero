#!/usr/bin/env python3

import torch
from functools import wraps
from torch.overrides import TorchFunctionMode
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar, Union

# We still need a fix for the vmap
# related issue: https://github.com/pytorch/pytorch/issues/124423
class TransformGetSetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[idx] or A[idx] += b, where idx is a scalar tensor.
    # When idx is a scalar tensor, Torch implicitly convert it to a python
    # scalar and create a view of A.
    # Workaround: We convert the scalar tensor to a 1D tensor with one element.
    # That is, we convert A[idx] to A[idx[None]][0], A[idx] += 1 to A[idx[None]] += 1.
    # This is a temporary solution until the issue is fixed in PyTorch.
    def __torch_function__(self, func, types, args, kwargs=None):
        # A[idx]
        if func == torch.Tensor.__getitem__:
            x, index = args
            new_index, any_scalar = _transform_scalar_index(index)
            x = func(x, tuple(new_index), **(kwargs or {}))
            if any_scalar:
                x = x.squeeze(0)
            return x
        # A[idx] = value
        elif func == torch.Tensor.__setitem__:
            x, index, value = args
            new_index, _ = _transform_scalar_index(index)
            return func(x, new_index, value, **(kwargs or {}))

        return func(*args, **(kwargs or {}))



def _transform_scalar_index(ori_index: Sequence[Any | torch.Tensor] | Any | torch.Tensor):
    if isinstance(ori_index, Sequence):
        index = tuple(ori_index)
    else:
        index = (ori_index,)
    any_scalar_tensor = False
    new_index = []
    for idx in index:
        if isinstance(idx, torch.Tensor) and idx.ndim == 0:
            new_index.append(idx[None])
            any_scalar_tensor = True
        else:
            new_index.append(idx)
    if not isinstance(ori_index, Sequence):
        new_index = new_index[0]
    return new_index, any_scalar_tensor


@wraps(torch.vmap)
def vmap(*args, **kwargs) -> Callable:
    """Fix the `torch.vmap`'s issue with __getitem__ and __setitem__.
    Related issue: https://github.com/pytorch/pytorch/issues/124423.
    """

    vmapped = torch.vmap(*args, **kwargs)

    def wrapper(*args, **kwargs):
        with TransformGetSetItemToIndex():
            return vmapped(*args, **kwargs)

    return wrapper