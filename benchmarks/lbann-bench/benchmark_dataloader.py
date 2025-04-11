"""
Data loader for benchmarking
"""

import numpy as np

np.random.seed(20250318)

# ----------------------------------------------
# Layer Normalization Benchmarking

def numpy_layer_norm(x, scale=None, bias=None, eps=1e-5, dims=-1):
    mean = x.mean(axis=dims, keepdims=True)
    std = x.std(axis=dims, ddof=0, keepdims=True)
    result = (x - mean) / (std + eps)
    if scale is not None:
        result *= scale
    if bias is not None:
        result += bias
    return result


_layernorm_sample_dims = (2048, 12288)


def get_layernorm_sample(index):
    # Generate data
    start_dim = -1
    sdim = (len(_layernorm_sample_dims) + start_dim) if start_dim < 0 else start_dim
    dims = tuple(d for d in range(sdim, len(_layernorm_sample_dims)))
    sample = np.random.rand(*_layernorm_sample_dims).astype(np.float32)
    # Generate reference data
    reference = numpy_layer_norm(sample, dims=dims)
    return np.concatenate([sample.flatten(), reference.flatten()])


def num_layernorm_samples():
    return 2


def layernorm_sample_dims():
    return (2 * _layernorm_sample_dims[0] * _layernorm_sample_dims[1], )
