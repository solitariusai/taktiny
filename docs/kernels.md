# Kernels

Low-level implementations live under `taktiny.kernels` (plural). Public layer
classes provide validated dispatch entry points so callers do not need to
depend directly on each kernel module.

These APIs are experimental. Shape, mask, dtype, and platform constraints are
part of each kernel contract; selecting a kernel is not a promise that it is
faster on every input or backend.

## Attention Dispatch

Query, key, and value use `[batch, sequence, heads, head_dim]` layout:

```python
from taktiny.layers import Attention

output = Attention.apply(
    query,
    key,
    value,
    kernel="dot_product",
    is_causal=True,
)
```

Supported dispatch names are:

| Kernel | Current contract |
| --- | --- |
| `dot_product` | JAX `dot_product_attention`; supports mask, bias, causality, and segment IDs |
| `flash` | Block-masked pure-JAX implementation; supports dense/shared masks and segment IDs, but not additive bias |
| `splash` | Current reference implementation with dense masks; not the TPU Mosaic Splash kernel directly |
| `ragged` | Decode-only kernel; query length must be one and `lengths` must be `int32[batch]` |
| `ring` | Requires a prebuilt `ring_kernel`; masking is fixed when that kernel is constructed |

MHA and GQA are accepted when the number of query heads is divisible by the
number of key/value heads.

Direct class methods are also available:

```python
flash = Attention.apply_flash_attention(query, key, value)
ragged = Attention.apply_ragged_attention(
    decode_query,
    key_cache,
    value_cache,
    lengths=prefix_lengths,
)
splash = Attention.apply_splash_attention(query, key, value, mask=mask)
ring = Attention.apply_ring_attention(
    query,
    key,
    value,
    ring_kernel=prebuilt_ring_kernel,
)
```

The ring implementation and the lower-level Mosaic/Pallas paths have
additional mesh and accelerator requirements. Validate them on the actual TPU
or GPU topology rather than relying on CPU interpretation as a performance
test.

## MoE Dispatch

`FusedGateMLP` and `MoeFFN` expose grouped matrix multiplication and token
routing operations:

```python
from taktiny.layers import MoeFFN

sorted_tokens, group_sizes = MoeFFN.apply_route(
    tokens,
    expert_indices,
    num_groups=num_experts,
)
expert_output = MoeFFN.apply_gmm(
    sorted_tokens,
    expert_weights,
    group_sizes,
)
output = MoeFFN.apply_unroute(expert_output, expert_indices)
```

`MoeFFN.apply(..., kernel="gmm")` is the unified GMM dispatcher. The underlying
Megablox implementation uses TPU Mosaic/Pallas where supported and retains its
own layout and dtype constraints. Routing helpers have custom differentiation
behavior for the sorting operation.

`GateMLP` is an ordinary dense block and does not expose these class methods.

## Embedding Dispatch

```python
from taktiny.nn import Embedding

values = Embedding.apply(
    table,
    indices,
    kernel="gather_reduce",
    weights=weights,
    reduce_group_size=1,
)
```

`gather_reduce` uses ordinary indexed gathering as a non-TPU fallback and the
SparseCore gather-reduce implementation on TPU. Ragged gather is exposed
directly because it needs offsets and lengths:

```python
values = Embedding.apply_ragged_gather(
    table,
    offsets,
    lengths,
)
```

## Testing Kernels

The repository tests compare dispatch paths with small JAX references and
exercise validation failures. Hardware-specific performance and supported
block configurations still need accelerator tests; passing the CPU reference
suite is a correctness check, not an accelerator benchmark.
