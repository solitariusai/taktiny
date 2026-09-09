# Layers

Taktiny separates general neural-network primitives in `taktiny.nn` from
transformer-oriented components in `taktiny.layers`.

## Neural-Network Primitives

### Linear

`nn.Linear` supports scalar or tuple input/output feature shapes, optional
bias, logical axis names, Qwix weights, and a custom `dot_general` callable.

```python
from taktiny import nn

rngs = nn.Rngs(42)
projection = nn.Linear(
    768,
    (12, 64),
    bias=False,
    dtype="bfloat16",
    rngs=rngs,
    axis_names=("embed", "heads", "head_dim"),
)
```

When `weight.value` is a Qwix `QArray`, the forward pass dispatches to
`qwix.dot_general`. Otherwise it uses the configured custom operation or
`jax.lax.dot_general`.

### Embedding

```python
embedding = nn.Embedding(
    num_embeddings=32_000,
    embedding_dim=768,
    dtype="bfloat16",
    rngs=rngs,
)
```

The parameter is named `embedding`, not `weight`. Quantized lookup dequantizes
the selected rows.

### Normalization

```python
rms_norm = nn.RMSNorm(768, eps=1e-6, dtype="float32")
layer_norm = nn.LayerNorm(768, eps=1e-5)
```

`RMSNorm` preserves the input dtype after its normalization calculation.

## Containers

| Container | Behavior |
| --- | --- |
| `nn.List` | Stores independent modules; indexing and iteration are explicit |
| `nn.SeqStack` | Stacks matching module PyTrees and runs a supplied body with `jax.lax.scan` |
| `nn.Stack` | Stacks matching module PyTrees and vectorizes their direct calls with `jax.vmap` |
| `nn.Sequential` | Calls a sequence of modules in order |

`SeqStack` and `Stack` receive module instances, for example:

```python
layers = nn.SeqStack([
    nn.Linear(8, 8, rngs=rngs)
    for _ in range(4)
])


def step(layer, carry):
    output = layer(carry)
    return output, None


output, intermediates = layers(step, input_array)
```

Every stacked instance must have the same PyTree structure. `SeqStack` is for
dependent sequential computation; `Stack` is for independent vectorized
computation.

## Transformations

`taktiny.tt` aliases `taktiny.transforms` and exposes decorator or functional
forms of `vmap` and `scan`:

```python
from taktiny import tt

mapped = tt.vmap(function, in_axes=0, out_axes=0)
scanned = tt.scan(function)
```

These are module-aware JAX transformations. They are unrelated to dataset
preprocessing; data operations live in `taktiny.data_utils`.

## Transformer Layers

### Attention

`layers.Attention` implements MHA, MQA, and GQA with separate query, key,
value, and output projections. It supports optional Q/K normalization, RoPE,
sliding windows, attention soft-capping, dropout, quantization, and selectable
kernel dispatch.

```python
from taktiny import layers

attention = layers.Attention(
    hidden_size=768,
    num_heads=12,
    num_kv_heads=4,
    head_dim=64,
    pos_emb=layers.RotaryEmbedding(64),
    dtype="bfloat16",
    rngs=rngs,
)
```

The regular call returns `(output, new_cache)` and accepts context, masks,
causal mode, KV cache, positions, output sharding, and a kernel name. Normal
model code supplies these values through `TransformerDecoderLayer` and
`TransformerContext`.

### Feed-Forward Layers

- `layers.GateMLP` implements the ordinary gated MLP used by dense decoders.
- `layers.FusedGateMLP` stores a fused gate/up projection and exposes grouped
  matrix multiplication and routing helpers.
- `layers.MoeFFN` implements routed experts and exposes the same low-level MoE
  helpers.

The kernel entry points `apply_gmm`, `apply_route`, and `apply_unroute` belong
to `FusedGateMLP` and `MoeFFN`; `GateMLP` does not provide them.

### Rotary Embeddings

```python
rope = layers.RotaryEmbedding(
    dim=64,
    max_position_embeddings=8192,
    base=10_000.0,
)
```

`RotaryEmbedding` accepts scalar cached offsets, one offset per batch row, or
two-dimensional per-token positions. Llama 3 frequency scaling is selected by
a `rope_scaling` mapping whose `rope_type` is `"llama3"`.
