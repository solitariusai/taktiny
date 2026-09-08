# Data Loading & Transforms

`taktiny.data` contains high-performance utilities for token packing, document transformations, and dataset batching.

## Document Templates & Token Packing

Packing multiple variable-length documents into a fixed sequence length removes padding overhead:

```python
from taktiny.data import pack_sequences

# Efficient sequence packing for causal language modeling
```

## Transforms Pipeline

Custom data transformations and batch iterators allow seamless streaming from Hugging Face Datasets or local files into JAX arrays.
