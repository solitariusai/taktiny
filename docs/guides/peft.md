# Parameter-Efficient Fine-Tuning (PEFT)

The `taktiny.takt` package provides adapters that transform existing models by injecting low-rank or parameter-efficient modules and automatically freezing the pre-existing base weights.

## Supported Adapters

- **`LoRAAdapter`**: Standard Low-Rank Adaptation.
- **`DoRAAdapter`**: Weight-Decomposed Low-Rank Adaptation (decouples magnitude and direction).
- **`AdaLoRAAdapter`**: Adaptive Low-Rank Adaptation (allocates rank dynamically).
- **`LoHaAdapter`**: Low-Rank Hadamard Product parameterization.
- **`LoKrAdapter`**: Low-Rank Kronecker Product parameterization.
- **`VeRAAdapter`**: Vector-based Random Matrix Adaptation (shares fixed random projections across all linear layers, learning small diagonal scaling vectors).

## Usage Example

```python
from taktiny import nn
from taktiny.takt import Takt, LoRAAdapter

# Define or load an existing model
model = nn.Sequential([
    nn.Linear(64, 128, rngs=nn.Rngs(0)),
    nn.Linear(128, 10, rngs=nn.Rngs(1)),
])

# Create adapter targeting specific layers via regex
adapter = LoRAAdapter(
    targets="0",  # Matches the first layer in Sequential
    rank=8,
    alpha=16.0,
    rngs=nn.Rngs(42),
)

# Injects adapter and freezes existing weights
adapted_model = Takt.apply_adapter(model, adapter)
```

## VeRA Shared Projections

`VeRAAdapter` automatically computes the maximum input and output dimensions across all matching layers, creating shared `vera_A` and `vera_B` projections that are reused across all targets to drastically reduce parameter count:

```python
from taktiny.takt import VeRAAdapter

vera = VeRAAdapter(
    targets=".*linear.*",
    rank=4,
    rngs=nn.Rngs(99),
)
adapted_model = Takt.apply_adapter(model, vera)
```
