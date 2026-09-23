# Base Classes

Core abstractions for neural network modules and parameters in Taktiny.

```{eval-rst}
.. autoclass:: taktiny.nn.Module
   :members: train, eval, state_dict, load_state_dict, flat_state_dict, load_flat_state_dict, flat_parameter_dict

.. autoclass:: taktiny.nn.Parameter
   :members: value

.. autofunction:: taktiny.nn.module

.. autoclass:: taktiny.nn.Pytree

.. autoclass:: taktiny.nn.Rngs
   :members: key, split_key, split_rngs
```
