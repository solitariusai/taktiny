# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Random Number Generators """
from __future__ import annotations
from collections.abc import Sequence
import jax
from jax.typing import ArrayLike
from jax._src.random.core import PRNGSpecDesc, KeyDTypeLike
from jax.tree_util import register_pytree_node_class
from taktiny.utils.typing import PRNGKey

@register_pytree_node_class
class Rngs:
    def __init__(
        self, seed: ArrayLike, *,
        impl: PRNGSpecDesc | None = None,
        dtype: KeyDTypeLike | None = None
    ) -> None:
        try:
            self._key = jax.random.key(seed, impl=impl, dtype=dtype)
        except TypeError:
            self._key = seed

    def __call__(self) -> PRNGKey:
        self._key, _k = jax.random.split(self._key, 2)
        return _k

    @property
    def key(self) -> PRNGKey:
        """Return the current key without advancing the stream."""
        return self._key

    def tree_flatten(self) -> tuple[tuple[PRNGKey], None]:
        return ((self._key,), None)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: Sequence[PRNGKey],
    ) -> Rngs:
        obj = object.__new__(cls)
        obj._key = children[0]
        return obj
