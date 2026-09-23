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
from contextvars import ContextVar, Token
from types import TracebackType

import jax
from jax._src.random.core import KeyDTypeLike, PRNGSpecDesc
from jax.extend.core import get_opaque_trace_state
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from taktiny.utils.typing import PRNGKey


@register_pytree_node_class
class Rngs:
    """A sequential random number generator class for maintaining PRNG state in JAX.

    Args:
        key (ArrayLike): Seed or PRNGKey to initialize the state.
        impl (PRNGSpecDesc | None, optional): PRNG implementation specification. Defaults to None.
        dtype (KeyDTypeLike | None, optional): The dtype of the key array. Defaults to None.
    """

    def __init__(
        self,
        key: ArrayLike,
        *,
        impl: PRNGSpecDesc | None = None,
        dtype: KeyDTypeLike | None = None
    ) -> None:
        try:
            self._key = jax.random.key(key, impl=impl, dtype=dtype)
        except TypeError:
            self._key = jax.numpy.asarray(key)
        self._trace_state = get_opaque_trace_state()

    def __call__(self) -> PRNGKey:
        """Generate a new PRNGKey by splitting the internal key state.

        Returns:
            PRNGKey: The new random key generated.
        """
        self._key, _k = jax.random.split(self._key, 2)
        return _k

    def split_key(self, num_splits: int) -> tuple[jax.Array, ...]:
        """Return independent keys and advance this stream once.

        Args:
            num_splits: Number of keys to return; must be a positive integer.

        Returns:
            A tuple of ``num_splits`` keys. The first key from the split is
            retained as this stream's new state.
        """
        if isinstance(num_splits, bool) or not isinstance(num_splits, int):
            raise TypeError('num_splits must be a positive integer')
        if num_splits < 1:
            raise ValueError('num_splits must be a positive integer')

        splits = jax.random.split(self.key, num_splits + 1)
        self._key = splits[0]
        return tuple(splits[1:])

    def split_rngs(self, num_splits: int) -> tuple[Rngs, ...]:
        """Return independent ``Rngs`` streams and advance this stream once.

        Args:
            num_splits: Number of streams to return; must be a positive integer.

        Returns:
            A tuple of ``num_splits`` streams, each initialized from a
            distinct key returned by :meth:`split_key`.
        """
        return tuple(Rngs(rng) for rng in self.split_key(num_splits))

    @property
    def key(self) -> PRNGKey:
        """Return the current PRNGKey state.

        Returns:
            PRNGKey: The current PRNG state.
        """
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
        obj._trace_state = get_opaque_trace_state()
        return obj


_context_rng: ContextVar[Rngs | None] = ContextVar('taktiny_context_rng', default=None)


def _check_context_rng(rngs: Rngs) -> None:
    if rngs._trace_state != get_opaque_trace_state():
        raise RuntimeError(
            'A context RNG cannot be captured across JAX transformations. '
            'Pass Rngs into the transformed function, use '
            'with set_context_rng(rngs) inside it, and return the updated Rngs.'
        )


class _ContextRngScope:
    """Single-use restoration handle for an immediately installed default."""

    def __init__(self, token: Token[Rngs | None], rngs: Rngs | None) -> None:
        self._token = token
        self._rngs = rngs
        self._entered = False

    def __enter__(self) -> Rngs | None:
        if self._entered:
            raise RuntimeError('set_context_rng scopes cannot be reused')
        self._entered = True
        return self._rngs

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        _context_rng.reset(self._token)


def set_context_rng(rngs: Rngs | None) -> _ContextRngScope:
    """Set the default runtime RNG, optionally restoring it with ``with``.

    The stream is installed immediately. Used as a context manager, the
    previous stream is restored on exit, including exceptional exits. Nested
    scopes do not reset the keys of previous streams. Pass None to clear or
    temporarily disable the default. Explicit module RNGs take precedence.
    Parameter initializers and raw jax.random functions are unaffected.

    Defaults are local to the current Python context, not process-wide.
    Async tasks inherit their parent's binding, including the mutable Rngs
    object; install separate streams in tasks that need independent state.

    Do not capture this mutable default in jit/grad/vmap. Pass RNG state into
    the transformed function, establish a scope inside that function, and
    return its updated state. Each mapped sample or scan iteration needs its
    own split key or explicitly threaded RNG state.

    Examples:
        >>> from taktiny import nn
        >>> _ = nn.set_context_rng(nn.Rngs(42))
        >>> with nn.set_context_rng(rngs=nn.Rngs(123)):
        ...     key = nn.get_context_rng()()
        >>> _ = nn.set_context_rng(None)

        >>> @jax.jit
        ... def step(x, rngs):
        ...     with nn.set_context_rng(rngs):
        ...         y = nn.Dropout(0.5)(x)
        ...     return y, rngs
        >>> y, rngs = step(jax.numpy.ones((8,)), nn.Rngs(0))
        >>> y, rngs = step(jax.numpy.ones((8,)), rngs)
    """
    if rngs is not None:
        if not isinstance(rngs, Rngs):
            raise TypeError('rngs must be an Rngs or None')
            
        _check_context_rng(rngs)
    return _ContextRngScope(_context_rng.set(rngs), rngs)


def get_context_rng() -> Rngs:
    """Return the runtime stream without consuming a key.

    Call the returned stream to obtain a fresh key for custom sampling.
    Raises ValueError if no default is installed, or RuntimeError when an
    outer mutable stream is accessed across a JAX transformation boundary.
    """
    rngs = _context_rng.get()
    if rngs is None:
        raise ValueError(
            'rngs is required for stochastic operations: pass an explicit '
            'Rngs or install one with set_context_rng(rngs).'
        )
    _check_context_rng(rngs)
    return rngs


__all__ = ['Rngs', 'get_context_rng', 'set_context_rng']
