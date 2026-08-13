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
"""Loss adapters shared by TakTiny trainers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from taktiny.utils.typing import Batch


def _batch_loss_arguments(
    batch: Batch,
) -> tuple[tuple[Batch], dict[str, Any]]:
    return (batch,), {}


class Loss:
    """Adapt a batch mapping to an arbitrary model loss signature.

    ``prepare`` receives the training batch and returns positional and keyword
    arguments for ``loss_fn``. Without a preparation function, the complete
    batch is passed as one positional argument, preserving TakTiny's standard
    ``loss_fn(model, batch)`` contract.

    Args:
        loss_fn: Callable that computes the loss. The model is always passed as
            its first argument.
        prepare: Optional callable returning ``(args, kwargs)`` from a batch.

    Example:
        >>> def prepare(batch):
        ...     return (batch['inputs'],), {'labels': batch['labels']}
        >>> def custom_loss(model, inputs, *, labels):
        ...     return model(inputs, labels=labels)
        >>> loss = Loss(custom_loss, prepare)
    """

    def __init__(
        self,
        loss_fn: Callable[..., Any],
        prepare: Callable[
            [Batch],
            tuple[tuple[Any, ...], Mapping[str, Any]],
        ] | None = None,
    ) -> None:
        if not callable(loss_fn):
            raise TypeError('loss_fn must be callable')
        if prepare is not None and not callable(prepare):
            raise TypeError('prepare must be callable or None')

        self.loss_fn = loss_fn
        self.prepare = (
            _batch_loss_arguments
            if prepare is None
            else prepare
        )

    def __call__(self, model: Any, batch: Batch) -> Any:
        prepared = self.prepare(batch)
        if not isinstance(prepared, tuple) or len(prepared) != 2:
            raise TypeError('prepare must return an (args, kwargs) tuple')

        args, kwargs = prepared
        if not isinstance(args, tuple):
            raise TypeError('prepared args must be a tuple')
        if not isinstance(kwargs, Mapping):
            raise TypeError('prepared kwargs must be a mapping')
        if any(not isinstance(key, str) for key in kwargs):
            raise TypeError('prepared keyword argument names must be strings')

        return self.loss_fn(model, *args, **kwargs)


__all__ = ['Loss']
