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
"""Adapter definitions used by :class:`taktiny.takt.Takt`."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from taktiny.nn.base import Module


class AdapterBase:
    """Describe how to replace matching modules with adapter modules.

    Subclasses set ``_adapter`` to a ``Module`` type whose first positional
    argument is the target module. Arguments passed after ``targets`` are
    forwarded to every replacement. An adapter may keep shared values on
    itself and pass them through those keyword arguments.

    ``BaseAdapter`` is intentionally not an ``nn.Module``. If it creates a
    shared ``Parameter``, it must pass that parameter to its replacements so
    it is part of the model parameter tree.
    """

    _adapter: type[Module] | None = None

    def __init__(
        self,
        targets: str | list[str] | tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._adapter is None:
            raise TypeError(
                f'{type(self).__name__} must define an _adapter module type'
            )

        normalized_targets = (
            (targets,)
            if isinstance(targets, str)
            else tuple(targets)
        )
        if not normalized_targets:
            raise ValueError('targets must contain at least one pattern')
        if not all(
            isinstance(target, str) and target
            for target in normalized_targets
        ):
            raise TypeError('targets must contain non-empty strings')

        try:
            patterns = tuple(
                re.compile(target)
                for target in normalized_targets
            )
        except re.error as error:
            raise ValueError(f'Invalid adapter target pattern: {error}') from error

        self.targets = normalized_targets
        self._patterns = patterns
        self._adapter_args = args
        self._adapter_kwargs = kwargs

    def matches(self, module_path: str) -> bool:
        """Return whether this adapter applies to ``module_path``."""
        return any(pattern.search(module_path) for pattern in self._patterns)

    def prepare(self, targets: Sequence[tuple[str, Module]]) -> None:
        """Prepare shared state after target discovery and before injection.

        Most adapters need no preparation. An adapter with global values may
        inspect all matched modules here, then pass those values to every
        replacement through ``self._adapter_kwargs``.
        """
        del targets

    def build(self, target: Module, *, module_path: str) -> Module:
        """Create the replacement for one matching target module."""
        del module_path
        replacement = self._adapter(
            target,
            *self._adapter_args,
            **self._adapter_kwargs,
        )
        if not isinstance(replacement, Module):
            raise TypeError(
                f'{type(self).__name__}._adapter must create an nn.Module, '
                f'got {type(replacement).__name__}'
            )
        return replacement

    def __call__(self, **kwargs: Any) -> Any:
        """Run an optional post-step update; ordinary adapters do nothing."""
        del kwargs
        return None


__all__ = ['AdapterBase']
