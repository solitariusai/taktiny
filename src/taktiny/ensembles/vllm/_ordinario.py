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
"""Runtime boundary between a Taktiny model and vLLM."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable


@runtime_checkable
class VLLMEngine(Protocol):
    """Platform-specific engine used by :class:`VLLM`.

    Implementations own the actual vLLM runtime and its TPU or GPU weight
    transport. The wrapper deliberately keeps this protocol independent of
    vLLM imports so importing Taktiny does not require an optional runtime.
    """

    def start(self) -> None:
        """Start the inference runtime and make it ready for requests."""

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate completions using the currently synchronized policy."""

    def sync(
        self,
        model: Any,
        *,
        policy_version: int,
        **kwargs: Any,
    ) -> None:
        """Synchronize ``model`` into the inference runtime."""

    def close(self) -> None:
        """Release resources owned by the inference runtime."""


class VLLM:
    """Run inference and RL rollouts for a Taktiny model through vLLM.

    ``VLLM`` is a runtime wrapper, not a Taktiny ``nn.Module``. The original
    trainable model remains available through :attr:`model`; generation is
    delegated to a platform-specific engine and :meth:`sync` publishes model
    updates to that engine.

    Args:
        model: Trainable Taktiny model wrapped by this runtime.
        engine: Existing engine implementing :class:`VLLMEngine`.
        engine_factory: Callable receiving ``model`` and the remaining engine
            options and returning a :class:`VLLMEngine`.
        auto_start: Whether to start the engine during construction. When
            false, the first call to :meth:`generate` or :meth:`sync` starts
            it automatically.
        **engine_options: Platform-specific options forwarded to
            ``engine_factory``.

    Example:
        >>> runtime = VLLM(model, tensor_parallel_size=8)
        >>> output_ids = runtime.generate(input_ids, max_new_tokens=128)
        >>> policy_version = runtime.sync()
    """

    def __init__(
        self,
        model: Any,
        *,
        engine: VLLMEngine | None = None,
        engine_factory: Callable[..., VLLMEngine] | None = None,
        auto_start: bool = True,
        **engine_options: Any,
    ) -> None:
        if model is None:
            raise ValueError('model is required')
        if engine is not None and engine_factory is not None:
            raise ValueError(
                'Pass either engine or engine_factory, not both'
            )
        if not isinstance(auto_start, bool):
            raise TypeError('auto_start must be a boolean')

        self.model = model
        self._engine = None
        self._engine_factory = engine_factory
        self._engine_options = dict(engine_options)
        self._started = False
        self._closed = False
        self._policy_version = 0

        if engine is not None:
            self._engine = self._validate_engine(engine)
        if auto_start:
            self.start()

    @staticmethod
    def _validate_engine(engine: Any) -> VLLMEngine:
        missing = [
            name
            for name in ('start', 'generate', 'sync', 'close')
            if not callable(getattr(engine, name, None))
        ]
        if missing:
            methods = ', '.join(missing)
            raise TypeError(
                'vLLM engine is missing required callable methods: '
                f'{methods}'
            )
        return engine

    def _create_engine(self) -> VLLMEngine:
        engine_factory = self._engine_factory
        if engine_factory is None:
            from ._local import LocalVLLMEngine

            engine_factory = LocalVLLMEngine
        engine = engine_factory(
            self.model,
            **self._engine_options,
        )
        return self._validate_engine(engine)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError('vLLM runtime is closed')

    @property
    def engine(self) -> VLLMEngine:
        """Return the engine, constructing it on first access."""
        self._require_open()
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def engine_options(self) -> dict[str, Any]:
        """Return a copy of the platform-specific engine options."""
        return dict(self._engine_options)

    @property
    def policy_version(self) -> int:
        """Version of the policy most recently synchronized to vLLM."""
        return self._policy_version

    @property
    def started(self) -> bool:
        """Whether the inference runtime has been started."""
        return self._started

    @property
    def closed(self) -> bool:
        """Whether this wrapper has been permanently closed."""
        return self._closed

    def start(self) -> Self:
        """Start the underlying engine once and return this wrapper."""
        self._require_open()
        if not self._started:
            self.engine.start()
            self._started = True
        return self

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate through vLLM using its current policy weights."""
        self.start()
        return self.engine.generate(*args, **kwargs)

    def sync(
        self,
        *,
        policy_version: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Synchronize the trainable model into vLLM.

        The policy version advances only after the engine reports a successful
        synchronization. This lets RL trainers reject rollouts generated by a
        stale or partially updated policy.

        Args:
            policy_version: Explicit version assigned to the synchronized
                policy. When omitted, the current version is incremented.
            **kwargs: Transport-specific synchronization options.

        Returns:
            The newly synchronized policy version.
        """
        if policy_version is None:
            policy_version = self._policy_version + 1
        elif (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version <= self._policy_version
        ):
            raise ValueError(
                'policy_version must be an integer greater than the current '
                'policy version'
            )
        self.start()
        self.engine.sync(
            self.model,
            policy_version=policy_version,
            **kwargs,
        )
        self._policy_version = policy_version
        return policy_version

    def close(self) -> None:
        """Close the engine once and make the wrapper unusable."""
        if self._closed:
            return
        if self._engine is not None:
            self._engine.close()
        self._started = False
        self._closed = True

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
