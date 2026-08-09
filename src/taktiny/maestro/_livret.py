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
"""Module for register any model architectures"""
from __future__ import annotations
from typing import Any

class Repertoire:
    def __init__(self) -> None:
        self._repertoire = {}
        self._classes = set()

    def register(self, key: Any, cls) -> None:
        registered = self._repertoire.get(key)
        if registered is not None and registered is not cls:
            raise ValueError(
                f'Architecture {key} is already registered to '
                f'{registered.__name__}'
            )

        self._classes.add(cls)
        self._repertoire[key] = cls

    def available(self) -> Any:
        return list(self._repertoire.keys())

    def available_classes(self) -> Any:
        return self._classes

    def get(self, key: Any) -> Any:
        return self._repertoire.get(key)

repertoire = Repertoire()

__all__ = ['Repertoire', 'repertoire']
