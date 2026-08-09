# Copyright 2026 Shinapri.
# coding=utf-8
# Copyright 2025 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Format utilities."""
from __future__ import annotations
import typing as tp


def _format_scaled(value: int | float, scale: int, suffix: str) -> str:
    number = f"{value / scale:.2f}".rstrip('0').rstrip('.')
    return f"{number}{suffix}"

def format_params(size: int) -> str:
    for scale, suffix in (
        (1_000_000_000_000, 'T'),
        (1_000_000_000, 'B'),
        (1_000_000, 'M'),
        (1_000, 'K'),
    ):
        if size >= scale:
            return _format_scaled(size, scale, suffix)
    return f"{size:,}"

def format_bytes(size: int) -> str:
    for scale, suffix in (
        (1024**4, 'TB'),
        (1024**3, 'GB'),
        (1024**2, 'MB'),
        (1024, 'KB'),
    ):
        if size >= scale:
            number = f"{size / scale:.2f}".rstrip('0').rstrip('.')
            return f"{number} {suffix}"
    return f"{int(size)} B"

def format_dtype(dtype: tp.Any) -> str:
    name = dtype.name
    if name == 'float32': return 'f32'
    if name == 'float16': return 'f16'
    if name == 'bfloat16': return 'bf16'
    if name == 'int32': return 'i32'
    if name == 'int64': return 'i64'
    return name

__all__ = [
    'format_params',
    'format_bytes',
    'format_dtype'
]
