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
"""Base module class"""
from __future__ import annotations
import jax
import operator
import qwix
from collections.abc import Iterator, Mapping, Sequence
from jax.tree_util import register_pytree_node_class
import typing as tp

from taktiny.utils.typing import ParameterDict, PyTree, StateDict
from taktiny.utils.format import format_params, format_bytes, format_dtype

def iter_children(obj: object) -> Iterator[tuple[str, Module | Parameter]]:
    if not hasattr(obj, '__dict__'): return
    for k, v in obj.__dict__.items():
        if isinstance(v, (Module, Parameter)):
            yield k, v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (Module, Parameter)) for x in v):
            for i, x in enumerate(v):
                name = str(i) if k == 'layers' else f"{k}.{i}"
                yield name, x
        elif isinstance(v, Mapping) and all(
            isinstance(x, (Module, Parameter)) for x in v.values()
        ):
            for key, x in v.items():
                name = str(key) if k == 'layers' else f'{k}.{key}'
                yield name, x

def build_tree_repr(
    name: str,
    obj: object,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = False,
) -> tuple[list[str], int, int]:
    lines = []
    total_params = 0
    total_bytes = 0

    current_prefix = "" if is_root else prefix + ("└── " if is_last else "├── ")
    child_prefix = "" if is_root else prefix + ("    " if is_last else "│   ")

    if isinstance(obj, Parameter):
        if isinstance(obj.value, qwix.QArray):
            p = obj.value.qvalue.size
            b = sum(
                getattr(leaf, 'nbytes', 0)
                for leaf in jax.tree_util.tree_leaves(obj.value)
            )
            dt = f'qwix[{obj.value.qtype}]'
        else:
            p = obj.value.size
            b = getattr(obj.value, 'nbytes', 0)
            dt = format_dtype(obj.value.dtype)
        sh = ", ".join(map(str, obj.value.shape))
        lines.append(f"{current_prefix}{name}: {dt}[{sh}]")
        return lines, p, b

    elif isinstance(obj, Module):
        children_items = list(iter_children(obj))

        child_lines = []
        for i, (c_name, c_obj) in enumerate(children_items):
            c_is_last = (i == len(children_items) - 1)
            c_lines, c_p, c_b = build_tree_repr(c_name, c_obj, child_prefix, c_is_last)
            child_lines.extend(c_lines)
            total_params += c_p
            total_bytes += c_b

        extra = obj.extra_repr()
        title = f"{obj.__class__.__name__}({extra})" if extra else obj.__class__.__name__

        if is_root:
            node_str = f"{title} ({format_params(total_params)} parameters, {format_bytes(total_bytes)})"
        else:
            node_str = (
                f"{current_prefix}{name}: {title} "
                f"({format_params(total_params)} parameters, "
                f"{format_bytes(total_bytes)})"
            )

        lines.insert(0, node_str)
        lines.extend(child_lines)
        return lines, total_params, total_bytes

    return [], 0, 0

def _is_dynamic(v: object) -> bool:
    if isinstance(v, (Module, Parameter, jax.Array)):
        return True
    if hasattr(jax.numpy, 'ndarray') and isinstance(v, jax.numpy.ndarray):
        return True
    if isinstance(v, (list, tuple)) and len(v) > 0 and all(_is_dynamic(x) for x in v):
        return True
    if isinstance(v, dict) and len(v) > 0 and all(_is_dynamic(x) for x in v.values()):
        return True
    return False

class Module:
    training: bool = True

    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    def train(self) -> tp.Self:
        """Set this module and every child module to training mode."""
        self.training = True
        for _, child in iter_children(self):
            if isinstance(child, Module) and not isinstance(child, Parameter):
                child.train()
        return self

    def eval(self) -> tp.Self:
        """Set this module and every child module to evaluation mode."""
        self.training = False
        for _, child in iter_children(self):
            if isinstance(child, Module) and not isinstance(child, Parameter):
                child.eval()
        return self

    def extra_repr(self) -> str: return ""
    def __repr__(self) -> str:
        lines, _, _ = build_tree_repr("", self, is_root=True)
        return "\n".join(lines)

    def tree_flatten(
        self,
    ) -> tuple[tuple[PyTree, ...], tuple[tuple[str, ...], dict[str, tp.Any]]]:
        dynamic_names = []
        dynamic_vals = []
        static_data = {}

        for k, v in self.__dict__.items():
            if _is_dynamic(v):
                dynamic_names.append(k)
                dynamic_vals.append(v)
            else:
                static_data[k] = v

        return tuple(dynamic_vals), (tuple(dynamic_names), static_data)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[Sequence[str], Mapping[str, tp.Any]],
        children: Sequence[PyTree],
    ) -> tp.Self:
        obj = object.__new__(cls)
        dynamic_names, static_data = aux_data

        obj.__dict__.update(static_data)
        for k, v in zip(dynamic_names, children):
            obj.__dict__[k] = v

        return obj

    def flat_state_dict(self, prefix: str = '') -> StateDict:
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child.value
            elif isinstance(child, Module):
                state.update(child.flat_state_dict(prefix + name + '.'))
        return state

    def flat_parameter_dict(self, prefix: str = '') -> ParameterDict:
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child
            elif isinstance(child, Module):
                state.update(child.flat_parameter_dict(prefix + name + '.'))
        return state

    def state_dict(self) -> StateDict:
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[name] = child.value
            elif isinstance(child, Module):
                state[name] = child.state_dict()
        return state

    def load_flat_state_dict(
        self,
        state: Mapping[str, PyTree],
        prefix: str = '',
    ) -> None:
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                full_name = prefix + name
                if full_name in state:
                    child.value = state[full_name]
            elif isinstance(child, Module):
                child.load_flat_state_dict(state, prefix + name + '.')

    def load_state_dict(self, state: Mapping[str, PyTree]) -> None:
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                if name in state:
                    child.value = state[name]
            elif isinstance(child, Module):
                if name in state:
                    child.load_state_dict(state[name])

class Parameter(Module):
    def __init__(self, array: PyTree, *, trainable: bool = True) -> None:
        self.value = array
        self.trainable = trainable

    def __repr__(self) -> str:
        return (
            "Parameter("
            f"shape={getattr(self.value, 'shape', 'None')}, "
            f"dtype={getattr(self.value, 'dtype', 'None')}, "
            f"trainable={self.trainable}"
            ")"
        )

    def __jax_array__(self) -> jax.Array:
        if isinstance(self.value, qwix.QArray):
            return qwix.dequantize(self.value)
        return self.value

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self.value, name)

    def tree_flatten(self) -> tuple[tuple[PyTree], dict[str, tp.Any]]:
        aux = {k: v for k, v in self.__dict__.items() if k != 'value'}
        return (self.value,), aux

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Mapping[str, tp.Any],
        children: Sequence[PyTree],
    ) -> tp.Self:
        obj = object.__new__(cls)
        obj.value = children[0]
        if aux_data:
            for k, v in aux_data.items():
                setattr(obj, k, v)
        return obj

def _make_magic_methods() -> None:
    for op in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'matmul',
               'eq', 'ne', 'lt', 'le', 'gt', 'ge']:
        magic = f'__{op}__'
        rmagic = f'__r{op}__'
        setattr(Parameter, magic, lambda self, other, o=op: getattr(operator, o)(self.value, other))
        setattr(Parameter, rmagic, lambda self, other, o=op: getattr(operator, o)(other, self.value))
    for op in ['neg', 'pos', 'abs', 'invert']:
        magic = f'__{op}__'
        setattr(Parameter, magic, lambda self, o=op: getattr(operator, o)(self.value))

    setattr(Parameter, '__getitem__', lambda self, key: operator.getitem(self.value, key))

_make_magic_methods()

def module(cls):
    if issubclass(cls, Module):
        return cls

    return type(
        cls.__name__,
        (cls, Module),
        {
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
        },
    )

__all__ = ['Module', 'Parameter', 'module']
