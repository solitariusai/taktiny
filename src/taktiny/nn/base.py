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

import typing as tp
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self

import jax
import qwix
from jax._src import config
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from taktiny.utils.format import format_bytes, format_dtype, format_params
from taktiny.utils.spmd import logical_to_mesh_axes
from taktiny.utils.typing import AxisNames, ParameterDict, PyTree, StateDict


def iter_children(obj: object) -> Iterator[tuple[str, Module | Parameter]]:
    """Iterates over the children Modules and Parameters of a given object.

    Args:
        obj: The object whose children to iterate over.

    Yields:
        Tuples of (name, child) for each child Module or Parameter.
    """
    try:
        attrs = vars(obj)
    except TypeError:
        return

    for k, v in attrs.items():
        if isinstance(v, (Module, Parameter)):
            yield k, v

        elif isinstance(
            v, (list, tuple)) and \
            all(isinstance(x, (Module, Parameter)) for x in v
        ):
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
    """Builds a string representation of the module tree.

    Args:
        name (str): The name of the current node.
        obj (object): The module or parameter object to represent.
        prefix (str, optional): Prefix string for the current line. Defaults to "".
        is_last (bool, optional): Whether this is the last child node. Defaults to True.
        is_root (bool, optional): Whether this is the root node. Defaults to False.

    Returns:
        tuple[list[str], int, int]: A tuple containing the list of representation string lines, the total number of parameters, and the total bytes.
    """
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
            node_str = f"{title} ({format_params(total_params)}, {format_bytes(total_bytes)})"
        else:
            node_str = (
                f"{current_prefix}{name}: {title} "
                f"({format_params(total_params)}, "
                f"{format_bytes(total_bytes)})"
            )

        lines.insert(0, node_str)
        lines.extend(child_lines)
        return lines, total_params, total_bytes

    return [], 0, 0

def _is_dynamic(v: object) -> bool:
    """Checks if a given value contains dynamic objects like Modules, Parameters, or Arrays.

    Args:
        v (object): The object to check.

    Returns:
        bool: True if the object contains dynamic properties, False otherwise.
    """
    if isinstance(v, (Module, Parameter, jax.Array, qwix.QArray)):
        return True
    if hasattr(jax.numpy, 'ndarray') and isinstance(v, jax.numpy.ndarray):
        return True
    if isinstance(v, (list, tuple)) and len(v) > 0 and all(_is_dynamic(x) for x in v):
        return True
    return bool(isinstance(v, dict) and len(v) > 0 and all(_is_dynamic(x) for x in v.values()))

class Module:
    """
    Base class for all neural network modules.
    """

    training: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Initializes subclasses and registers them as PyTree nodes.
        """
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    def train(self) -> Self:
        """Sets the module and all its children to training mode.

        Returns:
            Self: The module itself.
        """
        self.training = True
        for _, child in iter_children(self):
            if isinstance(child, Module) and not isinstance(child, Parameter):
                child.train()
        return self

    def eval(self) -> Self:
        """Sets the module and all its children to evaluation mode.

        Returns:
            Self: The module itself.
        """
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
    ) -> tuple[tuple[PyTree, ...], tuple[tuple[str, ...], dict[str, Any]]]:
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
        aux_data: Any,
        children: Sequence[PyTree],
    ) -> Self:
        obj = object.__new__(cls)
        dynamic_names, static_data = aux_data

        obj.__dict__.update(static_data)
        for k, v in zip(dynamic_names, children):
            obj.__dict__[k] = v

        return obj

    def flat_state_dict(self, prefix: str = '') -> StateDict:
        """Returns a flattened dictionary containing the module's state.

        Args:
            prefix (str, optional): A prefix to prepend to all keys. Defaults to ''.

        Returns:
            StateDict: A dictionary mapping flattened parameter names to their values.
        """
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child.value
            elif isinstance(child, Module):
                state.update(child.flat_state_dict(prefix + name + '.'))
        return state

    def flat_parameter_dict(self, prefix: str = '') -> ParameterDict:
        """Returns a flattened dictionary containing the module's parameters.

        Args:
            prefix (str, optional): A prefix to prepend to all keys. Defaults to ''.

        Returns:
            ParameterDict: A dictionary mapping flattened names to Parameter objects.
        """
        state = {}
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                state[prefix + name] = child
            elif isinstance(child, Module):
                state.update(child.flat_parameter_dict(prefix + name + '.'))
        return state

    def state_dict(self) -> StateDict:
        """Returns a hierarchical dictionary containing the module's state.

        Returns:
            StateDict: A nested dictionary representing the module state.
        """
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
        """Loads state values from a flattened dictionary into the module.

        Args:
            state (Mapping[str, PyTree]): Flattened dictionary of state values.
            prefix (str, optional): Prefix used in the flattened keys. Defaults to ''.
        """
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                full_name = prefix + name
                if full_name in state:
                    child._value = state[full_name]
            elif isinstance(child, Module):
                child.load_flat_state_dict(state, prefix + name + '.')

    def load_state_dict(self, state: Mapping[str, PyTree]) -> None:
        """Loads state values from a hierarchical dictionary into the module.

        Args:
            state (Mapping[str, PyTree]): Hierarchical dictionary of state values.
        """
        for name, child in iter_children(self):
            if isinstance(child, Parameter):
                if name in state:
                    child._value = state[name]
            elif isinstance(child, Module) and name in state:
                child.load_state_dict(state[name])

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...

class Parameter(Module):
    """A PyTree node that wraps a JAX array or quantized tensor as a layer parameter.

    Parameters delegate attribute access and arithmetic operations to their underlying
    values, allowing them to be used seamlessly in JAX mathematical operations.

    Args:
        array (PyTree): The underlying tensor data (e.g., a jax.Array or qwix.QArray).
        trainable (bool, optional): A metadata flag indicating if the parameter should be updated. 
            Note that this does not automatically freeze the parameter in raw JAX; it must be explicitly 
            filtered (e.g., by Taktiny's Trainer or JAX tree utilities) before being passed to an optimizer. Defaults to True.
        axis_names (AxisNames | None, optional): Logical axis names for advanced sharding or tensor parallelism. Defaults to None.
        partition_spec (PartitionSpec | None, optional): Explicit hardware sharding specification. If provided along with axis_names, active logical mapping rules will override this value. Defaults to None.
        metadata (dict[str, Any] | Sequence[tuple[str, Any]] | None, optional): Optional metadata dictionary for custom layer logic. Defaults to None.

    Example:
        >>> from taktiny.nn import Parameter
        >>> 
        >>> z = Parameter(jnp.ones((4, 4)))
        >>> output = jnp.dot(x, z)  # Parameter behaves like a normal array
    """

    def __init__(
        self, 
        array: PyTree, 
        *,
        trainable: bool = True,
        axis_names: AxisNames | None = None,
        partition_spec: PartitionSpec | None = None,
        metadata: dict[str, Any] | Sequence[tuple[str, Any]] | None = None 
    ) -> None:
        self.axis_names = axis_names
        self.partition_spec = partition_spec
        self._value = array
        self.trainable = trainable
        self.metadata = dict(metadata) if metadata is not None else None
        if axis_names is not None:
            if isinstance(
                axis_names, 
                (list, tuple)
            ) and len(axis_names) != array.ndim:
                raise ValueError(
                    f'axis_names length {len(axis_names)} must match '
                    f'array ndim {array.ndim}'
                )

            if isinstance(axis_names, PartitionSpec):
                raise TypeError(
                    "Passing a PartitionSpec to 'axis_names' is forbidden. "
                    "Use the 'partition_spec' argument instead."
                )

            # map to partition spec
            mapped_axes = logical_to_mesh_axes(axis_names)
            if mapped_axes is not None:
                self.partition_spec = mapped_axes
                
        # Automatically shard the array if a mesh and partition spec are present
        if self.partition_spec is not None:
            active_mesh = config.device_context.value
            if active_mesh is not None and not active_mesh.empty:
                try:
                    self._value = jax.device_put(self._value, device=NamedSharding(active_mesh, self.partition_spec))
                except Exception:  # noqa: BLE001, S110
                    pass

    @property
    def value(self) -> PyTree:
        """Returns the underlying tensor data."""
        return self._value

    def tree_flatten(
        self,
    ) -> tuple[tuple[PyTree, ...], Any]:
        static_data = {
            name: value
            for name, value in self.__dict__.items()
            if name != '_value'
        }
        return (self._value,), static_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Any,
        children: tp.Sequence[PyTree],
    ) -> Self:
        parameter = object.__new__(cls)
        parameter.__dict__.update(aux_data)
        parameter._value = children[0]
        return parameter

    def __repr__(self) -> str:
        return (
            "Parameter("
            f"shape={getattr(self._value, 'shape', 'None')}, "
            f"dtype={getattr(self._value, 'dtype', 'None')}, "
            f"trainable={self.trainable}"
            ")"
        )

    def __jax_array__(self) -> jax.Array:
        if isinstance(self._value, qwix.QArray):
            return qwix.dequantize(self._value)
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def __add__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value + other

    def __radd__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other + self._value

    def __sub__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value - other

    def __rsub__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other - self._value

    def __mul__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value * other

    def __rmul__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other * self._value

    def __truediv__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value / other

    def __rtruediv__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other / self._value

    def __floordiv__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value // other

    def __rfloordiv__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other // self._value

    def __mod__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value % other

    def __rmod__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other % self._value

    def __pow__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value**other

    def __rpow__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other**self._value

    def __matmul__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value @ other

    def __rmatmul__(self, other: ArrayLike | Parameter) -> jax.Array:
        return other @ self._value

    def __eq__(self, other: object) -> Any:
        return self._value == other

    def __ne__(self, other: object) -> Any:
        return self._value != other

    def __lt__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value < other

    def __le__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value <= other

    def __gt__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value > other

    def __ge__(self, other: ArrayLike | Parameter) -> jax.Array:
        return self._value >= other

    def __neg__(self) -> jax.Array:
        return -self._value

    def __pos__(self) -> jax.Array:
        return +self._value

    def __abs__(self) -> jax.Array:
        return abs(self._value)

    def __invert__(self) -> jax.Array:
        return ~self._value

    def __getitem__(self, key: Any) -> Any:
        sliced_value = self._value[key]
        
        def slice_axes(axes):
            if axes is None:
                return None
            _key = key if isinstance(key, tuple) else (key,)
            old_idx = 0
            names = []
            valid = True
            for k in _key:
                if k is Ellipsis:
                    valid = False; break
                elif k is None:
                    names.append(None)
                elif isinstance(k, (int, slice)):
                    if isinstance(k, slice) and old_idx < len(axes):
                        names.append(axes[old_idx])
                    old_idx += 1
                elif hasattr(k, 'ndim') and k.ndim == 0:
                    old_idx += 1
                else:
                    valid = False; break
            if valid:
                while old_idx < len(axes):
                    names.append(axes[old_idx])
                    old_idx += 1
                if len(names) == sliced_value.ndim:
                    return tuple(names)
            return None
            
        sliced_axis_names = slice_axes(self.axis_names)
        sliced_partition_spec = slice_axes(self.partition_spec)
        
        if sliced_partition_spec is not None:
            sliced_partition_spec = PartitionSpec(*sliced_partition_spec)
            
        p = Parameter(
            sliced_value, 
            axis_names=sliced_axis_names,
            partition_spec=sliced_partition_spec,
            trainable=self.trainable, 
            metadata=self.metadata
        )
        return p


def module(cls):
    """Class decorator to transform a generic class into a Module subclass.

    Returns:
        _type_: The newly created Module subclass.
    """
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
