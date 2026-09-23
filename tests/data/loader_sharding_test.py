import jax
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from taktiny.data import DataLoader
from taktiny.utils.spmd import map_logical_axis_names


def test_loader_logical_override_and_restore():
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    rows = [{'img': np.ones((2,), np.float32) * i, 'label': np.int32(i)} for i in range(8)]
    with jax.set_mesh(mesh), map_logical_axis_names({'batch': 'data', 'feature': None}):
        loader = DataLoader(rows, batch_size=4,
                            axis_names={'img': ('batch', 'feature')}, partition_spec=P())
        iterator = iter(loader)
        first = next(iterator)
        assert first['img'].sharding.spec == P('data', None)
        assert first['label'].sharding.spec == P()
        state = iterator.get_state()
        expected = next(iterator)
        restored = iter(loader)
        restored.set_state(state)
        actual = next(restored)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])


def test_loader_field_specs():
    mesh = Mesh(np.asarray(jax.devices()), ('data',))
    with jax.set_mesh(mesh):
        loader = DataLoader([{'x': np.int32(i), 'y': np.int32(i)} for i in range(4)],
                            batch_size=4, partition_spec={'x': P('data')})
        batch = next(iter(loader))
        assert isinstance(batch['x'], jax.Array)
        assert isinstance(batch['y'], np.ndarray)


def test_loader_sharding_errors():
    with pytest.raises(TypeError, match='axis_names'):
        DataLoader([np.ones(2)], axis_names=P('data'))
    with jax.set_mesh(Mesh(np.asarray(jax.devices()), ('data',))):
        with pytest.raises(ValueError, match='ndim'):
            next(iter(DataLoader([np.ones(2)], batch_size=1, axis_names=('batch',))))
