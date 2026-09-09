import doctest
from dataclasses import dataclass

import numpy as np
import pytest

from taktiny import data
from taktiny.data import loader, transforms


def test_array_records_and_composition():
    source = np.arange(24).reshape(4, 2, 3)
    preprocess = data.Compose(lambda x: x.astype(np.float32), data.Map(lambda x: x / 2))
    batches = list(data.DataLoader(source, operations=[preprocess], batch_size=3))
    np.testing.assert_array_equal(batches[0], source[:3] / 2)
    np.testing.assert_array_equal(batches[1], source[3:] / 2)
    assert batches[0].dtype == np.float32
    assert data.Compose()(source) is source


def test_fields_preserve_multimodal_records_without_mutation():
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    audio = np.arange(8, dtype=np.float32)
    source = {'image': image, 'audio': audio, 'label': 2, 'metadata': {'id': 'a'}}
    functions = {'image': lambda x: x.astype(np.float32) / 255}
    operation = data.MapFields(functions)
    functions.clear()
    result = operation(source)
    np.testing.assert_array_equal(result['image'], np.ones_like(image))
    np.testing.assert_array_equal(source['image'], image)
    assert result is not source
    assert result['audio'] is audio
    assert result['metadata'] is source['metadata']


def test_ragged_audio_custom_collation():
    def pad(rows):
        lengths = np.array([len(row) for row in rows])
        padded = np.zeros((len(rows), lengths.max()))
        for i, row in enumerate(rows):
            padded[i, :len(row)] = row
        return {'samples': padded, 'lengths': lengths}

    source = [np.ones(2), np.full(4, 2), np.full(1, 3)]
    batches = list(data.DataLoader(source, batch_size=2, collate_fn=pad))
    np.testing.assert_array_equal(batches[0]['samples'], [[1, 1, 0, 0], [2, 2, 2, 2]])
    np.testing.assert_array_equal(batches[0]['lengths'], [2, 4])
    assert batches[1]['samples'].shape == (1, 1)


@dataclass
class Example:
    value: int


def test_custom_objects_and_operation_order():
    source = [Example(i) for i in range(5)]
    batches = list(data.DataLoader(source, operations=[
        data.Filter(lambda row: row.value % 2 == 0),
        data.Batch(2, collate_fn=list),
        data.Map(lambda rows: sum(row.value for row in rows)),
    ]))
    assert batches == [2, 4]
    assert list(data.DataLoader(source, batch_size=2, collate_fn=list,
                                drop_remainder=True)) == [source[:2], source[2:4]]


class Source:
    def __init__(self):
        self.reads = []

    def __len__(self):
        return 5

    def __getitem__(self, index):
        assert isinstance(index, int)
        self.reads.append(index)
        return index * 2


def test_custom_source_is_lazy_and_split_views_use_python_indices():
    source = Source()
    pipeline = data.DataLoader(source)
    assert source.reads == []
    assert list(pipeline) == [0, 2, 4, 6, 8]
    train, validation = data.train_validation_split(source, 2, shuffle=False)
    assert list(data.DataLoader(train[:2])) == [4, 6]
    assert validation[-1] == 2


def test_random_maps_repeat_and_restore_rng_state():
    operation = data.RandomMap(lambda value, rng: (value, int(rng.integers(100000))))
    pipeline = data.DataLoader(list(range(10)), operations=[operation], seed=7, shuffle=True)
    expected = list(pipeline)
    assert list(pipeline) == expected
    assert list(data.DataLoader(list(range(10)), operations=[operation], seed=8,
                               shuffle=True)) != expected
    iterator = iter(pipeline)
    next(iterator)
    state = iterator.get_state()
    restored = iter(pipeline)
    restored.set_state(state)
    assert list(restored) == expected[1:]


def test_random_map_matches_across_workers():
    operation = data.RandomMap(lambda value, rng: (value, int(rng.integers(100000))))
    source = list(range(7))
    expected = list(data.DataLoader(source, operations=[operation], seed=42))
    assert list(data.DataLoader(source, operations=[operation], seed=42,
                               worker_count=2)) == expected


def test_repeated_and_unbounded_epochs_are_explicit():
    from itertools import islice

    assert list(data.DataLoader([1, 2], num_epochs=2)) == [1, 2, 1, 2]
    assert list(islice(data.DataLoader([1, 2], num_epochs=None), 5)) == [1, 2, 1, 2, 1]


@pytest.mark.parametrize('worker_count', [0, 1])
def test_index_map_supports_loader_and_restore(worker_count):
    pipeline = data.DataLoader([10, 20, 30], operations=[
        data.IndexMap(lambda i, value: {'index': i, 'value': value}),
    ], worker_count=worker_count)
    iterator = iter(pipeline)
    assert next(iterator) == {'index': 0, 'value': 10}
    state = iterator.get_state()
    restored = iter(pipeline)
    restored.set_state(state)
    assert list(restored) == [{'index': 1, 'value': 20}, {'index': 2, 'value': 30}]


def test_flat_map_changes_cardinality_and_restores_inside_expansion():
    pipeline = data.DataLoader([0, 3, 1], operations=[
        data.FlatMap(lambda n: range(n), max_fan_out=3),
    ])
    assert list(pipeline) == [0, 1, 2, 0]
    iterator = iter(pipeline)
    assert next(iterator) == 0
    state = iterator.get_state()
    restored = iter(pipeline)
    restored.set_state(state)
    assert list(restored) == [1, 2, 0]


def test_flat_map_enforces_bound_without_consuming_unbounded_output():
    from itertools import count

    operation = data.FlatMap(lambda _: count(), max_fan_out=2)
    with pytest.raises(ValueError, match='max_fan_out'):
        operation.flat_map(None)


@pytest.mark.parametrize('count', [0, 1, 2, 11, 12])
def test_data_shards_are_disjoint_and_complete(count):
    source = list(range(count))
    shards = [list(data.DataLoader(source, shuffle=True, seed=7,
                                  shard_index=i, shard_count=3)) for i in range(3)]
    assert sorted(value for shard in shards for value in shard) == source
    assert not set(shards[0]) & set(shards[1])


def test_sharded_rngs_match_global_order_and_resume():
    operation = data.RandomMap(lambda x, rng: (x, int(rng.integers(100000))))
    source = list(range(11))
    expected = list(data.DataLoader(source, operations=[operation], seed=7,
                                    shuffle=True, num_epochs=2))
    for i in range(3):
        pipeline = data.DataLoader(source, operations=[operation], seed=7,
                                   shuffle=True, num_epochs=2,
                                   shard_index=i, shard_count=3)
        assert list(pipeline) == expected[i::3]
        iterator = iter(pipeline)
        next(iterator)
        state = iterator.get_state()
        restored = iter(pipeline)
        restored.set_state(state)
        assert list(restored) == expected[i::3][1:]


@pytest.mark.parametrize('source', ['org/repo', b'path', {'column': [1]}, iter([1, 2])])
def test_source_does_not_trigger_implicit_loading_or_materialization(source):
    with pytest.raises(TypeError, match='source must'):
        data.DataLoader(source)


@pytest.mark.parametrize('kwargs', [
    {'batch_size': True}, {'batch_size': 0}, {'batch_size': 1.5},
    {'collate_fn': list}, {'drop_remainder': True}, {'drop_remainder': 1},
    {'batch_size': 2, 'collate_fn': False},
])
def test_batch_configuration_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        data.DataLoader([1, 2], **kwargs)


@pytest.mark.parametrize('factory', [
    lambda: data.RandomMap(None), lambda: data.IndexMap(None),
    lambda: data.Filter(None), lambda: data.Compose(None),
    lambda: data.Compose(data.Filter(lambda x: True)),
    lambda: data.MapFields([]), lambda: data.MapFields({'x': None}),
    lambda: data.FlatMap(lambda x: [], max_fan_out=False),
])
def test_transform_validation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_fields_report_missing_keys_and_nonmapping_records():
    operation = data.MapFields({'image': lambda x: x})
    with pytest.raises(KeyError, match='image'):
        operation({})
    with pytest.raises(TypeError, match='mapping'):
        operation([1, 2])


@pytest.mark.parametrize('size', [.01, .99])
def test_split_rejects_rounding_to_empty_partition(size):
    with pytest.raises(ValueError, match='nonempty'):
        data.train_validation_split([1, 2], size)

