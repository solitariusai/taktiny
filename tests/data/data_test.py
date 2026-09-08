import grain.python as grain
import numpy as np
import pytest
from datasets import Dataset

from taktiny.data import (
    ApplyTemplate,
    Batch,
    BatchMap,
    Compose,
    DataLoader,
    Map,
    Pack,
    train_validation_split,
)


def test_map_requires_callable():
    with pytest.raises(TypeError, match='function must be callable'):
        Map(None)  # ty: ignore[invalid-argument-type]


def test_from_datasets_applies_operations_in_order():
    source = Dataset.from_dict({'value': [1, 2, 3, 4]})
    dataloader = DataLoader(
        source,
        num_epochs=1,
        operations=[
            Map(lambda row: {'value': row['value'] * 2}),
            grain.Batch(2, drop_remainder=True),
            Map(lambda batch: {'value': batch['value'] + 1}),
        ],
    )

    batches = list(dataloader)

    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0]['value'], [3, 5])
    np.testing.assert_array_equal(batches[1]['value'], [7, 9])


def test_from_datasets_does_not_implicitly_batch():
    source = Dataset.from_dict({'value': [1, 2]})
    dataloader = DataLoader(source, num_epochs=1)

    assert list(dataloader) == [{'value': 1}, {'value': 2}]


def test_from_datasets_starts_workers_without_parsed_absl_flags():
    source = Dataset.from_dict({'value': [1, 2]})
    dataloader = DataLoader(
        source,
        num_epochs=1,
        worker_count=1,
    )

    assert list(dataloader) == [{'value': 1}, {'value': 2}]


def test_from_datasets_accepts_custom_sampler():
    source = Dataset.from_dict({'value': [1, 2, 3]})
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=0,
            shard_count=1,
            drop_remainder=False,
        ),
        shuffle=False,
        seed=0,
    )

    dataloader = DataLoader(
        source,
        sampler=sampler,
        operations=[Map(lambda row: row['value'])],
    )

    assert list(dataloader) == [1, 2, 3]


def test_map_batches_calls_function_once_per_buffer_and_unbatches_mapping():
    source = Dataset.from_dict({'value': [1, 2, 3, 4, 5]})
    calls = []

    def double(rows):
        calls.append([row['value'] for row in rows])
        return {
            'value': np.asarray([
                row['value'] * 2
                for row in rows
            ]),
        }

    dataloader = DataLoader(
        source,
        num_epochs=1,
        operations=[BatchMap(double, 3)],
    )

    assert list(dataloader) == [
        {'value': 2},
        {'value': 4},
        {'value': 6},
        {'value': 8},
        {'value': 10},
    ]
    assert calls == [[1, 2, 3], [4, 5]]


def test_map_batches_preserves_position_when_iterator_is_restored():
    source = Dataset.from_dict({'value': [1, 2, 3, 4]})
    dataloader = DataLoader(
        source,
        num_epochs=1,
        operations=[
            BatchMap(
                lambda rows: [row['value'] * 2 for row in rows],
                4,
            ),
        ],
    )
    iterator = iter(dataloader)
    assert next(iterator) == 2
    state = iterator.get_state()
    expected = next(iterator)

    restored = iter(dataloader)
    restored.set_state(state)

    assert next(restored) == expected


def test_map_batches_can_drop_partial_buffer():
    source = Dataset.from_dict({'value': [1, 2, 3]})
    dataloader = DataLoader(
        source,
        num_epochs=1,
        operations=[
            BatchMap(
                lambda rows: rows,
                2,
                drop_remainder=True,
            ),
        ],
    )

    assert list(dataloader) == [{'value': 1}, {'value': 2}]


def test_map_batches_rejects_changed_cardinality():
    source = Dataset.from_dict({'value': [1, 2]})
    dataloader = DataLoader(
        source,
        num_epochs=1,
        operations=[BatchMap(lambda rows: rows[:-1], 2)],
    )

    with pytest.raises(ValueError, match='returned 1 rows; expected 2'):
        list(dataloader)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'operations': None},
        {'shuffle': 1},
        {'seed': True},
        {'num_epochs': 0},
        {'shard_index': -1},
        {'shard_index': 1, 'shard_count': 1},
        {'shard_count': 0},
        {'worker_count': -1},
        {'worker_buffer_size': 0},
    ],
)
def test_from_datasets_validates_configuration(kwargs):
    source = Dataset.from_dict({'value': [1]})

    with pytest.raises((TypeError, ValueError)):
        DataLoader(source, **kwargs)


def test_from_datasets_defaults_to_single_epoch():
    import inspect

    sig = inspect.signature(DataLoader)

    assert sig.parameters['num_epochs'].default == 1


class _MockTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        ids = [ord(c) % 20 + 1 for c in text]
        if max_length:
            ids = ids[:max_length]
        return {'input_ids': np.asarray([ids], dtype=np.int32)}


def test_from_iterable_materializes_source():
    loader = DataLoader(
        [{'x': i} for i in range(4)],
        num_epochs=1,
        shuffle=False,
        seed=0,
    )

    assert [record['x'] for record in loader] == [0, 1, 2, 3]


def test_train_validation_split_by_fraction():
    source = [{'x': i} for i in range(10)]

    train, validation = train_validation_split(
        source,
        0.3,
        shuffle=False,
    )

    assert len(train) == 7
    assert len(validation) == 3
    assert train[0] == {'x': 3}
    assert validation[0] == {'x': 0}


def test_train_validation_split_by_count():
    source = [{'x': i} for i in range(10)]

    train, validation = train_validation_split(
        source,
        2,
        shuffle=False,
    )

    assert len(train) == 8
    assert len(validation) == 2


def test_pack_multichannel_audio_and_aligned_features():
    records = [
        {'audio': np.arange(6, dtype=np.float32).reshape(3, 2),
         'features': np.arange(9, dtype=np.int32).reshape(3, 3), 'id': 'first'},
        {'audio': np.full((2, 2), 10, dtype=np.float32),
         'features': np.full((2, 3), 20, dtype=np.int32), 'id': 'second'},
    ]
    operation = Pack(4, keys=['audio', 'features'], position_key='offset', mask_key='valid')
    outputs = list(operation.pack(records))
    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[0]['audio'], [[0, 1], [2, 3], [4, 5], [10, 10]])
    np.testing.assert_array_equal(outputs[1]['features'], [[20, 20, 20], [0, 0, 0],
                                                          [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(outputs[0]['offset'], [0, 1, 2, 0])
    np.testing.assert_array_equal(outputs[1]['offset'], [1, 0, 0, 0])
    np.testing.assert_array_equal(outputs[1]['valid'], [1, 0, 0, 0])
    assert outputs[0]['audio'].dtype == np.float32
    assert outputs[0]['features'].dtype == np.int32
    assert 'id' not in outputs[0]
    assert records[0]['audio'].shape == (3, 2)


def test_field_specific_negative_axes_and_shared_mask():
    record = {
        'audio': np.arange(10).reshape(2, 5),
        'features': np.arange(15).reshape(5, 3),
        'valid': np.array([0, 1, 0, 1, 1]),
    }
    packer = Pack(4, keys=('audio', 'features'), axis={'audio': -1, 'features': 0},
                  padding_values={'audio': -2}, mask_key='valid', position_key='offset')
    result = next(packer.pack([record]))
    np.testing.assert_array_equal(result['audio'], [[1, 3, 4, -2], [6, 8, 9, -2]])
    np.testing.assert_array_equal(result['features'][:3], record['features'][[1, 3, 4]])
    np.testing.assert_array_equal(result['offset'], [0, 1, 2, 0])
    np.testing.assert_array_equal(result['valid'], [1, 1, 1, 0])


@pytest.mark.parametrize('axis', [0, 1, 2, -1, -2, -3])
def test_packing_preserves_nonpacking_dimensions(axis):
    original = np.arange(30).reshape(5, 2, 3)
    value = np.moveaxis(original, 0, axis)
    packer = Pack(3, keys=['x'], axis=axis)
    results = list(packer.pack([{'x': value}]))
    restored = np.concatenate([np.moveaxis(row['x'], axis, 0) for row in results])[:5]
    np.testing.assert_array_equal(restored, original)


def test_no_text_specific_defaults_or_generated_fields():
    result = next(Pack(3, keys=['labels']).pack([{'labels': np.array([5])}]))
    assert set(result) == {'labels'}
    np.testing.assert_array_equal(result['labels'], [5, 0, 0])


@pytest.mark.parametrize('overflow,expected', [
    ('split', [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 0, 0]]),
    ('truncate', [[1, 2, 3, 4]]),
])
def test_overflow_modes(overflow, expected):
    source = [{'x': np.arange(1, 3)}, {'x': np.arange(3, 10)}]
    actual = [row['x'].tolist() for row in Pack(4, keys=['x'], overflow=overflow).pack(source)]
    assert actual == expected


def test_empty_inputs_masks_and_dropping_final_pack():
    packer = Pack(4, keys=['x'], mask_key='valid', drop_remainder=True)
    assert list(packer.pack([])) == []
    assert list(packer.pack([{'x': np.empty((0, 2))}])) == []
    assert list(packer.pack([{'x': np.ones((2, 2)), 'valid': [0, 0]}])) == []
    assert list(packer.pack([{'x': np.ones((2, 2))}])) == []
    assert len(list(packer.pack([{'x': np.ones((4, 2))}]))) == 1


def test_packing_is_lazy_and_iterators_do_not_share_buffers():
    reads = []

    def source():
        for i in range(3):
            reads.append(i)
            yield {'x': np.full((2, 2), i)}

    packer = Pack(2, keys=['x'])
    first = packer.pack(source())
    assert reads == []
    np.testing.assert_array_equal(next(first)['x'], np.zeros((2, 2)))
    assert reads == [0]
    second = packer.pack([{'x': np.full((1, 2), 9)}])
    np.testing.assert_array_equal(next(second)['x'], [[9, 9], [0, 0]])
    np.testing.assert_array_equal(next(first)['x'], np.ones((2, 2)))


def test_pack_runs_before_loader_batching():
    source = [{'wave': np.arange(6).reshape(3, 2)}, {'wave': np.full((3, 2), 8)}]
    result = next(iter(DataLoader(source, operations=[Pack(4, keys=['wave']), Batch(2)])))
    assert result['wave'].shape == (2, 4, 2)
    np.testing.assert_array_equal(result['wave'][1], [[8, 8], [8, 8], [0, 0], [0, 0]])


def test_generic_template_is_callable_and_composable_with_compatible_import():
    template = {'path': '{folder}/{name}.png', 'size': (8, 8)}
    operation = Compose(ApplyTemplate(template, return_key='image_info'), lambda row: row['image_info'])
    assert operation({'folder': 'images', 'name': 'sample'}) == {
        'path': 'images/sample.png', 'size': (8, 8),
    }
    assert template['path'] == '{folder}/{name}.png'


@pytest.mark.parametrize('kwargs', [
    {'length': True}, 
    {'length': 0}, 
    {'keys': []}, 
    {'keys': ['x', 'x']},
    {'keys': [None]}, 
    {'axis': True}, 
    {'axis': .5}, 
    {'axis': {}},
    {'axis': {'x': 0, 'y': 1}}, 
    {'padding_values': {'y': 1}},
    {'padding_values': {'x': [1, 2]}}, 
    {'padding_values': []},
    {'position_key': 'x'}, 
    {'position_key': 'same', 'mask_key': 'same'},
    {'position_key': ''}, 
    {'mask_key': 1}, 
    {'drop_remainder': 1}, 
    {'overflow': 'drop'},
])
def test_invalid_pack_configuration(kwargs):
    config = {'length': 4, 'keys': ['x'], **kwargs}
    with pytest.raises((TypeError, ValueError)):
        Pack(**config)  # ty: ignore


@pytest.mark.parametrize('records,match', [
    ([{'x': np.ones((2, 2))}, {'x': np.ones((2, 3))}], 'non-packing shape'),
    ([{'x': np.ones(2, dtype=np.int32)}, {'x': np.ones(2, dtype=np.float32)}], 'dtype'),
    ([{'x': np.array(3)}], 'out of range'),
    ([{}], 'missing'),
])
def test_invalid_record_schema(records, match):
    with pytest.raises((ValueError, KeyError), match=match):
        list(Pack(4, keys=['x']).pack(records))


def test_invalid_alignment_and_axis():
    with pytest.raises(ValueError, match='equal lengths'):
        list(Pack(4, keys=['x', 'y']).pack([{'x': np.ones((2, 3)), 'y': np.ones(3)}]))
    with pytest.raises(ValueError, match='out of range'):
        list(Pack(4, keys=['x'], axis=2).pack([{'x': np.ones((2, 3))}]))
    with pytest.raises(TypeError, match='mapping'):
        list(Pack(4, keys=['x']).pack([np.ones(3)]))

