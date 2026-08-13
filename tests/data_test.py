import grain.python as grain
import numpy as np
import pytest
from datasets import Dataset

from taktiny.data._prelude import BatchMap, DatasetUtils, Map


def test_map_requires_callable():
    with pytest.raises(TypeError, match='function must be callable'):
        Map(None)


def test_from_datasets_applies_operations_in_order():
    source = Dataset.from_dict({'value': [1, 2, 3, 4]})
    dataloader = DatasetUtils.from_datasets(
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
    dataloader = DatasetUtils.from_datasets(source, num_epochs=1)

    assert list(dataloader) == [{'value': 1}, {'value': 2}]


def test_from_datasets_starts_workers_without_parsed_absl_flags():
    source = Dataset.from_dict({'value': [1, 2]})
    dataloader = DatasetUtils.from_datasets(
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

    dataloader = DatasetUtils.from_datasets(
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

    dataloader = DatasetUtils.from_datasets(
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
    dataloader = DatasetUtils.from_datasets(
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
    dataloader = DatasetUtils.from_datasets(
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
    dataloader = DatasetUtils.from_datasets(
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
        DatasetUtils.from_datasets(source, **kwargs)
