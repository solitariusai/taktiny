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

import pytest

from taktiny.utils.format import parse_size


def test_parse_size_accepts_byte_counts():
    assert parse_size(0) == 0
    assert parse_size(512) == 512


def test_parse_size_accepts_unit_suffixes():
    assert parse_size('64KB') == 64 * 1024
    assert parse_size('128mb') == 128 * 1024 ** 2
    assert parse_size('2 GB') == 2 * 1024 ** 3
    assert parse_size('1TB') == 1024 ** 4


def test_parse_size_accepts_decimal_magnitudes():
    assert parse_size('1.5GB') == int(1.5 * 1024 ** 3)


@pytest.mark.parametrize(
    'value',
    [-1, 'banana', '-4MB', '10GiB', None, 1.5],
)
def test_parse_size_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_size(value)
