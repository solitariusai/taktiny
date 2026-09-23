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
"""Utilities for mapping and loading weights from various formats."""


def map_state_dict(source_state_dict: dict, module_map: list) -> dict:
    """
    Maps a generic state dict (e.g. from PyTorch, Safetensors, Megatron)
    to the Taktiny state dict format using a mapping ruleset.

    Args:
        source_state_dict: The original flat dictionary of weights.
        module_map: A list of tuples defining the mapping rules.
            Rules can be:
            - 1-to-1: ("source_prefix", "target_prefix")
            - 1-to-N: ("source_prefix", ["target1", "target2"], transform_lambda)
            - N-to-1: (["source1", "source2"], "target_prefix", transform_lambda)

    Returns:
        dict: The newly mapped state dict.
    """
    current_state_dict = source_state_dict.copy()

    for rule in module_map:
        is_identity = (len(rule) == 2)

        if len(rule) == 3:
            source_patterns, target_patterns, transform_fn = rule
        else:
            source_patterns, target_patterns = rule
            def transform_fn(x):
                return x

        if isinstance(source_patterns, str):
            source_patterns = [source_patterns]
        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]

        primary_source = source_patterns[0]

        if is_identity and len(source_patterns) == 1 and len(target_patterns) == 1:
            new_dict = {}
            for key, value in current_state_dict.items():
                if primary_source in key:
                    new_dict[key.replace(primary_source, target_patterns[0])] = value
                else:
                    new_dict[key] = value
            current_state_dict = new_dict
        else:
            new_dict = {}
            processed_this_rule = set()

            for key in list(current_state_dict.keys()):
                if key in processed_this_rule:
                    continue

                if primary_source in key:
                    arrays = []
                    valid_match = True
                    for p in source_patterns:
                        sibling_key = key.replace(primary_source, p)
                        if sibling_key in current_state_dict:
                            arrays.append(current_state_dict[sibling_key])
                        else:
                            valid_match = False
                            break

                    if valid_match:
                        transformed = transform_fn(*arrays)
                        if not isinstance(transformed, (list, tuple)):
                            transformed = [transformed]

                        for p in source_patterns:
                            sibling_key = key.replace(primary_source, p)
                            processed_this_rule.add(sibling_key)

                        for target_pattern, new_array in zip(target_patterns, transformed):
                            new_key = key.replace(primary_source, target_pattern)
                            new_dict[new_key] = new_array

            for key, value in current_state_dict.items():
                if key not in processed_this_rule:
                    new_dict[key] = value

            current_state_dict = new_dict

    return current_state_dict


__all__ = ['map_state_dict']