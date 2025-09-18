# Copyright 2019-2025, Relay Therapeutics
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

import os
from collections import defaultdict
from subprocess import check_output


def get_visible_gpus(num_workers: int) -> list[int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        return [int(i) for i in visible_devices.split(",")]
    return list(range(num_workers))


def get_gpu_count() -> int:
    # Expected to return a line delimited summary of each GPU
    try:
        output = check_output(["nvidia-smi", "-L"])
    except FileNotFoundError:
        return 0
    num_gpus = len([x for x in output.split(b"\n") if len(x)])
    gpu_list = get_visible_gpus(num_gpus)

    return len(gpu_list)


def batch_list(values: list, num_workers: int | None = None) -> list[list]:
    """
    Split a list of values into `num_workers` batches.
    If num_workers is None, then split each value into a separate batch.
    """
    batched_values = defaultdict(list)
    num_workers = num_workers or len(values)
    for i, value in enumerate(values):
        batched_values[i % num_workers].append(value)
    return list(batched_values.values())
