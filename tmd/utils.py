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

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources


def batches(n: int, batch_size: int) -> Iterator[int]:
    assert n >= 0
    assert batch_size > 0
    quot, rem = divmod(n, batch_size)
    for _ in range(quot):
        yield batch_size
    if rem:
        yield rem


def not_ragged(xss: Sequence[Sequence]) -> bool:
    return all(len(xs) == len(xss[0]) for xs in xss)


@contextmanager
def path_to_internal_file(module: str, file_name: str):
    with resources.as_file(resources.files(module).joinpath(file_name)) as path:
        yield path
