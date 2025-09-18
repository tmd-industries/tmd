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

import pytest
from hypothesis import given, seed
from hypothesis.strategies import composite, floats, lists, sampled_from

from tmd.fe.protocol_refinement import greedy_bisection_step

lambdas = floats(0.0, 1.0, allow_subnormal=False)

# https://github.com/python/mypy/issues/12617
lambda_schedules = lists(lambdas, min_size=2, unique=True).map(sorted)  # type: ignore


pytestmark = [pytest.mark.nocuda]


@composite
def greedy_bisection_step_args_instances(draw):
    protocol = draw(lambda_schedules)
    worst_pair_lam1 = draw(sampled_from(protocol[:-1]))

    def local_cost(lam1, _):
        return 1.0 if lam1 == worst_pair_lam1 else 0.0

    def make_intermediate(lam1, lam2):
        assert lam1 < lam2
        return draw(floats(lam1, lam2, allow_subnormal=False))

    return protocol, local_cost, make_intermediate


@given(greedy_bisection_step_args_instances())
@seed(2023)
def test_greedy_bisection_step(args):
    protocol, local_cost, make_intermediate = args
    refined_protocol, _ = greedy_bisection_step(protocol, local_cost, make_intermediate)
    assert len(refined_protocol) == len(protocol) + 1
    assert set(refined_protocol).issuperset(set(protocol))
    assert refined_protocol == sorted(refined_protocol)
