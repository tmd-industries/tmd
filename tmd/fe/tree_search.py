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

import heapq
from collections.abc import Iterator, Sequence
from typing import Callable, TypeVar

Node = TypeVar("Node")
State = TypeVar("State")


def best_first(
    expand: Callable[[Node, State], tuple[Sequence[Node], State]],
    root: Node,
    initial_state: State,
) -> Iterator[Node]:
    """Generic search algorithm returning an iterator over nodes.

    The best-first strategy proceeds by maintaining a priority queue of active search nodes, and at each iteration
    yielding the best (minimal) node and adding its children to the queue.

    Parameters
    ----------
    expand : Callable
       Function from node and initial state to children and updated state. If the search is stateless, this function may
       ignore its second argument and return an arbitrary second element (e.g. None).

    root : Node
       Starting node

    initial_state : State
       Initial value of the global search state. If the search is stateless, can be an arbitrary value (e.g. None)
    """
    state = initial_state
    queue = [root]
    while queue:
        node = heapq.heappop(queue)
        children, state = expand(node, state)
        yield node
        for child in children:
            heapq.heappush(queue, child)
