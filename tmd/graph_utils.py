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

from typing import TypeVar

import networkx as nx


def convert_to_nx(mol):
    """
    Convert an Chem.Mol into a networkx graph.
    """
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())

    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        g.add_edge(src, dst)

    return g


_Node = TypeVar("_Node")


def enumerate_simple_paths_from(graph: nx.Graph, start_node: _Node, length: int) -> list[list[_Node]]:
    """Return all simple paths of a given length starting from a given node.

    A simple path is a path without repeated nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph

    start_node : node
        Initial node for all paths

    length : int
        Length of returned paths

    Returns
    -------
    list of list of node
        Simple paths
    """

    def go(node, cutoff, visited):
        if cutoff == 1:
            return [[node]]
        return [
            [node, *path]
            for neighbor in nx.neighbors(graph, node)
            if neighbor not in visited
            for path in go(neighbor, cutoff - 1, visited | {node})
        ]

    return go(start_node, length, set())


def enumerate_simple_paths(graph: nx.Graph, length: int) -> list[list]:
    """Return all simple paths of a given length.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph

    length : int
        Length of returned paths

    Returns
    -------
    list of list of node
        Simple paths
    """
    return [path for start_node in graph for path in enumerate_simple_paths_from(graph, start_node, length)]
