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

import numpy as np
from rdkit import Chem

from tmd.fe.utils import get_romol_conf
from tmd.potentials.jax_utils import pairwise_distances


def get_radius_of_mol_pair(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Takes two molecules, computes the max pairwise distance within the molecule coordinates,
    treating that as a diameter and returns the radius
    """
    conf = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    diameter = np.max(pairwise_distances(conf))
    return diameter / 2
