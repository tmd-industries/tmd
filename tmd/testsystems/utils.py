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

from typing import Optional

from rdkit import Chem

from tmd.fe.utils import get_mol_name, read_sdf
from tmd.utils import path_to_internal_file


def fetch_freesolv(n_mols: Optional[int] = None, exclude_mols: Optional[set[str]] = None) -> list[Chem.Mol]:
    """
    Return the (potentially truncated) FreeSolv data set.

    Parameters
    ----------
    n_mols:
        Limit to this number of mols.
        Default of None means to keep all of the molecules.

    exclude_mols:
        Exclude molecules in the given set.

    """
    with path_to_internal_file("tmd.testsystems.freesolv", "freesolv.sdf") as freesolv_path:
        mols = read_sdf(str(freesolv_path))

    # filter and truncate
    exclude_mols = exclude_mols or set()
    filtered_mols = [mol for mol in mols if get_mol_name(mol) not in exclude_mols]
    first_n_filtered_mols = filtered_mols[:n_mols]

    return first_n_filtered_mols
