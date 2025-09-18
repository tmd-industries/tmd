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
from rdkit import Chem

from tmd.fe.restraints import setup_relative_restraints_using_smarts

pytestmark = [pytest.mark.nocuda]


def test_setting_up_restraints_using_smarts():
    smi_a = "CCCONNN"
    smi_b = "CCCNNN"
    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))

    smarts = "[#6]-[#6]-[#6]-[#7,#8]-[#7]-[#7]"

    # setup_relative_restraints_using_smarts assumes conformers approximately aligned
    for mol in [mol_a, mol_b]:
        mol.Compute2DCoords()

    core = setup_relative_restraints_using_smarts(mol_a, mol_b, smarts)

    expected_num_atoms = Chem.MolFromSmarts(smarts).GetNumAtoms()

    assert core.shape == (expected_num_atoms, 2)
