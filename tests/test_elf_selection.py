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
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.fe.utils import generate_conformations, get_romol_conf
from tmd.ff.handlers.elf import prune_conformers_elf


def test_prune_conformers_elf_remove_high_energy_conf():
    """Verify that calling prune_conformers_elf will remove conformers with an internal H-bond

    Modified example from https://github.com/openforcefield/openff-toolkit/blob/2bf586e036ffc96f631b99914a984ad69a69ef8b/openff/toolkit/_tests/test_toolkits.py#L1415
    """
    mol = Chem.MolFromMolBlock(
        """z_3_hydroxy_propenal
  -OEChem-12302017113D

  9  8  0     0  0  0  0  0  0999 V2000
    0.5477    0.3297   -0.0621 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1168   -0.7881    0.2329 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4803   -0.8771    0.1667 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2158    1.5206   -0.4772 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3353    2.5772   -0.7614 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.6274    0.3962   -0.0089 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.3388   -1.7170    0.5467 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7743   -1.7634    0.4166 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3122    1.4082   -0.5180 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  1  4  1  0  0  0  0
  4  5  2  0  0  0  0
  1  6  1  0  0  0  0
  2  7  1  0  0  0  0
  3  8  1  0  0  0  0
  4  9  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    # Coordinates are in angstroms, not nanometers
    initial_conformers = [
        # Add a conformer with an internal H-bond. Expected to be removed
        np.array(
            [
                [0.5477, 0.3297, -0.0621],
                [-0.1168, -0.7881, 0.2329],
                [-1.4803, -0.8771, 0.1667],
                [-0.2158, 1.5206, -0.4772],
                [-1.4382, 1.5111, -0.5580],
                [1.6274, 0.3962, -0.0089],
                [0.3388, -1.7170, 0.5467],
                [-1.8612, -0.0347, -0.1160],
                [0.3747, 2.4222, -0.7115],
            ]
        ),
        # Add a conformer without an internal H-bond.
        np.array(
            [
                [0.5477, 0.3297, -0.0621],
                [-0.1168, -0.7881, 0.2329],
                [-1.4803, -0.8771, 0.1667],
                [-0.2158, 1.5206, -0.4772],
                [0.3353, 2.5772, -0.7614],
                [1.6274, 0.3962, -0.0089],
                [0.3388, -1.7170, 0.5467],
                [-1.7743, -1.7634, 0.4166],
                [-1.3122, 1.4082, -0.5180],
            ]
        ),
    ]
    for conf in initial_conformers:
        new_conf = Chem.Conformer(mol.GetNumAtoms())
        new_conf.SetPositions(conf)
        mol.AddConformer(new_conf)

    pruned_mol = prune_conformers_elf(mol)
    assert len(pruned_mol.GetConformers()) == 1
    np.testing.assert_allclose(get_romol_conf(pruned_mol), initial_conformers[1] / 10.0)


@pytest.mark.parametrize("n_confs", [5, 50])
@pytest.mark.parametrize("seed", [2025])
@pytest.mark.parametrize(
    "smi",
    [
        # Trans OOH
        "[H]OC(=O)C([H])([H])C([H])([H])C([H])([H])[H]",  # mobley_820789 - trans carboxlic acid
        # 3-Hydroxypropionaldehyde - Small example
        "C(CO)C=O",
        # CHEMBL3918408 - Non-macrocycle
        "COc1ccc(-c2nc(-c3ccc(CO)o3)n3ccccc23)cc1",
        # CHEMBL2058675 - Macrocycle
        "CC[C@H](C)[C@@H]1NC(=O)c2csc(n2)[C@H]([C@@H](C)CC)NC(=O)c2csc(n2)CNC(=O)c2nc1oc2C",
    ],
)
def test_prune_conformers_elf(smi, seed, n_confs):
    limit = min(10, n_confs)
    percentage = 50.0  # set higher than the more practical value of 2.0 to ensure RMS is applied
    rms_elf_rms_tolerance = 0.05  # Half an angstrom, in nm
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert len(mol.GetConformers()) == 0
    # Set rms threshold lower than typical, to ensure we get more conformers
    generate_conformations(mol, n_confs=n_confs, seed=seed, rms_threshold=0.5)
    assert len(mol.GetConformers()) >= 1

    pruned_mol = prune_conformers_elf(mol, limit=limit, percentage=percentage, rms_tolerance=rms_elf_rms_tolerance)
    assert len(pruned_mol.GetConformers()) >= 1
    assert len(mol.GetConformers()) >= len(pruned_mol.GetConformers())
    assert len(pruned_mol.GetConformers()) <= limit

    confs = list(pruned_mol.GetConformers())
    if len(confs) > 1:
        assert np.all(np.array(AllChem.GetConformerRMSMatrix(pruned_mol)) >= rms_elf_rms_tolerance)
