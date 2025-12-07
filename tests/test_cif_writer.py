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

from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from openmm.app import PDBxFile

from tmd.fe.cif_writer import CIFWriter, build_openmm_topology, convert_single_topology_mols
from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import get_romol_conf
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]


def test_write_single_topology_frame():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    top = SingleTopology(mol_a, mol_b, core, forcefield)
    host_config = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b])

    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([host_config.omm_topology, mol_a, mol_b], temp.name)

        ligand_coords = top.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

        bad_coords = np.concatenate([host_config.conf, ligand_coords])

        with pytest.raises(ValueError):
            # Should fail, as incorrect number of coords
            bad_coords = bad_coords * 10
            writer.write_frame(bad_coords)

        good_coords = np.concatenate([host_config.conf, convert_single_topology_mols(ligand_coords, top)], axis=0)

        # tbd replace with atom map mixin
        writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == 1
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape


@pytest.mark.parametrize("n_frames", [1, 4, 5])
def test_cif_writer_context(n_frames):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    # test vacuum
    with NamedTemporaryFile(suffix=".cif") as temp:
        good_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        with CIFWriter([mol_a, mol_b], temp.name) as writer:
            for _ in range(n_frames):
                writer.write_frame(good_coords * 10)
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape


@pytest.mark.parametrize("n_frames", [1, 4, 5])
def test_cif_writer(n_frames):
    ff = Forcefield.load_default()

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    # test vacuum
    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([mol_a, mol_b], temp.name)
        good_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        for _ in range(n_frames):
            writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape
        np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords, atol=1e-7)

    solvent_host_config = builders.build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])

    # test solvent
    with NamedTemporaryFile(suffix=".cif") as temp:
        writer = CIFWriter([solvent_host_config.omm_topology, mol_a, mol_b], temp.name)
        good_coords = np.concatenate([solvent_host_config.conf, get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0)
        for _ in range(n_frames):
            writer.write_frame(good_coords * 10)
        writer.close()
        cif = PDBxFile(temp.name)
        assert cif.getNumFrames() == n_frames
        assert cif.getPositions(asNumpy=True).shape == good_coords.shape
        # Tolerance is difference due to shifting water coords such that the mols are at the center
        np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords, atol=1e-5)

    # test complex
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as path_to_pdb:
        complex_host_config = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff)

        with NamedTemporaryFile(suffix=".cif") as temp:
            writer = CIFWriter([complex_host_config.omm_topology, mol_a, mol_b], temp.name)
            good_coords = np.concatenate(
                [complex_host_config.conf, get_romol_conf(mol_a), get_romol_conf(mol_b)], axis=0
            )
            for _ in range(n_frames):
                writer.write_frame(good_coords * 10)
            writer.close()
            cif = PDBxFile(temp.name)
            assert cif.getNumFrames() == n_frames
            assert cif.getPositions(asNumpy=True).shape == good_coords.shape
            np.testing.assert_allclose(cif.getPositions(asNumpy=True), good_coords, atol=1e-7)


def test_build_openmm_topology():
    ff = Forcefield.load_default()

    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    for mol in [mol_a, mol_b]:
        mol_topo = build_openmm_topology([mol])
        assert mol_topo.getNumAtoms() == mol.GetNumAtoms()
        assert mol_topo.getNumBonds() == mol.GetNumBonds()
        assert mol_topo.getNumResidues() == 1
        assert next(mol_topo.residues()).name == "LIG"

    combined_mol_topo = build_openmm_topology([mol_a, mol_b])
    assert combined_mol_topo.getNumAtoms() == mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
    assert combined_mol_topo.getNumBonds() == mol_a.GetNumBonds() + mol_b.GetNumBonds()
    assert combined_mol_topo.getNumResidues() == 2
    assert all(res.name == "LIG" for res in combined_mol_topo.residues())

    solvent_host_config = builders.build_water_system(4.0, ff.water_ff)
    solvent_topo = build_openmm_topology([solvent_host_config.omm_topology])
    assert solvent_topo.getNumAtoms() == solvent_host_config.num_water_atoms
    assert solvent_topo.getNumBonds() == 2 * solvent_host_config.num_water_atoms // 3
    assert solvent_topo.getNumResidues() == solvent_host_config.num_water_atoms // 3
    assert all(res.name == "HOH" for res in solvent_topo.residues())

    solvent_mol_top = build_openmm_topology([solvent_host_config.omm_topology, mol_a, mol_b])
    assert (
        solvent_mol_top.getNumAtoms() == solvent_host_config.num_water_atoms + mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
    )
    assert (
        solvent_mol_top.getNumBonds()
        == (2 * solvent_host_config.num_water_atoms // 3) + mol_a.GetNumBonds() + mol_b.GetNumBonds()
    )
    assert solvent_mol_top.getNumResidues() == (solvent_host_config.num_water_atoms // 3) + 2
