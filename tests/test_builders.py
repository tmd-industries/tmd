# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025-2026, Forrest York
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

from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax
import numpy as np
import pytest
from common import ligand_from_smiles
from openmm import app, unit
from rdkit import Chem

from tmd.constants import DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, ONE_4PI_EPS0, NBParamIdx
from tmd.fe.free_energy import HostConfig
from tmd.fe.utils import get_romol_conf, read_sdf, read_sdf_mols_by_name, set_romol_conf
from tmd.ff import get_water_ff_model
from tmd.md.barostat.utils import compute_box_volume, get_bond_list, get_group_indices
from tmd.md.builders import (
    WATER_RESIDUE_NAME,
    build_host_config_from_omm,
    build_membrane_system,
    build_protein_system,
    build_water_system,
    construct_default_omm_system,
    get_box_from_coords,
    load_pdb_system,
    make_waters_contiguous,
    strip_units,
)
from tmd.md.minimizer import check_force_norm
from tmd.potentials import Nonbonded
from tmd.potentials.jax_utils import idxs_within_cutoff
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file


def test_build_water_system():
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    host_config = build_water_system(4.0, DEFAULT_WATER_FF, box_margin=0.1)
    host_with_mols_config = build_water_system(4.0, DEFAULT_WATER_FF, mols=[mol_a, mol_b], box_margin=0.1)

    # No waters should be deleted, but the box will be slightly larger
    assert len(host_config.conf) == len(host_with_mols_config.conf)
    assert compute_box_volume(host_config.box) < compute_box_volume(host_with_mols_config.box)

    mol_coords = np.concatenate([get_romol_conf(mol_a), get_romol_conf(mol_b)])
    mol_centroid = np.mean(mol_coords, axis=0)

    water_centeroid = np.mean(host_config.conf, axis=0)

    # The centroid of the water particles should be near the centroid of the ligand
    mol_water_centeroid = np.mean(host_with_mols_config.conf, axis=0)
    # Should be within one angstrom of the centroid.
    np.testing.assert_allclose(mol_centroid, mol_water_centeroid, atol=0.1)

    # Dependent on the molecule (where it was posed in complex), but centroid of water will not be near the ligands
    assert not np.allclose(mol_centroid, water_centeroid, atol=0.1)

    for bp in host_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(host_config.conf, host_config.box, compute_u=False)
        check_force_norm(-du_dx)

    for bp in host_with_mols_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(
            host_with_mols_config.conf, host_with_mols_config.box, compute_u=False
        )
        check_force_norm(-du_dx)


@pytest.mark.nocuda
@pytest.mark.parametrize("water_ff", ["amber14/tip4pfb", "tip5p", "swm4ndp"])
def test_build_water_system_raises_on_water_ff_with_virtual_sites(water_ff):
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    with pytest.raises(ValueError, match="TMD does not support water models that use virtual sites"):
        build_water_system(4.0, water_ff, mols=[mol_a, mol_b], box_margin=0.1)


@pytest.mark.nocuda
@pytest.mark.parametrize("water_ff", ["tip3p", "spce", "opc3", "amber14/tip3p", "amber14/spce", "amber14/opc3"])
def test_build_water_system_different_water_ffs(water_ff):
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    host_config = build_water_system(4.0, water_ff, mols=[mol_a, mol_b], box_margin=0.1)
    for bp in host_config.host_system.get_U_fns():
        # Skip the nonbonded potential, as a lot of memory is required when using the CPU JAX platform
        if isinstance(bp.potential, Nonbonded):
            continue
        assert np.isfinite(bp(host_config.conf, host_config.box))


@pytest.mark.nocuda
def test_build_protein_system_returns_correct_water_count():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.pfkfb3", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)
    # Pick two arbitrary mols
    mol_a = mols[0]
    mol_b = mols[1]
    last_num_waters = None
    # Verify that even adding different molecules produces the same number of waters in the system
    for mols in (None, [], [mol_a], [mol_b], [mol_a, mol_b]):
        with path_to_internal_file("tmd.testsystems.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as pdb_path:
            host_config = build_protein_system(str(pdb_path), DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=mols)
            # The builder should not modify the number of atoms in the protein at all
            # Hard coded to the number of protein atoms in the PDB, refer to 6hvi_prepared.pdb for the actual
            # number of atoms
            assert host_config.conf.shape[0] - host_config.num_water_atoms == 6748
            if last_num_waters is not None:
                assert last_num_waters == host_config.num_water_atoms
            last_num_waters = host_config.num_water_atoms


@pytest.mark.nocuda
def test_build_protein_system_with_membrane():
    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)
    # Add all the mols in a single pass, as this runs OpenMM CPU MD which is slow
    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "a2a_hip278.pdb") as pdb_path:
        host_config = build_membrane_system(str(pdb_path), "amber14/protein.ff14SB", "amber14/tip3p", mols=mols)
    assert host_config.num_membrane_atoms == 30016


def validate_host_config_ions_and_charge(
    host_config: HostConfig,
    mol: Chem.Mol | None,
    ionic_concentration: float,
    expected_host_charge: int,
    neutralized: bool,
    input_host_charge: int = 0,
):
    mol_formal_charge = 0

    if mol is not None:
        mol_formal_charge = Chem.GetFormalCharge(mol)

    assert isinstance(host_config.host_system.nonbonded_all_pairs.params, (np.ndarray, jax.Array))
    test_charges = np.sum(host_config.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]) / np.sqrt(
        ONE_4PI_EPS0
    )
    np.testing.assert_array_almost_equal(
        np.float32(test_charges),
        np.float32(expected_host_charge),
    )

    bond_indices = get_bond_list(host_config.host_system.bond.potential)

    all_group_idxs = get_group_indices(bond_indices, host_config.conf.shape[0])
    ions = [group for group in all_group_idxs if len(group) == 1]
    num_ions = len(ions)
    if ionic_concentration > 0.0:
        assert num_ions > 0
        if neutralized:
            assert num_ions % 2 == abs(mol_formal_charge + input_host_charge) % 2
        else:
            assert num_ions % 2 == 0
    elif neutralized:
        # Should have the number of ions extra to account for the charge of the ligand
        assert num_ions == abs(mol_formal_charge + input_host_charge)
    else:
        assert num_ions == 0


@pytest.mark.nocuda
@pytest.mark.parametrize("ionic_concentration", [0.0, 0.15])
@pytest.mark.parametrize("neutralize", [False, True])
def test_water_system_ion_concentration_and_neutralization(ionic_concentration, neutralize):
    positive_mol = ligand_from_smiles("c1cc[nH+]cc1")
    negative_mol = ligand_from_smiles("[N+](=O)([O-])[O-]")
    neutral_mol = ligand_from_smiles("c1ccccc1")

    box_size = 2.0

    host_config_no_ions = build_water_system(box_size, DEFAULT_WATER_FF, ionic_concentration=0.0)
    # Host system will have zero net charge if no ionic concentration and not neutralized
    assert np.sum(host_config_no_ions.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]) == 0.0

    # Can't mix ligands of different charges when neutralizing the system
    if neutralize:
        with pytest.raises(AssertionError):
            build_water_system(
                box_size,
                DEFAULT_WATER_FF,
                mols=[positive_mol, negative_mol],
                ionic_concentration=ionic_concentration,
                neutralize=neutralize,
            )
    else:
        build_water_system(
            box_size,
            DEFAULT_WATER_FF,
            mols=[positive_mol, negative_mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
    host_config = build_water_system(
        box_size, DEFAULT_WATER_FF, mols=[], ionic_concentration=ionic_concentration, neutralize=neutralize
    )
    validate_host_config_ions_and_charge(host_config, None, ionic_concentration, 0, neutralize)
    for mol in [positive_mol, negative_mol, neutral_mol]:
        host_config = build_water_system(
            box_size, DEFAULT_WATER_FF, mols=[mol], ionic_concentration=ionic_concentration, neutralize=neutralize
        )
        expected_charge = 0
        if neutralize:
            # Since the ligand isn't in the system, should be missing the charge of the ligand
            expected_charge = -Chem.GetFormalCharge(mol)
        validate_host_config_ions_and_charge(host_config, mol, ionic_concentration, expected_charge, neutralize)


@pytest.mark.nocuda
def test_build_systems_large_batch_of_ligands():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)
    batched_mols = mols * 25
    assert len(batched_mols) > 1000
    host_config = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, box_margin=0.1, mols=batched_mols
    )

    def verify_no_nearby_waters(host_config: HostConfig, mols: list[Chem.Mol]):
        water_coords = host_config.conf[-host_config.num_water_atoms :]
        box = host_config.box
        for i, mol in enumerate(mols):
            clashy_idxs = idxs_within_cutoff(water_coords, get_romol_conf(mol), box, cutoff=0.2)
            assert len(clashy_idxs) == 0, f"Mol {i} has water within 2 angstroms"

    verify_no_nearby_waters(host_config, batched_mols)

    host_config = build_water_system(4.0, DEFAULT_WATER_FF, box_margin=0.1, mols=batched_mols)
    verify_no_nearby_waters(host_config, batched_mols)


@pytest.mark.nocuda
@pytest.mark.parametrize("ionic_concentration", [0.0, 0.15])
@pytest.mark.parametrize("neutralize", [False, True])
def test_protein_system_ion_concentration_and_neutralization(ionic_concentration, neutralize):
    # Note that none of this ligands go with the protein, but as long as we don't minimize, all is well.
    positive_mol = ligand_from_smiles("c1cc[nH+]cc1")
    negative_mol = ligand_from_smiles("[N+](=O)([O-])[O-]")
    neutral_mol = ligand_from_smiles("c1ccccc1")

    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)

    host_config_no_ions = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, ionic_concentration=0.0, neutralize=False
    )
    # Without neutralizing the system, the protein system may have some charge
    reference_protein_charge = np.sum(
        host_config_no_ions.host_system.nonbonded_all_pairs.params[:, NBParamIdx.Q_IDX]
    ) / np.sqrt(ONE_4PI_EPS0)

    # Can't mix ligands of different charges when neutralizing the system
    if neutralize:
        with pytest.raises(AssertionError):
            build_protein_system(
                host_pdbfile,
                DEFAULT_PROTEIN_FF,
                DEFAULT_WATER_FF,
                mols=[positive_mol, negative_mol],
                ionic_concentration=ionic_concentration,
                neutralize=neutralize,
            )
    else:
        build_protein_system(
            host_pdbfile,
            DEFAULT_PROTEIN_FF,
            DEFAULT_WATER_FF,
            mols=[positive_mol, negative_mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
    host_config = build_protein_system(
        host_pdbfile,
        DEFAULT_PROTEIN_FF,
        DEFAULT_WATER_FF,
        mols=[],
        ionic_concentration=ionic_concentration,
        neutralize=neutralize,
    )
    input_host_charge = int(np.rint(reference_protein_charge))
    expected_charge = reference_protein_charge
    if neutralize:
        expected_charge = 0
    validate_host_config_ions_and_charge(
        host_config,
        None,
        ionic_concentration,
        expected_charge,
        neutralize,
        input_host_charge=input_host_charge,
    )
    for mol in [positive_mol, negative_mol, neutral_mol]:
        host_config = build_protein_system(
            host_pdbfile,
            DEFAULT_PROTEIN_FF,
            DEFAULT_WATER_FF,
            mols=[mol],
            ionic_concentration=ionic_concentration,
            neutralize=neutralize,
        )
        expected_charge = reference_protein_charge
        if neutralize:
            # Since the ligand isn't in the system, should be missing the charge of the ligand
            expected_charge = -Chem.GetFormalCharge(mol)
        validate_host_config_ions_and_charge(
            host_config,
            mol,
            ionic_concentration,
            expected_charge,
            neutralize,
            input_host_charge=input_host_charge,
        )


@pytest.mark.nocuda
def test_deserialize_protein_system_1_4_exclusions():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)

    exclusion_idxs = host_config.host_system.nonbonded_all_pairs.potential.exclusion_idxs
    scale_factors = host_config.host_system.nonbonded_all_pairs.potential.scale_factors

    kvs = dict()
    for (src, dst), (q_sf, lj_sf) in zip(exclusion_idxs, scale_factors):
        kvs[(src, dst)] = (q_sf, lj_sf)

    # 1-4 torsion between H-ACE and carbonyl=O, expected behavior:
    # we should remove 1/6th of the electrostatic strength
    # we should remove 1/2 of the lennard jones strength
    np.testing.assert_almost_equal(kvs[(2, 3)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 3)][1], 0.5, decimal=4)

    np.testing.assert_almost_equal(kvs[(2, 4)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 4)][1], 0.5, decimal=4)

    np.testing.assert_almost_equal(kvs[(2, 5)][0], 0.5, decimal=4)  # TODO: differs from OFF 1/6
    np.testing.assert_almost_equal(kvs[(2, 5)][1], 0.5, decimal=4)

    # 1-3 angle term should be completely removed
    np.testing.assert_almost_equal(kvs[(3, 4)][0], 1.0, decimal=4)
    np.testing.assert_almost_equal(kvs[(3, 4)][1], 1.0, decimal=4)


@pytest.mark.nocuda
def test_build_protein_system_waters_before_protein():
    num_waters = 100
    # Construct a PDB file with the waters before the protein, to verify that the code correctly handles it.
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        host_pdbfile = app.PDBFile(str(pdb_path))

    host_ff = app.ForceField(f"{DEFAULT_PROTEIN_FF}.xml", f"{DEFAULT_WATER_FF}.xml")

    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)
    modeller.addSolvent(host_ff, numAdded=num_waters, neutralize=False, model=get_water_ff_model(DEFAULT_WATER_FF))
    assert modeller.getTopology().getNumAtoms() == num_waters * 3

    modeller.add(host_pdbfile.topology, host_pdbfile.positions)

    with NamedTemporaryFile(suffix=".pdb") as temp:
        with open(temp.name, "w") as ofs:
            app.PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), file=ofs)

        build_protein_system(temp.name, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)


def test_build_protein_system():
    rng = np.random.default_rng(2024)
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, box_margin=0.1)
    num_host_atoms = host_config.conf.shape[0] - host_config.num_water_atoms

    host_with_mols_config = build_protein_system(
        host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b], box_margin=0.1
    )
    num_host_atoms_with_mol = host_with_mols_config.conf.shape[0] - host_with_mols_config.num_water_atoms

    assert num_host_atoms == num_host_atoms_with_mol
    assert host_config.num_water_atoms == host_with_mols_config.num_water_atoms
    # Water in the pocket will be deleted if mol provided
    assert compute_box_volume(host_config.box) != compute_box_volume(host_with_mols_config.box)

    for bp in host_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(host_config.conf, host_config.box, compute_u=False)
        check_force_norm(-du_dx)

    for bp in host_with_mols_config.host_system.get_U_fns():
        (
            du_dx,
            _,
        ) = bp.to_gpu(np.float32).bound_impl.execute(
            host_with_mols_config.conf, host_with_mols_config.box, compute_u=False
        )
        check_force_norm(-du_dx)

    # Pick a random water atom, will center the ligands on the atom and verify that the box is slightly
    # larger
    water_atom_idx = rng.choice(host_with_mols_config.num_water_atoms)
    new_ligand_center = host_with_mols_config.conf[num_host_atoms_with_mol + water_atom_idx]
    for mol in [mol_a, mol_b]:
        conf = get_romol_conf(mol)
        centroid = np.mean(conf, axis=0)
        conf = conf - centroid + new_ligand_center
        set_romol_conf(mol, conf)

    moved_host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b])
    assert moved_host_config.num_water_atoms == host_with_mols_config.num_water_atoms
    host_atoms_with_moved_ligands = moved_host_config.conf.shape[0] - moved_host_config.num_water_atoms
    assert num_host_atoms == host_atoms_with_moved_ligands
    assert compute_box_volume(host_config.box) < compute_box_volume(moved_host_config.box)


@pytest.mark.nocuda
def test_build_host_config_from_omm():
    """Verify that it is possible to setup a system that is not handled by the default build_protein_system

    This test demonstrates setting up a system with DNA.
    """
    lipid_patch = Path(app.__file__).parent / "data" / "POPC.pdb"
    with pytest.raises(ValueError, match=r"No template found for residue 1"):
        build_protein_system(str(lipid_patch), DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF)

    omm_pdb = app.PDBFile(str(lipid_patch))
    host_ff = app.ForceField("amber14/lipid17.xml", f"{DEFAULT_WATER_FF}.xml")
    modeller = app.Modeller(omm_pdb.topology, omm_pdb.positions)
    host_config = build_host_config_from_omm(
        modeller,
        host_ff,
        padding=1.0,
        box_margin=0.1,
    )
    assert host_config is not None
    # Host atoms are excluded from membrane atom detection. This is janky...
    assert host_config.num_membrane_atoms == 0
    assert host_config.num_water_atoms > 0

    called = False

    def verify_system_build_func_without_ions(ff, modeller, residue_templates):
        nonlocal called
        called = True
        assert len(residue_templates) == 0, "Residues unexpectedly included"
        assert isinstance(ff, app.ForceField)
        assert isinstance(modeller, app.Modeller)
        return construct_default_omm_system(ff, modeller, residue_templates)

    host_config = build_host_config_from_omm(
        modeller,
        host_ff,
        padding=0.5,
        box_margin=0.1,
        construct_system_func=verify_system_build_func_without_ions,
    )
    assert host_config is not None
    assert host_config.num_membrane_atoms == 0
    assert host_config.num_water_atoms > 0

    assert called

    called = False

    def verify_system_build_func_with_ions(ff, modeller, residue_templates):
        nonlocal called
        called = True
        # Should have residue templates thanks to the ions
        assert len(residue_templates) > 0, "No residues included"
        assert isinstance(ff, app.ForceField)
        assert isinstance(modeller, app.Modeller)
        return construct_default_omm_system(ff, modeller, residue_templates)

    host_config = build_host_config_from_omm(
        modeller,
        host_ff,
        padding=0.5,
        box_margin=0.1,
        ionic_concentration=0.15,
        construct_system_func=verify_system_build_func_with_ions,
    )
    assert host_config is not None
    assert host_config.num_membrane_atoms == 0
    assert host_config.num_water_atoms > 0

    assert called


@pytest.mark.nocuda
def test_build_protein_system_removal_of_clashy_waters_in_pdb():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.cdk8", "ligands.sdf") as sdf_path:
        mols_by_name = read_sdf_mols_by_name(sdf_path)
    mol_a = mols_by_name["43"]
    mol_b = mols_by_name["44"]

    with path_to_internal_file("tmd.testsystems.fep_benchmark.cdk8", "cdk8_structure.pdb") as pdb_path:
        host_pdbfile = str(pdb_path)
    pdb_obj = app.PDBFile(host_pdbfile)
    pdb_coords = strip_units(pdb_obj.positions)

    box = get_box_from_coords(pdb_coords)

    mols = [mol_a, mol_b]
    ligand_coords = np.concatenate([get_romol_conf(mol) for mol in mols])

    cutoff = 0.2

    clashy_idxs = idxs_within_cutoff(pdb_coords, ligand_coords, box, cutoff=cutoff)
    assert len(clashy_idxs) >= 1
    clash_set = set(clashy_idxs.tolist())
    clashy_residues = [res for res in pdb_obj.topology.residues() for atom in res.atoms() if atom.index in clash_set]
    assert all(res.name == WATER_RESIDUE_NAME for res in clashy_residues)

    host_config = build_protein_system(host_pdbfile, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=mols, box_margin=0.1)

    clashy_idxs = idxs_within_cutoff(host_config.conf, ligand_coords, host_config.box, cutoff=cutoff)
    clashy_residues = [
        res for res in host_config.omm_topology.residues() for atom in res.atoms() if atom.index in clashy_idxs
    ]
    assert len(clashy_idxs) == 0


@pytest.mark.nocuda
def test_build_protein_wat_residue_names():
    """Test to verify that if waters are specified with the WAT residue name, that they are correctly identified with the WATER_RESIDUE_NAME.
    If this isn't the case our handling of waters is invalid
    """
    with NamedTemporaryFile(suffix=".pdb") as temp:
        with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as protein_path:
            # Convert the HOH water code to WAT code
            with open(protein_path) as ifs:
                with open(temp.name, "w") as ofs:
                    for line in ifs.readlines():
                        ofs.write(line.replace(WATER_RESIDUE_NAME, "WAT"))
        host_config = load_pdb_system(temp.name, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, box_margin=0.1)
        water_res_names_match = [
            res.name == WATER_RESIDUE_NAME for res in host_config.omm_topology.residues() if len(list(res.atoms())) == 3
        ]
        assert len(water_res_names_match) > 0 and all(water_res_names_match)


@pytest.mark.nocuda
def test_adjusting_water_order_doesnt_change_positions():
    """Verify that adjusting the location of waters does not change the positions"""
    cdk8_system = Path(__file__).parent / "data" / "cdk8_incorrectly_ordered_waters.pdb"

    host_pdbfile = app.PDBFile(str(cdk8_system))

    modeller = app.Modeller(host_pdbfile.topology, host_pdbfile.positions)
    water_residues = [residue for residue in modeller.topology.residues() if residue.name == WATER_RESIDUE_NAME]
    water_indices = np.concatenate([[a.index for a in res.atoms()] for res in water_residues])
    assert np.any(np.diff(water_indices) != 1)

    adjusted_modeller = deepcopy(modeller)
    make_waters_contiguous(adjusted_modeller)
    assert adjusted_modeller.topology.getNumAtoms() == modeller.topology.getNumAtoms()

    updated_positions = strip_units(adjusted_modeller.positions)
    ref_positions = strip_units(modeller.positions)

    # Ordering has changed, so positions won't match
    assert not np.all(updated_positions == ref_positions)

    new_water_residues = [
        residue for residue in adjusted_modeller.topology.residues() if residue.name == WATER_RESIDUE_NAME
    ]
    new_water_indices = np.concatenate([[a.index for a in res.atoms()] for res in new_water_residues])

    # Verify that the water positions are identical
    np.testing.assert_equal(updated_positions[new_water_indices], ref_positions[water_indices])

    all_idxs = np.arange(modeller.topology.getNumAtoms())
    reference_other_idxs = np.delete(all_idxs, water_indices)
    updated_other_idxs = np.delete(all_idxs, new_water_indices)

    np.testing.assert_equal(updated_positions[updated_other_idxs], ref_positions[reference_other_idxs])
