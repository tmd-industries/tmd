import numpy as np
import pytest

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe import atom_mapping
from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.fe.single_topology import AtomMapMixin, SingleTopology
from tmd.fe.topology import BaseTopology, MultiTopology
from tmd.fe.utils import get_romol_conf, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.md.trajectory import frames_to_mdtraj_trajectory
from tmd.utils import path_to_internal_file

with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as ligands_path:
    hif2a_ligands = read_sdf_mols_by_name(ligands_path)

forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")


@pytest.mark.nogpu
@pytest.mark.mdtraj
@pytest.mark.parametrize(
    "topology",
    [
        BaseTopology(hif2a_ligands["23"], forcefield),
        MultiTopology(list(hif2a_ligands.values()), forcefield),
        AtomMapMixin(
            hif2a_ligands["23"],
            hif2a_ligands["338"],
            atom_mapping.get_cores(hif2a_ligands["23"], hif2a_ligands["338"], **DEFAULT_ATOM_MAPPING_KWARGS)[0],
        ),
        SingleTopology(
            hif2a_ligands["23"],
            hif2a_ligands["338"],
            atom_mapping.get_cores(hif2a_ligands["23"], hif2a_ligands["338"], **DEFAULT_ATOM_MAPPING_KWARGS)[0],
            forcefield,
        ),
        SingleTopologyREST(
            hif2a_ligands["23"],
            hif2a_ligands["338"],
            atom_mapping.get_cores(hif2a_ligands["23"], hif2a_ligands["338"], **DEFAULT_ATOM_MAPPING_KWARGS)[0],
            forcefield,
            3.0,
        ),
    ],
)
@pytest.mark.parametrize("leg", ["vacuum", "solvent", "complex"])
def test_frames_to_mdtraj(leg, topology):
    pytest.importorskip("mdtraj")

    mols = []
    if isinstance(topology, MultiTopology):
        mols.extend(topology.mols)
    elif isinstance(topology, BaseTopology):
        mols.append(topology.mol)
    else:
        mols.append(topology.mol_a)
        mols.append(topology.mol_b)

    host_config = None
    if leg == "vacuum":
        assert host_config is None
    elif leg == "solvent":
        host_config = builders.build_water_system(3.0, forcefield.water_ff, mols=mols)
    elif leg == "complex":
        with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as pdb_path:
            host_config = builders.build_protein_system(
                str(pdb_path), forcefield.protein_ff, forcefield.water_ff, mols=mols
            )

    box = np.eye(3) * 100.0
    coords = np.concatenate([get_romol_conf(mol) for mol in mols])
    if host_config is not None:
        coords = np.concatenate([host_config.conf, coords])
        box = host_config.box

    traj = frames_to_mdtraj_trajectory([coords], [box], topology, host_config=host_config, time=[1.0])
    assert traj.xyz.shape == (1, *coords.shape)

    # Verify that each ligand has its own residue
    ligand_residue_atoms = traj.topology.select("resname LIG")
    assert len(ligand_residue_atoms) == sum([mol.GetNumAtoms() for mol in mols])
    lig_residues = [res for res in traj.topology.residues if res.name == "LIG"]
    assert len(lig_residues) == len(mols)
    for res, mol in zip(lig_residues, mols):
        assert res.n_atoms == mol.GetNumAtoms()
