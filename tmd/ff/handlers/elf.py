# Copyright 2025, Forrest York
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

"""
Re-implementation of OpenEye ELF(Electrostatically Least-interacting Functional groups) approach.

https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralRad
from scipy.spatial.distance import squareform

from tmd.fe.utils import get_romol_conf
from tmd.ff.handlers.utils import canonicalize_bond
from tmd.graph_utils import convert_to_nx
from tmd.potentials.jax_utils import delta_r

# Dictionary of types and smarts to seek out and handle differently.
SMARTS_TO_CHECK = {
    "trans_cooh": "[#6X3:2](=[#8:1])(-[#8X2H1:3]-[#1:4])",
}


def contains_troublesome_moieties(mol: Chem.Mol) -> bool:
    for pattern in SMARTS_TO_CHECK.values():
        if matches_smarts(mol, pattern):
            return True
    return False


def matches_smarts(mol: Chem.Mol, pattern: str):
    query_mol = Chem.MolFromSmarts(pattern)
    assert query_mol is not None
    return len(mol.GetSubstructMatches(query_mol)) > 0


def prune_conformers_of_troublesome_moieties(mol: Chem.Mol):
    for name, pattern in SMARTS_TO_CHECK.items():
        if name == "trans_cooh":
            half_pi = np.pi / 2.0
            query_mol = Chem.MolFromSmarts(pattern)
            assert query_mol is not None
            confs_to_prune = set()
            for match in mol.GetSubstructMatches(query_mol):
                for conf in mol.GetConformers():
                    if GetDihedralRad(conf, *match) > half_pi:
                        confs_to_prune.add(conf.GetId())
            # Relies on the fact that conformer ids are unique and not consecutive
            for conf_id in confs_to_prune:
                mol.RemoveConformer(conf_id)
        else:
            assert 0, f"Unknown handling of smarts {name}"


def build_12_13_exclusion_set(mol: Chem.Mol) -> set[tuple[int, int]]:
    # Build an exclusion list for 1-2 and 1-3 interactions.
    exclusions = set()

    g = convert_to_nx(mol)
    for path in nx.all_pairs_shortest_path_length(g, cutoff=2):
        src = path[0]
        for dst, length in path[1].items():
            if length == 0:
                continue
            elif length != 1 and length != 2:
                assert 0
            exclusions.add(canonicalize_bond((src, dst)))
    return exclusions


def compute_conformer_electrostatics_mmff94(mol: Chem.Mol, conformers: NDArray) -> NDArray:
    """
    Based off of https://github.com/openforcefield/openff-toolkit/blob/2bf586e036ffc96f631b99914a984ad69a69ef8b/openff/toolkit/utils/rdkit_wrapper.py#L1927
    with modifications to only rely on RDKit + Numpy + TM
    """
    assert len(conformers.shape) == 3
    assert conformers.shape[-1] == 3, "Unexpected dimension to conformers"
    mmff_properties = AllChem.MMFFGetMoleculeProperties(mol, "MMFF94")
    charges = np.array([mmff_properties.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]).reshape(-1, 1)

    abs_charges = np.abs(charges)

    # Generate all of the pairwise distances
    distances = delta_r(conformers[:, :, None], conformers[:, None, :])
    d2_ij = np.sum(distances**2, axis=3)

    d_ij = np.sqrt(d2_ij)

    inv_d_ij = np.reciprocal(d_ij, out=np.zeros_like(d_ij), where=~np.isclose(d_ij, 0.0))

    exclusions_x, exclusions_y = zip(*build_12_13_exclusion_set(mol))

    q_ij = abs_charges @ abs_charges.T
    # Zero out any exclusions
    q_ij[exclusions_x, exclusions_y] = 0.0
    q_ij[exclusions_y, exclusions_x] = 0.0

    conf_energies = 0.5 * np.sum(q_ij * inv_d_ij, axis=(1, 2))
    assert np.all(np.isfinite(conf_energies))
    return conf_energies


def select_diverse_conformers_by_rms(mol: Chem.Mol, limit: int, rms_tolerance: float) -> NDArray:
    """Select a diverse set of conformers as measured by RMS

    Return a set of conformers that are all different by rms_tolerance, up to limit.
    """
    # If there is only a single good conformer, no need to prune by rmsd
    if len(mol.GetConformers()) == 1:
        # Add dimension to make it of shape (1, n_atoms, 3)
        return get_romol_conf(mol)[None, :]

    # Get the RMS matrix, only lower triangular
    rms = AllChem.GetConformerRMSMatrix(mol)
    # Convert to square matrix, and to nanometers to match units of rms_tolerance
    rms_matrix = squareform(rms) / 10.0
    selected = [0]
    while len(selected) < limit:
        rms_of_selected = rms_matrix[selected]
        within_tolerance = np.any(rms_matrix[selected] < rms_tolerance, axis=0)
        if np.all(within_tolerance):
            # No more conformers are diverse given the existing selection
            break
        next_most_diverse = np.argmax(np.where(within_tolerance, 0.0, rms_of_selected.sum(axis=0)))
        selected.append(int(next_most_diverse))

    diverse_confs = np.array(
        [get_romol_conf(mol, conf_id=conf.GetId()) for i, conf in enumerate(mol.GetConformers()) if i in selected]
    )
    return diverse_confs


def prune_conformers_elf(
    mol: Chem.Mol, limit: int = 10, percentage: float = 2.0, rms_tolerance: float = 0.05
) -> Chem.Mol:
    """
    Prune conformers on an RDKit molecule using OpenEye's ELF(Electrostatically Least-interacting Functional groups)
    method.

    Steps:
    1. Compute the MMFF94 electrostatic potential of each conformer
    2. Take a percentage of the lowest energy conformers
    3. Select the most diverse set of conformers by RMS up to the limit.

    Parameters
    ----------
    mol: mol
        Molecule containing conformers

    limit: int
        Maximum number of conformers to return. Defaults to 10

    percentage: float
        Percentage of the lowest energy conformers to keep. Defaults to 2.0

    rms_tolerance: float
        RMS, in nm, to use when selecting diverse compounds. Defaults to 0.05.

    Returns
    -------
        Copy of mol with the valid conformers still available

    Notes
    -----
    Default values correspond to the defaults specified by OpenEye. 2% of lowest energy, with up to 10 conformers.

    References
    ----------
        https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection
    """
    assert 0.0 < percentage <= 100.0
    assert limit > 0
    assert rms_tolerance > 0.0
    if len(mol.GetConformers()) == 0:
        raise ValueError("Mol provided has no conformers")
    assert len(mol.GetConformers()) > 0

    if contains_troublesome_moieties(mol):
        prune_conformers_of_troublesome_moieties(mol)

    if len(mol.GetConformers()) == 0:
        raise ValueError("Mol has no conformers after pruning troublesome moieties")

    conformer_coords = np.array([get_romol_conf(mol, conf_id=conf.GetId()) for conf in mol.GetConformers()])
    conformer_energies = compute_conformer_electrostatics_mmff94(mol, conformer_coords)
    sort_idxs = np.argsort(conformer_energies)

    confs_to_keep = max(1, int(len(sort_idxs) * (percentage / 100.0)))

    # Take top N percent of the lowest energy coords
    good_confs = conformer_coords[sort_idxs[:confs_to_keep]]

    def mol_from_confs(confs: NDArray):
        assert len(confs.shape) == 3
        conf_mol = Chem.Mol(mol)
        conf_mol.RemoveAllConformers()
        for conf in confs:
            new_conf = Chem.Conformer(conf_mol.GetNumAtoms())
            new_conf.SetPositions(conf.astype(np.float64) * 10.0)
            conf_mol.AddConformer(new_conf, assignId=True)
        return conf_mol

    good_conf_mol = mol_from_confs(good_confs)

    final_confs = select_diverse_conformers_by_rms(good_conf_mol, limit, rms_tolerance)

    final_mol = mol_from_confs(final_confs)
    return final_mol
