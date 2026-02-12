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

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from openmm import app, openmm, unit
from rdkit import Chem

from tmd.fe.system import HostSystem
from tmd.fe.utils import get_romol_conf
from tmd.ff import get_water_ff_model
from tmd.ff.handlers import openmm_deserializer
from tmd.potentials.jax_utils import idxs_within_cutoff
from tmd.utils import path_to_internal_file

WATER_RESIDUE_NAME = "HOH"
SODIUM_ION_RESIDUE = "NA"
CHLORINE_ION_RESIDUE = "CL"
MAGNESIUM_ION_RESIDUE = "MG"
POPC_RESIDUE_NAME = "POP"

# Custom values defined in tmd/ff/params/openmm_custom_templates.xml
DUMMY_ATOM_TEMPLATE = "DUM"  # Used as a place holder when replacing clashy waters


@dataclass(frozen=True)
class HostConfig:
    host_system: HostSystem
    masses: NDArray
    conf: NDArray
    box: NDArray
    num_water_atoms: int
    omm_topology: app.topology.Topology
    num_membrane_atoms: int = 0

    def __post_init__(self):
        object.__setattr__(self, "masses", np.asarray(self.masses, dtype=np.float32))
        object.__setattr__(self, "conf", np.asarray(self.conf, dtype=np.float32))
        object.__setattr__(self, "box", np.asarray(self.box, dtype=np.float32))


def strip_units(coords) -> NDArray[np.float64]:
    return np.array(coords.value_in_unit_system(unit.md_unit_system))


def get_box_from_coords(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    box_lengths = np.max(coords, axis=0) - np.min(coords, axis=0)
    return np.eye(3) * box_lengths


def get_ion_residue_templates(modeller) -> dict[app.Residue, str]:
    """When using amber99sbildn with the amber14 water models the ion templates get duplicated. This forces
    the use of the NA/CL/MG templates (from the amber14 water models) rather than the amber99sbildn templates.
    """
    residue_templates = {}
    for res_name in (SODIUM_ION_RESIDUE, CHLORINE_ION_RESIDUE, MAGNESIUM_ION_RESIDUE, DUMMY_ATOM_TEMPLATE):
        residue_templates.update({res: res_name for res in modeller.getTopology().residues() if res.name == res_name})
    return residue_templates


def replace_clashy_waters(
    modeller: app.Modeller,
    box: NDArray[np.float64],
    mols: list[Chem.Mol],
    host_ff: app.ForceField,
    water_ff: str,
    clash_distance: float = 0.2,
):
    """Replace waters that clash with a set of molecules with waters at the boundaries rather than
    clashing with the molecules. The number of atoms in the system will be identical before and after

    Note:
    This will modify the host_ff by adding custom OpenMM templates from tmd/ff/params/openmm_custom_templates.xml.
    You may experience collisions with any custom templates.

    Parameters
    ----------
    modeller: app.Modeller
        Modeller to update in place

    box: NDArray[np.float64]
        Box to evaluate PBCs under

    mols: list[Mol]
        List of molecules to determine which waters are clashy

    host_ff: app.ForceField
        The forcefield used for the host

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    clash_distance: float
        Distance from a ligand atom to a water atom to consider as a clash, in nanometers
    """
    if len(mols) == 0:
        return

    with path_to_internal_file("tmd.ff.params", "openmm_custom_templates.xml") as custom_templates_path:
        host_ff.loadFile(str(custom_templates_path))

    ligand_coords = np.concatenate([get_romol_conf(mol) for mol in mols])

    def get_clashy_idxs() -> NDArray[np.int32]:
        # Hard coded value for the maximum number of ligand atoms to consider when evaluating water idxs
        # Without this setting up a system with thousands of molecules can lead to JAX failures
        batch_size = 100

        water_idxs = np.concatenate(
            [[a.index for a in res.atoms()] for res in modeller.topology.residues() if res.name == WATER_RESIDUE_NAME]
        )
        water_coords = strip_units(modeller.positions)[water_idxs]
        idxs_set = set()
        for batch_offset in range(0, len(ligand_coords), batch_size):
            idxs_batch = idxs_within_cutoff(
                water_coords, ligand_coords[batch_offset : batch_offset + batch_size], box, cutoff=clash_distance
            )
            idxs_set = idxs_set.union(set(idxs_batch.tolist()))
        idxs = np.array(list(idxs_set), dtype=np.int32)
        if len(idxs) > 0:
            # Offset the clashy idxs with the first atom idx, else could be pointing at non-water atoms
            idxs += np.min(water_idxs)
        return idxs

    clashy_idxs = get_clashy_idxs()
    if len(clashy_idxs) == 0:
        return
    # Determine the number of atoms in the original system. Should end up with the same number of atoms
    num_system_atoms = len(modeller.positions)

    def get_waters_to_delete():
        all_atoms = list(modeller.topology.atoms())
        waters_to_delete = set()
        for idx in clashy_idxs:
            atom = all_atoms[idx]
            if atom.residue.name == WATER_RESIDUE_NAME:
                waters_to_delete.add(atom.residue)
            else:
                raise RuntimeError(f"Clashy atom is not a water: Residue name {atom.residue.name}")
        return waters_to_delete

    dummy_chain_id = "DUMMY"

    topology = app.Topology()
    dummy_chain = topology.addChain(dummy_chain_id)
    for mol in mols:
        # Add a bunch of dummy Argon atoms in place of the ligand atoms. This will prevent OpenMM from
        # placing atoms in the binding pocket. OpenMM only looks at the parameters of waters, not the solute
        # so this is fine.
        for atom in mol.GetAtoms():
            res = topology.addResidue(DUMMY_ATOM_TEMPLATE, dummy_chain)
            topology.addAtom(DUMMY_ATOM_TEMPLATE, app.Element.getBySymbol("U"), res)
    modeller.add(topology, ligand_coords * unit.nanometers)

    # Get the latest residue templates then update with the input templates
    ion_res_templates = get_ion_residue_templates(modeller)

    clashy_waters = get_waters_to_delete()
    # First add back in the number of waters that are clashy and we know we need to delete
    modeller.addSolvent(
        host_ff,
        numAdded=len(clashy_waters),
        neutralize=False,
        model=get_water_ff_model(water_ff),
        residueTemplates=ion_res_templates,
    )
    clashy_waters = get_waters_to_delete()
    modeller.delete(list(clashy_waters))
    # Remove the chain filled with the dummy atoms
    ligand_chain = [chain for chain in modeller.topology.chains() if chain.id == dummy_chain_id]
    modeller.delete(ligand_chain)
    assert num_system_atoms == modeller.topology.getNumAtoms(), "replace_clashy_waters changed the number of atoms"


def make_waters_contiguous(modeller):
    """Modifies the OpenMM modeller and topology such that waters are contiguous. Done to ensure that water
    sampling is possible in downstream code.

    If the waters are contiguous the modeller is not modified.

    Parameters
    ----------
    modeller: app.Modeller
        Modeller to update in place
    """
    water_residues = [residue for residue in modeller.topology.residues() if residue.name == WATER_RESIDUE_NAME]
    water_indices = np.concatenate([[a.index for a in res.atoms()] for res in water_residues])
    if np.all(np.diff(water_indices) == 1):
        return

    num_system_atoms = modeller.topology.getNumAtoms()

    water_positions = strip_units(modeller.positions)[water_indices]
    modeller.delete([atom for res in water_residues for atom in res.atoms()])
    topology = app.Topology()
    water_chain = topology.addChain("W")
    for res in water_residues:
        water_res = topology.addResidue(res.name, water_chain)
        old_atom_to_new = {}
        for atom in res.atoms():
            new_atom = topology.addAtom(atom.name, atom.element, water_res)
            old_atom_to_new[atom.id] = new_atom
        for bond in res.bonds():
            atom_a = old_atom_to_new[bond.atom1.id]
            atom_b = old_atom_to_new[bond.atom2.id]
            topology.addBond(atom_a, atom_b, type=bond.type, order=bond.order)
    modeller.add(topology, water_positions * unit.nanometers)
    assert num_system_atoms == modeller.topology.getNumAtoms(), "make_waters_contiguous changed the number of atoms"


def solvate_modeller(
    modeller: app.Modeller,
    box: NDArray[np.float64],
    ff: app.ForceField,
    water_ff: str,
    mols: list[Chem.Mol] | None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
    membrane: bool = False,
):
    """Solvates a system while handling ions and neutralizing the system by adding dummy ions then removing them.

    Parameters
    ----------
    modeller: app.Modeller
        Modeller to update in place

    box: NDArray[np.float64]
        Box to evaluate PBCs under

    mols: list[Mol]
        List of molecules to determine which waters are clashy

    ff: app.ForceField
        The forcefield used for the host

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: list[Mol] or None
        List of molecules to determine the charge. Mols are expected to have the same charge

    ionic_concentration: float
        Molar concentration of ions

    neutralize: bool
        Whether or not to neutralize the system.

    membrane: bool
        Whether or not to add a membrane to the system
    """
    dummy_chain_id = "DUMMY"
    add_dummy_ions = neutralize and mols is not None and len(mols) > 0
    if add_dummy_ions:
        assert mols is not None and len(mols) > 0
        # Since we do not add the ligands to the OpenMM system, we add ions that emulate the net charge
        # of the ligands so that `addSolvent` will neutralize the system correctly.
        charges = [Chem.GetFormalCharge(mol) for mol in mols]
        # If the charges are not the same for all mols, unable to neutralize the system effectively
        # TBD: Decide if we want to weaken this assertion, most likely for charge hopping
        assert all([charge == charges[0] for charge in charges])
        charge = charges[0]
        if charge != 0:
            topology = app.Topology()
            dummy_chain = topology.addChain(dummy_chain_id)
            for _ in range(abs(charge)):
                if charge < 0:
                    res = topology.addResidue(CHLORINE_ION_RESIDUE, dummy_chain)
                    topology.addAtom(CHLORINE_ION_RESIDUE, app.Element.getBySymbol("Cl"), res)
                else:
                    res = topology.addResidue(SODIUM_ION_RESIDUE, dummy_chain)
                    topology.addAtom(SODIUM_ION_RESIDUE, app.Element.getBySymbol("Na"), res)
            coords = np.zeros((topology.getNumAtoms(), 3)) * unit.angstroms
            modeller.add(topology, coords)
        # Get the latest residue templates then update with the input templates
    ion_res_templates = get_ion_residue_templates(modeller)
    if not membrane:
        modeller.addSolvent(
            ff,
            boxSize=np.diag(box) * unit.nanometers,
            ionicStrength=ionic_concentration * unit.molar,
            model=get_water_ff_model(water_ff),
            neutralize=neutralize,
            residueTemplates=ion_res_templates,
        )
    else:
        assert get_water_ff_model(water_ff) == "tip3p", "Only supports tip3p waters"
        modeller.addMembrane(
            ff,
            ionicStrength=ionic_concentration * unit.molar,
            neutralize=neutralize,
            residueTemplates=ion_res_templates,
        )
    if add_dummy_ions:
        current_topo = modeller.getTopology()
        # Remove the chain filled with the dummy ions
        bad_chains = [chain for chain in current_topo.chains() if chain.id == dummy_chain_id]
        modeller.delete(bad_chains)
    try:
        water_res = next(
            [atom for atom in res.atoms()] for res in modeller.topology.residues() if res.name == WATER_RESIDUE_NAME
        )
        assert len(water_res) == 3, "Expect water residues to have three atoms"
    except StopIteration:
        pass


def load_pdb_system(
    host_pdbfile: app.PDBFile | str,
    protein_ff: str,
    water_ff: str,
    box_margin: float = 0.0,
    cutoff: float = 1.2,
) -> HostConfig:
    """
    Load a protein system. Useful for when using an pre-existing system that has been solvated/equilibrated.

    Parameters
    ----------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        The protein forcefield name (excluding .xml) to parametrize the host_pdbfile with.

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    box_margin: Amount of box_margin to add to box
        Avoids clashes within the system

    cutoff: float
        Nonbonded cutoff to use. Defaults to 1.2

    Returns
    -------
    HostConfig
    """

    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)
    host_coords = strip_units(host_pdb.positions)

    water_residues_in_pdb = [residue for residue in host_pdb.topology.residues() if residue.name == WATER_RESIDUE_NAME]
    num_water_atoms = sum([len(list(residue.atoms())) for residue in water_residues_in_pdb])

    ion_res_templates = get_ion_residue_templates(modeller)

    omm_host_system = construct_default_omm_system(host_ff, modeller, ion_res_templates)

    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(
        omm_host_system, cutoff=cutoff
    )

    host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )

    # Note that getPeriodicBoxVectors() can produce a significantly different box
    # to get_box_from_coords. Use getPeriodicBoxVectors() as it appears to produce smaller
    # boxes when loading a PDB on its own
    box = host_pdb.topology.getPeriodicBoxVectors()
    box = strip_units(box)
    box += np.eye(3) * box_margin

    assert modeller.topology.getNumAtoms() == len(host_coords)

    return HostConfig(
        host_system=host_system,
        conf=host_coords,
        box=box,
        num_water_atoms=num_water_atoms,
        num_membrane_atoms=0,
        omm_topology=modeller.topology,
        masses=np.asarray(masses),
    )


def construct_default_omm_system(
    ff: app.ForceField, modeller: app.Modeller, residue_templates: dict[app.Residue, str]
) -> openmm.System:
    """
    Parameters
    ----------
    modeller: app.Modeller
        Modeller with topology and starting coordinates

    host_ff: str
        The OpenMM forcefield defining relevant parameters

    residue_templates: dict[app.Residue, str]
        Residue templates for custom handling of residue templates.
        See https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html#openmm.app.modeller.Modeller.addSolvent
        for more details. Only valid to provide residues that are in the host_pdbfile and if host_pdbfile is an app.PDBFile.
        The residue templates in get_ion_residue_templates will be applied, but will be overridden by any user provided input.

    Returns
    -------
        OpenMM openmm.System
    """
    return ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        residueTemplates=residue_templates,
    )


def build_host_config_from_omm(
    modeller: app.Modeller,
    host_ff: app.ForceField,
    construct_system_func: Callable[
        [app.ForceField, app.Modeller, dict[app.Residue, str]], openmm.System
    ] = construct_default_omm_system,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
    mols: list[Chem.Mol] | None = None,
    padding: float = 1.0,
    box_margin: float = 0.0,
    water_model: str = "tip3p",
    add_membrane: bool = False,
):
    """
    Build a solvated system system from an existing OpenMM modeller object and Forcefield. Useful for having more fine-grain
    control over system parameterization.

    Parameters
    ----------
    modeller: app.Modeller
        Modeller with topology and starting coordinates

    host_ff: str
        The OpenMM forcefield defining relevant parameters

    construct_system_func: func(app.ForceField, app.Modeller, dict[app.Residue, str]) -> openmm.System
        Function used to construct the OpenMM system object. Defaults to tmd.md.builders.construct_default_omm_system
        Function is not allowed to change the number of atoms within the topology.

    ionic_concentration: optional float
        Concentration of ions, in molars, to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    mols: optional list of mols
        Molecules to be part of the system, will avoid placing water molecules that clash with the mols.

    padding: Solvent padding to add to box, in nanometers
        If there are no particles will define the box dimensions

    box_margin: Amount of box_margin to add to box, in nanometers
        Avoids clashes within the system

    water_model: str
        Water model used when adding waters. Can use tmd.ff.get_water_ff_model to get the appropriate
        model. Defaults to tip3p.

    add_membrane: bool
        Whether or not to add a membrane to the system. Defaults to False

    Returns
    -------
    HostConfig
    """
    host_coords = strip_units(modeller.positions)
    box = get_box_from_coords(host_coords)
    box += np.eye(3) * padding

    # Make sure to account for any waters that might come along in the modeller already
    starting_water_atoms = (
        len([residue for residue in modeller.topology.residues() if residue.name == WATER_RESIDUE_NAME]) * 3
    )
    if starting_water_atoms > 0 and mols is not None:
        # Called twice because it is faster to adjust a smaller part of the system
        make_waters_contiguous(modeller)

    num_host_atoms = len(host_coords) - starting_water_atoms

    solvate_modeller(
        modeller,
        box,
        host_ff,
        water_model,
        mols=mols,
        neutralize=neutralize,
        ionic_concentration=ionic_concentration,
        membrane=add_membrane,
    )

    if mols is not None:
        replace_clashy_waters(modeller, box, mols, host_ff, water_model)
    solvated_host_coords = strip_units(modeller.positions)

    ion_res_templates = get_ion_residue_templates(modeller)

    solvated_omm_host_system = construct_system_func(host_ff, modeller, ion_res_templates)

    make_waters_contiguous(modeller)

    assert modeller.topology.getNumAtoms() == solvated_host_coords.shape[0], (
        "Modeller no longer matches number of atoms in the system"
    )

    num_water_atoms = (
        len([residue for residue in modeller.topology.residues() if residue.name == WATER_RESIDUE_NAME]) * 3
    )
    num_membrane_atoms = 0
    if add_membrane:
        num_membrane_atoms = sum(
            [
                len(list(residue.atoms()))
                for residue in modeller.topology.residues()
                if residue.name == POPC_RESIDUE_NAME
            ]
        )

    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(
        solvated_omm_host_system, cutoff=1.2
    )

    solvated_host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )

    # Determine box from the system's coordinates
    box = get_box_from_coords(solvated_host_coords) + np.eye(3) * box_margin

    if num_membrane_atoms > 0:
        print(
            f"building a system with {num_host_atoms:d} host atoms, {num_water_atoms} water atoms and {num_membrane_atoms} membrane atoms"
        )
    else:
        print(f"building a system with {num_host_atoms:d} host atoms and {num_water_atoms} water atoms")

    return HostConfig(
        host_system=solvated_host_system,
        conf=solvated_host_coords,
        box=box,
        num_water_atoms=num_water_atoms,
        num_membrane_atoms=num_membrane_atoms,
        omm_topology=modeller.topology,
        masses=np.asarray(masses),
    )


def build_protein_system(
    host_pdbfile: app.PDBFile | str,
    protein_ff: str,
    water_ff: str,
    mols: list[Chem.Mol] | None = None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
    box_margin: float = 0.0,
) -> HostConfig:
    """
    Build a solvated protein system with a 10A padding.

    Parameters
    ----------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        The protein forcefield name (excluding .xml) to parametrize the host_pdbfile with.

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will avoid placing water molecules that clash with the mols.

    ionic_concentration: optional float
        Concentration of ions, in molars, to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    box_margin: Amount of box_margin to add to box, in nanometers
        Avoids clashes within the system

    Returns
    -------
    HostConfig
    """

    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)

    return build_host_config_from_omm(
        modeller,
        host_ff,
        ionic_concentration=ionic_concentration,
        neutralize=neutralize,
        padding=1.0,
        mols=mols,
        box_margin=box_margin,
    )


def build_membrane_system(
    host_pdbfile: app.PDBFile | str,
    protein_ff: str,
    water_ff: str,
    mols: list[Chem.Mol] | None = None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
    box_margin: float = 0.0,
) -> HostConfig:
    """
    Build a solvated protein+membrane system with a 10A padding. Assumes the PDB file is posed such that the XY plane
    will contain the membrane, this matches with OpenMM.

    Note that this will produce different numbers of water molecules due to non-deterministic minimization in OpenMM

    Parameters
    ----------
    host_pdbfile: str or app.PDBFile
        PDB of the host structure

    protein_ff: str
        The protein forcefield name (excluding .xml) to parametrize the host_pdbfile with.

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will avoid placing water molecules that clash with the mols.
        If water molecules provided in the PDB clash with the mols, will do nothing.

    ionic_concentration: optional float
        Concentration of ions, in molars, to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    box_margin: Amount of box_margin to add to box
        Avoids clashes within the system

    Returns
    -------
    HostConfig
    """

    assert protein_ff == "amber14/protein.ff14SB", "Requires the Amber14SB protein ForceField"
    host_ff = app.ForceField(f"{protein_ff}.xml", f"{water_ff}.xml", "amber14/lipid17.xml")
    if isinstance(host_pdbfile, str):
        assert os.path.exists(host_pdbfile)
        host_pdb = app.PDBFile(host_pdbfile)
    elif isinstance(host_pdbfile, app.PDBFile):
        host_pdb = host_pdbfile
    else:
        raise TypeError("host_pdbfile must be a string or an openmm PDBFile object")

    modeller = app.Modeller(host_pdb.topology, host_pdb.positions)

    return build_host_config_from_omm(
        modeller,
        host_ff,
        ionic_concentration=ionic_concentration,
        neutralize=neutralize,
        padding=1.0,
        mols=mols,
        box_margin=box_margin,
        add_membrane=True,
    )


def build_water_system(
    box_width: float,
    water_ff: str,
    mols: list[Chem.Mol] | None = None,
    ionic_concentration: float = 0.0,
    neutralize: bool = False,
    box_margin: float = 0.0,
) -> HostConfig:
    """
    Build a water system with a cubic box with each side of length box_width.

    Parameters
    ---------
    box_width: float
        The length of each size of the box

    water_ff: str
        The water forcefield name (excluding .xml) to parametrize the water with.

    mols: optional list of mols
        Molecules to be part of the system, will remove water molecules that clash with the mols.

    ionic_concentration: optional float
        Molar concentration of ions to add to the system. Defaults to 0.0, meaning no ions are added.

    neutralize: optional bool
        Whether or not to add ions to the system to ensure the system has a net charge of 0.0. Defaults to False.

    box_margin: Amount of box_margin to add to box
        Avoids clashes within the system

    Returns
    -------
    4-Tuple
        OpenMM host system, coordinates, box, OpenMM topology
    """
    if ionic_concentration < 0.0:
        raise ValueError("Ionic concentration must be greater than or equal to 0.0")
    ff = app.ForceField(f"{water_ff}.xml")

    # Create empty topology and coordinates.
    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)

    box = np.eye(3) * box_width

    solvate_modeller(
        modeller,
        box,
        ff,
        water_ff,
        mols=mols,
        neutralize=neutralize,
        ionic_concentration=ionic_concentration,
    )

    def get_centered_coords():
        host_coords = strip_units(modeller.positions)
        # If mols provided, center waters such that the center is the mols centroid
        # Done to avoid placing mols at the edges and moves the water coordinates to avoid
        # changing the mol coordinates which are finalized downstream of the builder
        if mols is not None and len(mols) > 0:
            mol_coords = np.concatenate([get_romol_conf(mol) for mol in mols])
            mols_centroid = np.mean(mol_coords, axis=0)
            host_centroid = np.mean(host_coords, axis=0)
            host_coords = host_coords - host_centroid + mols_centroid
        return host_coords

    # Don't use build_host_config_from_omm here because of the recentering
    modeller = app.Modeller(modeller.topology, get_centered_coords())

    if mols is not None:
        replace_clashy_waters(modeller, box.astype(np.float64), mols, ff, water_ff)

    solvated_host_coords = strip_units(modeller.positions)

    assert modeller.topology.getNumAtoms() == solvated_host_coords.shape[0]

    ion_res_templates = get_ion_residue_templates(modeller)
    omm_host_system = construct_default_omm_system(ff, modeller, ion_res_templates)

    (bond, angle, proper, improper, nonbonded), masses = openmm_deserializer.deserialize_system(
        omm_host_system, cutoff=1.2
    )

    solvated_host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )

    # Determine box from the system's coordinates
    box = get_box_from_coords(solvated_host_coords) + np.eye(3) * box_margin
    num_water_atoms = len(solvated_host_coords)

    return HostConfig(
        host_system=solvated_host_system,
        conf=solvated_host_coords,
        box=box,
        num_water_atoms=num_water_atoms,
        num_membrane_atoms=0,
        omm_topology=modeller.topology,
        masses=np.asarray(masses),
    )
