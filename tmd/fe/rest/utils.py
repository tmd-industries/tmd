import warnings

from rdkit import Chem

from tmd.fe.rest.single_topology import REST_REGION_ATOM_FLAG
from tmd.fe.utils import get_mol_name, match_smarts


def assign_rest_atoms_from_smarts(mol: Chem.Mol, smarts: str):
    """Assigns flags to an RDKit Mol that indicate atoms should be part of the REST region using a SMARTS pattern.

    Refer to tmd.fe.rest.single_topology.SingleTopologyREST for more details on REST

    Flags are preserved when written out using RDKit's SDWriter.
    """
    matches = match_smarts(mol, smarts)
    if len(matches) == 0:
        warnings.warn(f"Mol {get_mol_name(mol)} failed to find REST atoms with SMARTS '{smarts}'")
        return
    for match in matches:
        for idx in match:
            mol.GetAtomWithIdx(idx).SetBoolProp(REST_REGION_ATOM_FLAG, True)
    # Create AtomBoolPropertyList to ensure flags are preserved when serialized to SDF
    Chem.CreateAtomBoolPropertyList(mol, REST_REGION_ATOM_FLAG)
