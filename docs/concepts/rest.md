# TMD REST


The [REST](https://doi.org/10.1021/jp204407d) (Replica Exchange with Solute Tempering) method allows for enhanced sampling by selectively scaling certain interactions based on a temperature scale factor. You can customize which atoms participate in the REST region to focus sampling enhancement on specific parts of your system.

## Default REST Region Setup

By default, the REST region is determined as follows:

**Important Notes:**
- The REST region is determined automatically based on:
  1. All alchemical atoms
  2. Complete rings containing alchemical atoms
  3. Terminal atoms connected to REST atoms

In addition to the REST region, the force constant of proper torsions that have atoms in the REST region and are part of a rotatable bond or an aliphatic ring are weakened. It may be desirable to expand the REST region to cover torsions if the sampling is slow.

## Custom REST Region Methods

Sometimes it can be useful to select a custom REST region, such as when a particular moiety is difficult to sample when not included in the alchemical region. There are two ways to specify a custom REST region:


### 1. Using SMARTS patterns (Example Script)

TMD supports providing a set of SMARTS patterns that can be used to expand the REST region. These SMARTS patterns do not have to match atoms in every molecule. The following is an example of passing SMARTS patterns to the `run_rbfe_legs.py`.

**Example Usage:**

```bash
python examples/run_rbfe_legs.py \
    --sdf_path ligands.sdf \
    --mol_a ligand_A \
    --mol_b ligand_B \
    --pdb_path complex.pdb \
    --rest_max_temperature_scale 3.0 \
    --rest_temperature_scale_interpolation exponential \
    --rest_smarts_patterns "[O]"
```

### 2. Using Atom Flags (Programmatic Approach)

You can mark individual atoms as included or excluded from the REST region by setting boolean properties on RDKit molecules before creating the `SingleTopologyREST` object. If an atom doesn't have a boolean property, the default REST behavior is relied upon.

**Atom Property Flag:**
- Property Name: `"TMDRESTAtom"` (available as variable at `tmd.fe.rest.single_topology.REST_REGION_ATOM_FLAG`)
- Values: `True` (include in REST region) or `False` (exclude from REST region)

**Example:**

```python
from rdkit import Chem
from tmd.fe.rest.single_topology import SingleTopologyREST, REST_REGION_ATOM_FLAG

# Create or load your molecule
mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))  # benzene example

# Mark specific atoms to be included in REST region
atom_idx_to_include = 0
mol.GetAtomWithIdx(atom_idx_to_include).SetBoolProp(REST_REGION_ATOM_FLAG, True)

# Mark specific atoms to be excluded from REST region
atom_idx_to_exclude = 1
mol.GetAtomWithIdx(atom_idx_to_exclude).SetBoolProp(REST_REGION_ATOM_FLAG, False)

# Create SingleTopologyREST - atoms with flags will be respected
st = SingleTopologyREST(
    mol_a, mol_b, core, forcefield,
    max_temperature_scale=3.0,
    temperature_scale_interpolation="exponential"
)
```

## Storing Custom REST Region Atoms in SDF Files

To set REST region flags in an SDF file:

```python
from rdkit import Chem
from tmd.fe.rest.single_topology import REST_REGION_ATOM_FLAG

mol = Chem.MolFromMolFile("input.sdf")

# Mark atoms for REST region
for idx in [0, 1, 2]:  # atom indices
    mol.GetAtomWithIdx(idx).SetBoolProp(REST_REGION_ATOM_FLAG, True)

# Mark atoms to exclude from REST region
for idx in [5, 6]:
    mol.GetAtomWithIdx(idx).SetBoolProp(REST_REGION_ATOM_FLAG, False)

# Make sure to add a property list, else the properties won't be written out
Chem.CreateAtomBoolPropertyList(mol, REST_REGION_ATOM_FLAG)

with Chem.SDWriter("output.sdf") as writer:
    writer.write(mol)
```
