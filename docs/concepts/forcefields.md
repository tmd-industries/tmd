# TMD Forcefields
## Ligand Forcefields
All ligand valence and Lennard-Jones terms currently originate from the [Open Forcefield](https://openforcefield.org/) releases. Though any forcefield that is defined by the Smirnoff format is convertible, with the limitation that virtual sites are not supported. Use the [tmd/ff/smirnoff_converter.py](https://github.com/tmd-industries/tmd/blob/master/tmd/ff/smirnoff_converter.py) script to convert from Smirnoff to the TMD format.

The following forcefields are included with TMD:
- `smirnoff_2_0_0_<charge_model>.py` - OpenFF 2.0.0
- `smirnoff_2_1_0_<charge_model>.py` - OpenFF 2.1.0
- `smirnoff_2_2_1_<charge_model>.py` - OpenFF 2.2.1

> [!TIP]
> TMD suggests Open Forcefield 2.0.0, as later versions can impact RBFE performance; always benchmark systems to determine the most appropriate Forcefield.

### Ligand Charges
TMD supports several ligand charging methods. By default, it utilizes both [OpenEye's AM1BCCELF10](https://www.eyesopen.com/quacpac) (requires an OpenEye license) and Amber AM1BCC charges with ELF10 (license-free, but slower).  Users can also provide user-defined-charges.

> [!TIP]
> TMD suggests OpenEye's AM1BCCELF10 charges for speed and robustness.

TMD includes pre-constructed forcefields with charge type extensions:

- `smirnoff_x_x_x_ccc.py` - Smirnoff with Correctable BCCs, using OpenEye AM1ELF10 as base charges, suitable for BCC refitting.
- `smirnoff_x_x_x_am1bcc.py` - Smirnoff with OpenEye AM1BCCELF10 charges; supports phosphorus.
- `smirnoff_x_x_x_amber_am1bcc.py` - Smirnoff with Amber AM1BCCELF10 charges; may be very slow.
- `smirnoff_x_x_x_amber_am1ccc.py` - Smirnoff with Amber AM1BCCELF10 charges; may be very slow. Suitable for refitting ligand charges.
- `smirnoff_x_x_x_precomputed.py` - Smirnoff with precomputed charges assigned to ligands.

#### Adding Precomputed Charges
To use user-defined-charges, write atom properties compatible with RDKit:

```python
rdkit_mol.SetProp("atom.dprop.PartialCharge", " ".join(str(x) for x in charges))
```

> [!WARNING]
> Loading SDFs with `atom.dprop.PartialCharge` may fail in RDKit if `removeHs=True`. See RDKit [issue 8918](https://github.com/rdkit/rdkit/issues/8918).

## Protein and Water Forcefields
TMD utilizes [OpenMM](https://github.com/openmm/openmm/) for system building, so the protein and water forcefields need to be readable by OpenMM.  The default protein forcefield is `amber99sbildn`, and the default water forcefield is `amber14/tip3p` (identical to `tip3p`, but includes ions).

> [!NOTE]
> TMD excludes the `.xml` suffix to the protein and water forcefield.


It is possible to store forcefields with different protein and water forceields by doing the following:

```python
from dataclasses import replace
from tmd.ff import Forcefield

# Loads smirnoff_2_0_0_ccc.py
ff = Forcefield.load_default()
new_ff = replace(ff, protein_ff="amber14/protein.ff14SB", water_ff="amber14/spce")
with open("smirnoff_2_0_0_ccc_amber14_spce.py", "w") as ofs:
    ofs.write(new_ff.serialize())
```

> [!WARNING]
> 4 or 5 point water models and polarizable forcefields are not supported.

## Forcefield Gotchas
Modifications to forcefields in TMD:

1. Charges are multiplied by `sqrt(ONE_4PI_EPS0)` for optimization.
2. The epsilon parameter in the Lennard-Jones potential is replaced by alpha, where alpha^2 = epsilon, to avoid negative epsilon values during training.
3. A consistent 0.5 scaling is applied to 1-4 terms for both Lennard-Jones and electrostatic interactions.
4. The reaction field used is the real part of PME with a beta (alpha) coefficient of 2.0.
