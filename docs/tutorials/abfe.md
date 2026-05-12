# Absolute Binding Free Energy Tutorial

This tutorial covers setting up an [Absolute Binding Free Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/) (ABFE) calculation using the TYK2 dataset from the [RBFE datasets](https://github.com/tmd-industries/rbfe-datasets/) repository.

> [!WARNING]
> ABFE support is experimental. The current protocol has performed reasonably on some systems and poorly on others.

## Installation

Refer to the [installation details](https://github.com/tmd-industries/tmd?tab=readme-ov-file#installation) for initial setup.

For guidance on optimizing TMD performance with MPS or experimental batched MD, see the [benchmarking guide](benchmarking.md).

## Run an ABFE Calculation

ABFE calculates the absolute binding free energy of a ligand to its receptor by decoupling the ligand from the complex (protein + solvent) and from solvent. The binding free energy is computed as:

```
Delta_G_binding = -Delta_G_complex + Delta_G_solvent + correction
```

The complex leg uses Boresch-style restraints to keep the ligand in the binding site when it is decoupled, and an analytical correction accounts for the free energy of the restraints.

### Select a Forcefield

TMD relies on [Open Forcefield forcefields](https://openforcefield.org/force-fields/force-fields/) for small molecule parameterization and [Amber Forcefields in OpenMM](https://github.com/openmm/openmm/tree/master/wrappers/python/openmm/app/data) for protein and solvent parameterization.

For this example, we will use the OpenFF 2.0.0 forcefield with Amber AM1BCCELF10 charges.

> [!NOTE]
> TMD recommends [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection) charges for their reliability and speed of generation. Amber AM1BCCELF10 charges are also reliable and don't require a license, but can be slower to generate. Precomputed charges via methods like [NAGL](https://github.com/openforcefield/openff-nagl-models) or RESP are also possible.

#### Forcefield Defaults

* Water Forcefield: amber14/tip3p
* Protein Forcefield: amber99sbildn
* Ligand Forcefield: OpenFF 2.0.0
* Ligand Charges: [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection)

### Prepare the PDB

The TYK2 PDB structure has already been prepared for this tutorial. In practice, the PDB structure must be prepared for OpenMM, typically by capping residues and mutating non-standard residues to standard residues.

### Running the Simulation

```
python examples/run_abfe.py --sdf_path ../rbfe-datasets/datasets/jacs_set/tyk2/ligands.sdf --pdb_path ../rbfe-datasets/datasets/jacs_set/tyk2/tyk2_structure.pdb --local_md_steps 390 --mps_workers 6 --forcefield smirnoff_2_0_0_amber_am1bcc.py --output_dir tyk2_abfe_tutorial --legs complex solvent
```

On an RTX 4090, a typical Tyk2 ABFE run should take approximately 6-12 hours depending on the number of legs and GPU, though this may take longer for larger systems.

Once finished, the CSV `dg_results.csv` will be written to the output directory, along with per-ligand directories containing simulation plots and results.

> [!NOTE]
> The command is idempotent, so if you need to rerun it, all completed legs will be skipped.
