# Relative Binding Free Energy Tutorial

This tutorial covers setting up a [Relative Binding Free Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/) (RBFE) graph using the TYK2 dataset from the [RBFE datasets](https://github.com/tmd-industries/rbfe-datasets/) repository.

## Installation

Refer to the [installation details](https://github.com/tmd-industries/tmd?tab=readme-ov-file#installation) for initial setup.

For guidance on optimizing TMD performance with [Nvidia MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) or experimental batched MD, see the [benchmarking guide](benchmarking.md).

## Build an RBFE Graph

RBFE requires defining a series of edges connecting different compounds. This example builds a graph with a k min cut of 3, providing a robust estimate of node predictions at the cost of increased runtime.

> [!NOTE]
> You will likely see a warning with a message that says `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.`, this is expected and even desirable. The CUDA-enabled JAX can create unexpected issues and is not supported.

```
python examples/build_rbfe_graph.py ../rbfe-datasets/datasets/jacs_set/tyk2/ligands.sdf tyk2_map.json --greedy_k_min_cut 3
```

> [!IMPORTANT]
> When building maps, it is recommended to keep the number of dummy atoms below 30. Transformations with large numbers of dummy atoms often fail to converge and may produce unreliable predictions.

## Running the RBFE Graph

Once the `tyk2_map.json` map is generated, you can run an RBFE simulation. First, you will need to select a forcefield.

### Select a Forcefield

TMD relies on [Open Forcefield forcefields](https://openforcefield.org/force-fields/force-fields/) for small molecule parameterization and [Amber Forcefields in OpenMM](https://github.com/openmm/openmm/tree/master/wrappers/python/openmm/app/data) for protein and solvent parameterization.

For this example, we will use the OpenFF 2.0.0 forcefield with Amber AM1BCCELF10 charges, as the Tyk2 dataset includes pre-generated Amber AM1BCCELF10 charges and Amber doesn't require a license.

> [!NOTE]
> TMD recommends [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection) charges for their reliability and speed of generation. Amber AM1BCCELF10 charges are also reliable and don't require a license, but can be slower to generate. Precomputed charges via methods like [NAGL](https://github.com/openforcefield/openff-nagl-models) or RESP are also possible but are not covered in this tutorial.

#### Forcefield Defaults

* Water Forcefield: amber14/tip3p
* Protein Forcefield: amber99sbildn
* Ligand Forcefield: OpenFF 2.0.0
* Ligand Charges: [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection)

### Prepare the PDB

The TYK2 PDB structure has already been prepared for this tutorial. In practice, the PDB structure must be prepared for OpenMM, typically by capping residues and mutating non-standard residues to standard residues.

### Run Graph

You can now run the RBFE graph. This example uses 390 local MD steps followed by 10 global MD steps per frame. You may want to reduce the number of local MD steps, depending on your system. Using 390 local MD steps has been shown to be 18x faster than using only global MD on the CDK2 dataset, with similar performance, but results may vary depending on the target.

> [!IMPORTANT]
> Ensure you set `--mps_workers` to the appropriate value. 6 may be insufficient for larger GPUs and excessive for smaller GPUs. If [MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) is not enabled, performance will be worse than expected. If MPS is not available consider using the [batched mode](benchmarking.md).

```
python examples/run_rbfe_graph.py --sdf_path ../rbfe-datasets/datasets/jacs_set/tyk2/ligands.sdf --graph_json tyk2_map.json --pdb_path ../rbfe-datasets/datasets/jacs_set/tyk2/tyk2_structure.pdb --local_md_steps 390 --mps_workers 6 --forcefield smirnoff_2_0_0_amber_am1bcc.py --output_dir tyk2_rbfe_tutorial --legs complex solvent
```

On an RTX 4090, a typical Tyk2 graph should take approximately 6 hours, though this may take 1–2 days on a slower GPU.

Once finished, the CSVs `dg_results.csv` and `ddg_results.csv` will be written to `tyk2_rbfe_tutorial/`, which can be used to plot the results of the RBFE run. The directory also contains plots for analyzing each edge.

> [!NOTE]
> The command is idempotent, so if you need to rerun it, all completed legs will be skipped.
