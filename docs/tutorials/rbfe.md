# Relative Binding Free Energy Tutorial

This tutorial covers setting up an [Relative Binding Free Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/) (RBFE) graph. It will use the TYK2 dataset in the [RBFE datasets](https://github.com/tmd-industries/rbfe-datasets/) repository.

## Installation

Refer to the [installation details](https://github.com/tmd-industries/tmd?tab=readme-ov-file#installation) for initial setup.


## (optional) Determine Optimal MPS Performance

Nvidia provides a [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) (MPS) that improves [throughput of multiple simulations](https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/) on a single GPU. Enabling MPS is important to get the most from your GPU as well as from TMD, especially if leveraging [local md](https://pubmed.ncbi.nlm.nih.gov/37706456/). Running with MPS can provide up to a 4x improvement in throughput, though it is dependent on the GPU.

TMD provides the `examples/mps_benchmark.py` script for determining the appropriate number of MPS processes to run in production.

```
nvidia-cuda-mps-control -d # Start MPS if not already enabled

# Run benchmarks script to determine the number of MPS processes
python examples/mps_benchmark.py --processes 1 2 4 6 8 10 12 --local_md --local_md_steps 390 --hrex --system pfkfb3-rbfe
```

Pick the number of processes that produces the highest ns/day as indicated by the logs.

> [!WARNING]
> Increasing the number of MPS processes also increases the memory and disk usage of RBFE, which may lead to crashes in production. This is dependent on the the system and the parameters used to run the simulations.

## Build a RBFE Graph

RBFE requires setting up a series of edges to connect different compounds. In this example, we will build a graph that has a k min cut of 3. This helps provide a robust estimate of node predictions, at the cost of run time.

```
python examples/build_rbfe_graph.py ../rbfe-datasets/jacs/tyk2/ligands.sdf tyk2_map.json --greedy_k_min_cut 3 --verbose
```

> [!IMPORTANT]
> When building maps for datasets, it is recommended to keep the number of dummy atoms below 30. Transformations with large numbers of dummy atoms tend not to converge and will likely produce unreliable predictions.


## Running the RBFE Graph

Once the `tyk2_map.json` map has been generated, it is possible to run an RBFE simulation. First we will need to determine the forcefield.

### Select a Forcefield
TMD relies on the [Open Forcefield forcefields](https://openforcefield.org/force-fields/force-fields/) for the small molecule parameterization and the [Amber Forcefields in OpenMM](https://github.com/openmm/openmm/tree/master/wrappers/python/openmm/app/data) for protein and solvent parameterization.

For this example, we will be using the OpenFF 2.0.0 forcefield with Amber AM1BCCELF10 charges. This is because the Tyk2 dataset has pre-generated Amber AM1BCCELF10 charges and Amber doesn't require a license.

> [!NOTE]
> TMD recommends the use of [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection) charges for their reliable charges and for the speed of generating charges. Amber AM1BCC charges are also reliable, however generating charges can be quite slow. It is also possible to use precomputed charges via methods like [NAGL](https://github.com/openforcefield/openff-nagl-models) or RESP, but that is not covered in this tutorial.

#### Forcefield Defaults
* Water Forcefield: amber14/tip3p
* Protein Forcefield: amber99sbildn
* Ligand Forcefield: OpenFF 2.0.0
* Ligand Charges: [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection)


### Prepare the PDB

For this tutorial the Tyk2 PDB structure has already been prepared. In practice the PDB structure will have to prepared such that OpenMM can read it. This typically requires capping the residues and mutating non-standard residues to standard residues.

### Run Graph

At this stage we can run the RBFE graph. In this example we will be running 390 local MD steps followed by 10 global MD steps. In practice you may want to reduce the number of local MD steps, depending on your system. Running with 390 local MD steps has been shown to be 18x faster on the CDK2 dataset with similar performance, but your mileage may vary depending on the target and chemical series.

> [!IMPORTANT]
> Make sure to set `--mps_workers` to the appropriate value. 6 is likely too low for larger GPUs and may be too high for smaller GPUs.

```
python examples/run_rbfe_graph.py --sdf_path ../rbfe-datasets/jacs/tyk2/ligands.sdf --graph_json tyk2_map.json --pdb_path ../rbfe-datasets/jacs/tyk2/tyk2_structure.pdb --local_md_steps 390 --mps_workers 6 --forcefield smirnoff_2_0_0_amber_am1bcc.py --output_dir tyk2_rbfe_tutorial
```

On a RTX 4090 a typical Tyk2 graph should only take about 6 hours, however this may take a day or two depending on the GPU.

Once finished the CSVs `dg_results.csv` and `ddg_results.csv` will be written to `tyk2_rbfe_tutorial/`, which can be used to plot the performance of the RBFE run. The directory also contains plots for analyzing the simulation of each edge.

> [!NOTE]
> The command is idempotent, so if you need to kill the process you can just run it again and it will skip any finished tasks.
