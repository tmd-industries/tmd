# Relative Binding Free Energy Tutorial

This tutorial covers setting up a [Relative Binding Free Energy](https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/) (RBFE) graph using the TYK2 dataset from the [RBFE datasets](https://github.com/tmd-industries/rbfe-datasets/) repository.

## Installation

Refer to the [installation details](https://github.com/tmd-industries/tmd?tab=readme-ov-file#installation) for initial setup.

## (optional) Determine Optimal MPS Performance

Nvidia’s [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) (MPS) improves [throughput of multiple simulations](https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/) on a single GPU. Enabling MPS is important for maximizing GPU utilization and performance with TMD, especially when leveraging [local md](https://pubmed.ncbi.nlm.nih.gov/37706456/). Running with MPS can provide up to a 4x improvement in throughput, though this depends on the GPU.

TMD provides the `examples/mps_benchmark.py` script for determining the appropriate number of MPS processes for production.

```
nvidia-cuda-mps-control -d # Start MPS if not already enabled

# Run benchmarks script to determine the number of MPS processes
python examples/mps_benchmark.py --processes 1 2 4 6 8 10 12 --local_md --local_md_steps 390 --hrex --system pfkfb3-rbfe
```

Select the number of processes that yields the highest ns/day as indicated by the logs.

> [!WARNING]
> Increasing the number of MPS processes also increases memory and disk usage, potentially leading to crashes. This is system and protocol dependent.

## (optional) Experimental Batched MD

As of 0.3.0 TMD supports batching simulations directly on the GPU without the need for MPS. This can be enabled setting environment variable `TMD_BATCH_MODE=on`. If using this option be sure to set `--mps_workers 1` when running a graph, else both MPS and batching will run which can hurt performance and lead to poor performance. The benefit of batching MD rather than using MPS is the ability to reduce the runtime of legs to below 1 hour, allowing better use of Spot Instances in the cloud.

> [!WARNING]
> The experimental batched mode currently can consume large amounts of GPU memory and can lead to OOMs. CDK8 (~70k atoms) has OOM'd on GPUs with 40GB, so it is suggested to test your system before committing to batching. In the future the memory consumption will be addressed and batching will become the standard for TMD.

## Build a RBFE Graph

RBFE requires defining a series of edges connecting different compounds. This example builds a graph with a k min cut of 3, providing a robust estimate of node predictions at the cost of increased runtime.

```
python examples/build_rbfe_graph.py ../rbfe-datasets/jacs/tyk2/ligands.sdf tyk2_map.json --greedy_k_min_cut 3 --verbose
```

> [!IMPORTANT]
> When building maps, it is recommended to keep the number of dummy atoms below 30. Transformations with large numbers of dummy atoms often fail to converge and may produce unreliable predictions.

## Running the RBFE Graph

Once the `tyk2_map.json` map is generated, you can run an RBFE simulation. First, you’ll need to select a forcefield.

### Select a Forcefield

TMD relies on [Open Forcefield forcefields](https://openforcefield.org/force-fields/force-fields/) for small molecule parameterization and [Amber Forcefields in OpenMM](https://github.com/openmm/openmm/tree/master/wrappers/python/openmm/app/data) for protein and solvent parameterization.

For this example, we will use the OpenFF 2.0.0 forcefield with Amber AM1BCCELF10 charges, as the Tyk2 dataset includes pre-generated Amber AM1BCCELF10 charges and Amber doesn't require a license.

> [!NOTE]
> TMD recommends [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection) charges for their reliability and speed of generation. Amber AM1BCC charges are also reliable and don't require a license, but can be slower to generate.  Precomputed charges via methods like [NAGL](https://github.com/openforcefield/openff-nagl-models) or RESP are also possible but are not covered in this tutorial.

#### Forcefield Defaults

* Water Forcefield: amber14/tip3p
* Protein Forcefield: amber99sbildn
* Ligand Forcefield: OpenFF 2.0.0
* Ligand Charges: [OpenEye AM1BCCELF10](https://docs.eyesopen.com/toolkits/python/quacpactk/molchargetheory.html#elf-conformer-selection)

### Prepare the PDB

The Tyk2 PDB structure has already been prepared for this tutorial. In practice, the PDB structure must be prepared for OpenMM, typically by capping residues and mutating non-standard residues to standard residues.

### Run Graph

Now, you can run the RBFE graph. This example uses 390 local MD steps followed by 10 global MD steps. You may want to reduce the number of local MD steps, depending on your system. Running with 390 local MD steps has been shown to be 18x faster on the CDK2 dataset with similar performance, but your results may vary.

> [!IMPORTANT]
> Ensure you set `--mps_workers` to the appropriate value. 6 may be too low for larger GPUs and too high for smaller GPUs.

```
python examples/run_rbfe_graph.py --sdf_path ../rbfe-datasets/jacs/tyk2/ligands.sdf --graph_json tyk2_map.json --pdb_path ../rbfe-datasets/jacs/tyk2/tyk2_structure.pdb --local_md_steps 390 --mps_workers 6 --forcefield smirnoff_2_0_0_amber_am1bcc.py --output_dir tyk2_rbfe_tutorial --legs complex solvent
```

On a RTX 4090, a typical Tyk2 graph should take approximately 6 hours, though this may take a day or two depending on the GPU.

Once finished, the CSVs `dg_results.csv` and `ddg_results.csv` will be written to `tyk2_rbfe_tutorial/`, which can be used to plot the RBFE run’s performance. The directory also contains plots for analyzing each edge’s simulation.

> [!NOTE]
> The command is idempotent, so if you need to rerun the command all completed legs will be skiped.
