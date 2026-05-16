# Benchmarking TMD Performance

This tutorial covers the steps to determine optimal performance for your TMD simulations.

## (optional) Determine Optimal MPS Performance

Nvidia's [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) (MPS) improves the [throughput of multiple simulations](https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/) on a single GPU. Enabling MPS is important for maximizing GPU utilization and performance with TMD, especially when leveraging [local md](https://pubmed.ncbi.nlm.nih.gov/37706456/). Running with MPS can provide up to a 4x improvement in throughput, though this depends on the GPU.

TMD provides the `examples/benchmark.py` script for determining the appropriate number of MPS processes for production.

```
nvidia-cuda-mps-control -d # Start MPS if not already enabled

# Run benchmark script to determine the number of MPS processes
python examples/benchmark.py --processes 1 2 4 6 8 10 12 --local_md --local_md_steps 390 --hrex --system pfkfb3-rbfe
```

Select the number of processes that yields the highest ns/day as indicated by the logs.

> [!WARNING]
> Increasing the number of MPS processes also increases CPU/GPU memory and disk usage, potentially leading to crashes. This is system and protocol dependent.

## (optional) Experimental Batched MD

As of 0.3.0 TMD supports batching simulations directly on the GPU without the need for MPS. This can be enabled by setting the environment variable `TMD_BATCH_MODE=on`. If using this option be sure to set `--mps_workers 1` when running a graph, else both MPS and batching will run which can hurt overall performance. The benefit of batching MD rather than using MPS is the ability to reduce the runtime of legs to below 1 hour, allowing for better use of Spot Instances in the cloud.

The `examples/benchmark.py` script can be used to determine the batched performance, though this isn't required in practice since the number of windows will be determined by the lambda schedule in each leg.

```
# Run benchmark script to determine the performance of batched MD
python examples/benchmark.py --processes 1 2 4 6 8 10 12 --local_md --local_md_steps 390 --hrex --system pfkfb3-rbfe --batch_mode
```

> [!WARNING]
> The experimental batched mode currently can consume large amounts of GPU memory and may lead to OOMs. CDK8 (~70k atoms) can consume over 16GB of GPU memory with 48 windows, so it is suggested to test your system before committing to batching. In the future, batching will become the standard for TMD.
