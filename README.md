# tmd

![Nightly Tests](https://github.com/badisa/tmd/actions/workflows/nightly-tests.yml/badge.svg)

A high-performance differentiable molecular dynamics and forcefield engine.

This is a fork of [Timemachine](https://github.com/proteneer/timemachine).

## Installation

### Pre-requisites

* Python >=3.12
* RDKit
* OpenMM
* Cuda 12.4+
* CMake 3.24.3
* OpenEye Toolkits (optional)
* AmberTools (optional)

### Setup using Anaconda

If using conda the following can be used to configure your environment. Conda is not required, only important if relying on AM1 charges from AmberTools

```shell
conda env create -f environment.yml
conda activate tmd
```

### Install tmd

The CUDA extension module implementing custom ops is only supported on Linux, but partial functionality is still available on non-Linux OSes.

```shell
pip install -r requirements.txt
pip install .
```

## Developing TMD

### Installing in developer mode

```shell
pip install -r requirements.txt  # Install the pinned requirements
pip install -e .[dev,test]
```

### Running Tests

```shell
pytest
```

Note: we currently only support and test on python 3.12, use other versions at your own peril.

## Forcefield Gotchas

Most of the training is using the correctable charge corrections [ccc forcefield](https://github.com/proteneer/timemachine/blob/1a721dd3f05d6011cf028b0588e066682d38ba59/ff/params/smirnoff_2_0_0_ccc.py), which is SMIRNOFF 2.0.0 augmented with BCCs ported via the [recharge](https://github.com/openforcefield/openff-recharge) project. There are some additional modifications:

1. The charges have been multiplied by sqrt(ONE_4PI_EPS0) as an optimization.
2. The eps parameter in LJ have been replaced by an alpha such that alpha^2=eps in order to avoid negative eps values during training.
3. We use a consistent 0.5 scaling for the 1-4 terms across LJ and electrostatics.
4. The reaction field used is the real part of PME with a beta (alpha) coefficient of 2.0
5. The recharge BCC port is not yet complete, as there are some missing types that will cause very large errors (eg. P=S moieties).

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
