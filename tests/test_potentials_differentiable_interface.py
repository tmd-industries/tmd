# Copyright 2019-2025, Relay Therapeutics
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import get_romol_conf
from tmd.ff import Forcefield
from tmd.potentials import SummedPotential
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_jax_differentiable_interface(precision):
    """Assert that the computation of U and its derivatives using the
    C++ code path produces equivalent results to doing the
    summation in Python"""
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = get_romol_conf(st.mol_a)
    x_b = get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    box = np.eye(3) * 100

    bps = vac_sys.get_U_fns()
    potentials = [bp.potential for bp in bps]

    sys_params = [np.array(bp.params, dtype=precision) for bp in bps]

    coords = coords.astype(precision)
    box = box.astype(precision)

    gpu_impls = [p.to_gpu(precision) for p in potentials]

    def U_ref(coords, sys_params, box):
        return jnp.sum(jnp.array([f(coords, params, box) for f, params in zip(gpu_impls, sys_params)]))

    U = SummedPotential(potentials, sys_params).to_gpu(precision).call_with_params_list
    args = (coords, sys_params, box)
    np.testing.assert_array_equal(precision(U(*args)), precision(U_ref(*args)))

    argnums = (0, 1)
    dU_dx_ref, dU_dps_ref = jax.grad(U_ref, argnums)(*args)
    dU_dx, dU_dps = jax.grad(U, argnums)(*args)

    np.testing.assert_allclose(dU_dx, dU_dx_ref, rtol=5e-6)

    assert len(dU_dps) == len(dU_dps_ref)
    for dU_dp, dU_dp_ref in zip(dU_dps, dU_dps_ref):
        np.testing.assert_allclose(dU_dp, dU_dp_ref)
