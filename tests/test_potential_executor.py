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

import itertools

import numpy as np
import pytest

from tmd.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from tmd.fe.rbfe import setup_optimized_host
from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import get_romol_conf
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from tmd.md import builders
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.potentials import HarmonicBond
from tmd.potentials.potential import get_bound_potential_by_type
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

pytestmark = [pytest.mark.memcheck]


def get_potentials_and_frames(host_name: str | None, precision):
    dt = 1.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    barostat_interval = 25
    seed = 2025
    lamb = 0.0
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    host_config = None
    if host_name == "complex":
        with path_to_internal_file("tmd.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_config = builders.build_protein_system(
                str(protein_path), ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b], box_margin=0.1
            )
    elif host_name == "solvent":
        host_config = builders.build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b], box_margin=0.1)

    ligand_masses = st.combine_masses()
    x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb=0.0).astype(precision)
    if host_config is not None:
        host_config = setup_optimized_host(host_config, [mol_a, mol_b], ff)
        hgs = st.combine_with_host(host_config.host_system, lamb, host_config.num_water_atoms, host_config.omm_topology)

        x0 = np.concatenate([host_config.conf, x0])
        box = host_config.box
        masses = np.concatenate([host_config.masses, ligand_masses])
        u_fns = hgs.get_U_fns()
    else:
        masses = np.array(ligand_masses)
        u_fns = st.setup_intermediate_state(0.0).get_U_fns()
        box = np.eye(3, dtype=precision) * 10.0

    x0 = x0.astype(precision)
    v0 = np.zeros_like(x0)

    bps = []
    for bp in u_fns:
        bound_impl = bp.to_gpu(precision=precision).bound_impl
        bps.append(bound_impl)  # get the bound implementation

    bond_list = get_bond_list(get_bound_potential_by_type(u_fns, HarmonicBond).potential)
    group_idxs = get_group_indices(bond_list, len(masses))

    movers = []
    if host_config is not None:
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            barostat_interval,
            seed,
        )

        movers.append(baro.impl(bps, precision=precision))

    intg = LangevinIntegrator(temperature, dt, 1.0, np.asarray(masses), seed)

    num_steps = 200

    if precision == np.float32:
        ctxt = custom_ops.Context_f32(x0, v0, box, intg.impl(precision), bps, movers=movers)  # type: ignore
    else:
        ctxt = custom_ops.Context_f64(x0, v0, box, intg.impl(precision), bps, movers=movers)  # type: ignore
    xs, boxes = ctxt.multiple_steps(num_steps, num_steps // 10)

    return u_fns, xs, boxes


@pytest.fixture(
    scope="module",
    params=[
        (None, np.float32, 1e-4, 1e-5),
        ("solvent", np.float32, 1e-4, 1e-5),
        ("complex", np.float32, 1e-4, 1e-5),
        (None, np.float64, 1e-6, 1e-8),
        ("solvent", np.float64, 1e-6, 1e-8),
        ("complex", np.float64, 1e-6, 1e-8),
    ],
)
def pots_and_frames(request):
    host_name, precision, rtol, atol = request.param
    return host_name, precision, rtol, atol, get_potentials_and_frames(host_name, precision)


def test_potential_executor_execute(pots_and_frames):
    """Test PotentialExecutor's execute function. Intention is to parallelize a set of potentials from Python"""
    host_name, precision, rtol, atol, (bps, xs, boxes) = pots_and_frames
    ubps = []
    references_vals_by_pot = []
    for bp in bps:
        pot = bp.potential.to_gpu(precision=precision).unbound_impl
        ubps.append(pot)
        du_dx, du_dp, u = pot.execute_batch(xs, [bp.params.astype(precision)], boxes, True, True, True)
        assert du_dp is not None
        assert du_dx is not None
        assert u is not None

        # Toss out the parameters dimension
        references_vals_by_pot.append((du_dx.squeeze(axis=1), du_dp.squeeze(axis=1), u.squeeze(axis=1)))

    if precision == np.float32:
        potential_executor = custom_ops.PotentialExecutor_f32()
    else:
        potential_executor = custom_ops.PotentialExecutor_f64()

    pot_params = [bp.params.astype(precision) for bp in bps]

    for i, (x, b) in enumerate(zip(xs, boxes)):
        comp_du_dxs, comp_du_dps, comp_us = potential_executor.execute(ubps, x, pot_params, b)
        assert comp_du_dxs.shape == (len(ubps), *xs[0].shape)
        assert comp_us.size == len(ubps)
        assert len(comp_du_dps) == len(ubps)
        for j in range(len(ubps)):
            ref_du_dx, ref_du_dp, ref_u = references_vals_by_pot[j]
            np.testing.assert_array_equal(comp_us[j], ref_u[i])
            np.testing.assert_array_equal(comp_du_dps[j], ref_du_dp[i])
            np.testing.assert_array_equal(comp_du_dxs[j], ref_du_dx[i])

        for compute_du_dx, compute_du_dp, compute_u in itertools.product([False, True], repeat=3):
            if not (compute_u or compute_du_dp or compute_du_dx):
                continue
            select_du_dxs, select_du_dps, select_us = potential_executor.execute(
                ubps, x, pot_params, b, compute_u=compute_u, compute_du_dx=compute_du_dx, compute_du_dp=compute_du_dp
            )

            if compute_du_dx:
                np.testing.assert_allclose(select_du_dxs, comp_du_dxs, rtol=rtol, atol=atol)
            else:
                assert select_du_dxs is None
            if compute_du_dp:
                for j in range(len(ubps)):
                    np.testing.assert_allclose(select_du_dps[j], comp_du_dps[j], rtol=rtol, atol=atol)
            else:
                assert select_du_dps is None
            if compute_u:
                np.testing.assert_allclose(select_us, comp_us, rtol=rtol, atol=atol)
            else:
                assert select_us is None


def test_potential_executor_execute_batch(pots_and_frames):
    """Test PotentialExecutor's execute_batch function."""
    host_name, precision, rtol, atol, (bps, xs, boxes) = pots_and_frames
    rng = np.random.default_rng(2025)
    ubps = []
    pot_params = [bp.params.astype(precision) for bp in bps]

    n_params = 3
    parameter_sets = []
    for params in pot_params:
        new_params = [params]
        for i in range(n_params - 1):
            # Make very slight modifications to the params
            new_params.append(
                np.array(params, dtype=params.dtype) + rng.uniform(0.0, 1e-3, size=params.shape).astype(params.dtype)
            )
        parameter_sets.append(np.stack(new_params))

    references_vals_by_pot = []
    for i, bp in enumerate(bps):
        pot = bp.potential.to_gpu(precision=precision).unbound_impl
        ubps.append(pot)
        du_dx, du_dp, u = pot.execute_batch(xs, parameter_sets[i], boxes, True, True, True)
        assert du_dp is not None
        assert du_dx is not None
        assert u is not None

        references_vals_by_pot.append((du_dx, du_dp, u))

    if precision == np.float32:
        potential_executor = custom_ops.PotentialExecutor_f32()
    else:
        potential_executor = custom_ops.PotentialExecutor_f64()

    # Basic error checking

    # Verify that pot params have to have an extra dimension
    with pytest.raises(RuntimeError, match="number of parameter batches must match for each potential"):
        potential_executor.execute_batch(ubps, xs, pot_params, boxes)

    with pytest.raises(RuntimeError, match="number of potentials and the number of parameter sets must match"):
        potential_executor.execute_batch(ubps, xs, pot_params[:1], boxes)

    with pytest.raises(RuntimeError, match="params must have at least 2 dimensions"):
        potential_executor.execute_batch(ubps, xs, [np.zeros(1).astype(precision) for _ in pot_params], boxes)

    with pytest.raises(RuntimeError, match="must compute either du_dx, du_dp or energy"):
        potential_executor.execute_batch(
            ubps, xs, [np.zeros(1).astype(precision) for _ in pot_params], boxes, False, False, False
        )

    comp_du_dxs, comp_du_dps, comp_us = potential_executor.execute_batch(ubps, xs, parameter_sets, boxes)
    assert comp_du_dxs.shape == (len(ubps), len(xs), n_params, *xs[0].shape)
    assert comp_us.shape == (len(ubps), len(xs), n_params)
    assert len(comp_du_dps) == len(ubps)

    for i in range(len(ubps)):
        ref_du_dx, ref_du_dp, ref_u = references_vals_by_pot[i]
        np.testing.assert_array_equal(comp_us[i], ref_u)
        np.testing.assert_array_equal(comp_du_dxs[i], ref_du_dx)
        np.testing.assert_array_equal(comp_du_dps[i], ref_du_dp)

    comp_du_dxs, comp_du_dps, comp_us = potential_executor.execute_batch(ubps, xs, parameter_sets, boxes)
    assert comp_du_dxs.shape == (len(ubps), len(xs), n_params, *xs[0].shape)
    assert comp_us.shape == (len(ubps), len(xs), n_params)
    assert len(comp_du_dps) == len(ubps)

    for i in range(len(ubps)):
        ref_du_dx, ref_du_dp, ref_u = references_vals_by_pot[i]

        np.testing.assert_array_equal(comp_us[i], ref_u)
        np.testing.assert_array_equal(comp_du_dxs[i], ref_du_dx)
        np.testing.assert_array_equal(comp_du_dps[i], ref_du_dp)

    for compute_du_dx, compute_du_dp, compute_u in itertools.product([False, True], repeat=3):
        if not (compute_u or compute_du_dp or compute_du_dx):
            continue
        select_du_dxs, select_du_dps, select_us = potential_executor.execute_batch(
            ubps,
            xs,
            parameter_sets,
            boxes,
            compute_u=compute_u,
            compute_du_dx=compute_du_dx,
            compute_du_dp=compute_du_dp,
        )
        if compute_du_dx:
            np.testing.assert_allclose(select_du_dxs, comp_du_dxs, rtol=rtol, atol=atol)
        else:
            assert select_du_dxs is None
        if compute_du_dp:
            for i in range(len(ubps)):
                np.testing.assert_allclose(select_du_dps[i], comp_du_dps[i], rtol=rtol, atol=atol)
        else:
            assert select_du_dps is None
        if compute_u:
            np.testing.assert_allclose(select_us, comp_us, rtol=rtol, atol=atol)
        else:
            assert select_us is None


def test_potential_executor_execute_batch_sparse(pots_and_frames):
    """Test PotentialExecutor's execute_batch_sparse function."""
    host_name, precision, rtol, atol, (bps, xs, boxes) = pots_and_frames
    rng = np.random.default_rng(2025)
    ubps = []
    pot_params = [bp.params.astype(precision) for bp in bps]

    n_params = 3
    parameter_sets = []
    for params in pot_params:
        new_params = [params]
        for i in range(n_params - 1):
            new_params.append(
                np.array(params, dtype=params.dtype) + rng.uniform(0.0, 1e-3, size=params.shape).astype(params.dtype)
            )
        parameter_sets.append(np.stack(new_params))

    coords_batch_idxs = rng.choice(np.arange(len(xs), dtype=np.uint32), size=len(xs) // 2, replace=False)
    params_batch_idxs = rng.choice(np.arange(n_params, dtype=np.uint32), size=len(coords_batch_idxs))

    references_vals_by_pot = []
    for i, bp in enumerate(bps):
        pot = bp.potential.to_gpu(precision=precision).unbound_impl
        ubps.append(pot)
        du_dx, du_dp, u = pot.execute_batch_sparse(
            xs, parameter_sets[i], boxes, coords_batch_idxs, params_batch_idxs, True, True, True
        )
        assert du_dp is not None
        assert du_dx is not None
        assert u is not None

        references_vals_by_pot.append((du_dx, du_dp, u))

    if precision == np.float32:
        potential_executor = custom_ops.PotentialExecutor_f32()
    else:
        potential_executor = custom_ops.PotentialExecutor_f64()

    # Basic error checking

    # Verify that pot params have to have an extra dimension
    with pytest.raises(RuntimeError, match="number of parameter batches must match for each potential"):
        potential_executor.execute_batch_sparse(ubps, xs, pot_params, boxes, coords_batch_idxs, params_batch_idxs)

    with pytest.raises(RuntimeError, match="number of potentials and the number of parameter sets must match"):
        potential_executor.execute_batch_sparse(ubps, xs, pot_params[:1], boxes, coords_batch_idxs, params_batch_idxs)

    with pytest.raises(RuntimeError, match="params must have at least 2 dimensions"):
        potential_executor.execute_batch_sparse(
            ubps, xs, [np.zeros(1).astype(precision) for _ in pot_params], boxes, coords_batch_idxs, params_batch_idxs
        )

    with pytest.raises(RuntimeError, match="must compute either du_dx, du_dp or energy"):
        potential_executor.execute_batch_sparse(
            ubps,
            xs,
            [np.zeros(1).astype(precision) for _ in pot_params],
            boxes,
            coords_batch_idxs,
            params_batch_idxs,
            False,
            False,
            False,
        )

    with pytest.raises(RuntimeError, match="coords_batch_idxs and params_batch_idxs must have the same length"):
        potential_executor.execute_batch_sparse(
            ubps, xs, parameter_sets, boxes, coords_batch_idxs, params_batch_idxs[:1]
        )

    comp_du_dxs, comp_du_dps, comp_us = potential_executor.execute_batch_sparse(
        ubps, xs, parameter_sets, boxes, coords_batch_idxs, params_batch_idxs
    )
    assert comp_du_dxs.shape == (len(ubps), len(coords_batch_idxs), *xs[0].shape)
    assert comp_us.shape == (len(ubps), len(coords_batch_idxs))
    assert len(comp_du_dps) == len(ubps)

    for i in range(len(ubps)):
        ref_du_dx, ref_du_dp, ref_u = references_vals_by_pot[i]
        np.testing.assert_array_equal(comp_us[i], ref_u)
        np.testing.assert_array_equal(comp_du_dxs[i], ref_du_dx)
        np.testing.assert_array_equal(comp_du_dps[i], ref_du_dp)

    comp_du_dxs, comp_du_dps, comp_us = potential_executor.execute_batch_sparse(
        ubps, xs, parameter_sets, boxes, coords_batch_idxs, params_batch_idxs
    )
    assert comp_du_dxs.shape == (len(ubps), len(coords_batch_idxs), *xs[0].shape)
    assert comp_us.shape == (len(ubps), len(coords_batch_idxs))
    assert len(comp_du_dps) == len(ubps)

    for i in range(len(ubps)):
        ref_du_dx, ref_du_dp, ref_u = references_vals_by_pot[i]

        np.testing.assert_array_equal(comp_us[i], ref_u)
        np.testing.assert_array_equal(comp_du_dxs[i], ref_du_dx)
        np.testing.assert_array_equal(comp_du_dps[i], ref_du_dp)

    for compute_du_dx, compute_du_dp, compute_u in itertools.product([False, True], repeat=3):
        if not (compute_u or compute_du_dp or compute_du_dx):
            continue
        select_du_dxs, select_du_dps, select_us = potential_executor.execute_batch_sparse(
            ubps,
            xs,
            parameter_sets,
            boxes,
            coords_batch_idxs,
            params_batch_idxs,
            compute_u=compute_u,
            compute_du_dx=compute_du_dx,
            compute_du_dp=compute_du_dp,
        )
        if compute_du_dx:
            np.testing.assert_allclose(select_du_dxs, comp_du_dxs, rtol=rtol, atol=atol)
        else:
            assert select_du_dxs is None
        if compute_du_dp:
            for i in range(len(ubps)):
                np.testing.assert_allclose(select_du_dps[i], comp_du_dps[i], rtol=rtol, atol=atol)
        else:
            assert select_du_dps is None
        if compute_u:
            np.testing.assert_allclose(select_us, comp_us, rtol=rtol, atol=atol)
        else:
            assert select_us is None
