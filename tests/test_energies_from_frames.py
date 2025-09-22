# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025 Forrest York
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
from common import fixed_overflowed

from tmd.constants import DEFAULT_PRESSURE, DEFAULT_TEMP
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from tmd.lib.fixed_point import fixed_to_float
from tmd.md import builders, minimizer
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.potentials import SummedPotential
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

pytestmark = [pytest.mark.memcheck]


@pytest.mark.parametrize(
    "precision,rtol,atol",
    [(np.float32, 1e-7, 1e-7)],
)
@pytest.mark.parametrize("seed", [1234, 2025, 2022, 2021, 814])
def test_recomputation_of_energies(precision, rtol, atol, seed):
    """Verify that recomputing the energies of frames that have already had energies computed
    before, will produce nearly identical energies.

    The expectation is that `pot.execute(x, params, box)` will produce deterministic outputs when the
    parameters are identical (ie: Permutation of the values for compute_du_dx, compute_u and compute_du_dp). When
    the parameters differ the energies should be very close. This is due to the fact that kernels are templated to
    the specific parameters and may differ due to different code paths.
    """
    dt = 1.5e-3
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    barostat_interval = 25
    proposals_per_move = 1000
    targeted_water_sampling_interval = 100
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    # build the protein system.
    with path_to_internal_file("tmd.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_config = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff, mols=[mol_a, mol_b])

    min_coords = minimizer.fire_minimize_host([mol_a, mol_b], host_config, ff)

    x0 = min_coords.astype(precision)
    v0 = np.zeros_like(x0).astype(precision)

    bond_list = get_bond_list(host_config.host_system.bond.potential)
    group_idxs = get_group_indices(bond_list, len(host_config.masses))
    water_idxs = [group for group in group_idxs if len(group) == 3]

    baro = MonteCarloBarostat(
        x0.shape[0],
        pressure,
        temperature,
        group_idxs,
        barostat_interval,
        seed,
    )

    nb = host_config.host_system.nonbonded_all_pairs

    # Select the protein as the target for targeted insertion
    radius = 1.0
    target_idxs = next(group for group in group_idxs if len(group) > 3)
    tibdem = custom_ops.TIBDExchangeMove_f32(
        x0.shape[0],
        target_idxs,
        water_idxs,
        nb.params.astype(precision),
        DEFAULT_TEMP,
        nb.potential.beta,
        nb.potential.cutoff,
        radius,
        seed,
        proposals_per_move,
        targeted_water_sampling_interval,
    )

    intg = LangevinIntegrator(temperature, dt, 1.0, np.array(host_config.masses), seed)

    host_U_fns = host_config.host_system.get_U_fns()
    host_params = [bp.params.astype(np.float32) for bp in host_U_fns]
    summed_pot = SummedPotential([bp.potential for bp in host_U_fns], host_params)
    bps = []
    ubps = []

    ref_pot = summed_pot.to_gpu(precision).bind_params_list(host_params).bound_impl
    for bp in host_U_fns:
        bound_impl = bp.to_gpu(precision=precision).bound_impl
        bps.append(bound_impl)  # get the bound implementation
        ubps.append(bound_impl.get_potential())  # Get unbound potential

    baro_impl = baro.impl(bps)
    num_steps = 200
    for movers in [None, [baro_impl], [tibdem], [baro_impl, tibdem]]:
        if movers is not None:
            for mover in movers:
                # Make sure we are actually running all of the movers
                mover.set_step(0)
                assert mover.get_interval() <= num_steps
        ctxt = custom_ops.Context_f32(x0, v0, host_config.box, intg.impl(), bps, movers=movers)
        xs, boxes = ctxt.multiple_steps(num_steps, 10)

        for x, b in zip(xs, boxes):
            ref_du_dx, ref_U = ref_pot.execute(x, b)
            minimizer.check_force_norm(-ref_du_dx)
            test_u = 0.0
            test_u_selective = 0.0
            test_U_fixed = np.uint64(0)
            for fn, unbound, bp in zip(host_U_fns, ubps, bps):
                U_fixed = bp.execute_fixed(x, b)
                assert not fixed_overflowed(U_fixed)
                test_U_fixed += U_fixed
                _, U = bp.execute(x, b)
                test_u += U
                _, _, U_selective = unbound.execute(x, fn.params.astype(precision), b, False, False, True)
                test_u_selective += U_selective
                # Verify that executing the potential twice produces identical results
                for combo in itertools.product([False, True], repeat=3):
                    compute_du_dx, compute_du_dp, compute_u = combo
                    ref_du_dx, ref_du_dp, ref_u = unbound.execute(
                        x, fn.params.astype(precision), b, compute_du_dx, compute_du_dp, compute_u
                    )
                    comp_du_dx, comp_du_dp, comp_u = unbound.execute(
                        x, fn.params.astype(precision), b, compute_du_dx, compute_du_dp, compute_u
                    )
                    if compute_du_dx:
                        np.testing.assert_array_equal(comp_du_dx, ref_du_dx)
                    else:
                        assert comp_du_dx is None and ref_du_dx is None
                    if compute_du_dp:
                        np.testing.assert_array_equal(comp_du_dp, ref_du_dp)
                    else:
                        assert comp_du_dp is None and ref_du_dp is None
                    if compute_u:
                        np.testing.assert_array_equal(comp_u, ref_u)
                    else:
                        assert comp_u is None and ref_u is None

            # cast to expected level of precision
            np.testing.assert_allclose(
                precision(test_u), precision(test_u_selective), rtol=rtol, atol=atol, err_msg=str(movers)
            )
            np.testing.assert_allclose(precision(ref_U), precision(fixed_to_float(test_U_fixed)), rtol=rtol, atol=atol)
            np.testing.assert_allclose(ref_U, test_u, rtol=rtol, atol=atol)
            np.testing.assert_allclose(ref_U, test_u_selective, rtol=rtol, atol=atol)
