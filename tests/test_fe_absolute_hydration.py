# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025-2026, Forrest York
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

from dataclasses import replace

import numpy as np
import pytest
from jax import jit, value_and_grad
from jax import numpy as jnp
from openeye.oechem import OEChemIsLicensed
from scipy.optimize import minimize

from tmd import testsystems
from tmd.constants import BOLTZ, KCAL_TO_KJ
from tmd.fe import absolute_hydration
from tmd.fe.free_energy import LocalMDParams, MDParams
from tmd.fe.reweighting import one_sided_exp
from tmd.ff import Forcefield
from tmd.potentials.nonbonded import coulomb_interaction_group_energy, coulomb_prefactors_on_traj
from tmd.potentials.potential import get_bound_potential_by_type
from tmd.potentials.potentials import Nonbonded
from tmd.testsystems import fetch_freesolv


def test_run_solvent_absolute_hydration():
    seed = 2022
    n_frames = 10
    n_eq_steps = 100
    n_windows = 8
    steps_per_frame = 10
    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    md_params = MDParams(seed=seed, n_eq_steps=n_eq_steps, n_frames=n_frames, steps_per_frame=steps_per_frame)
    res, host_config = absolute_hydration.run_solvent(mol, ff, None, md_params=md_params, n_windows=n_windows)

    assert res.plots.overlap_summary_png is not None
    assert np.linalg.norm(res.final_result.dG_errs) < 20.0
    assert len(res.frames) == n_windows
    assert len(res.boxes) == n_windows
    assert len(res.frames[0]) == n_frames
    assert len(res.frames[-1]) == n_frames
    assert len(res.boxes[0]) == n_frames
    assert len(res.boxes[-1]) == n_frames
    assert res.md_params == md_params
    assert host_config.omm_topology is not None
    # The number of waters in the system should stay constant
    assert host_config.num_water_atoms == 6282
    assert host_config.conf.shape == (res.frames[0][0].shape[0] - mol.GetNumAtoms(), 3)
    assert host_config.box.shape == (3, 3)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "forcefield",
    [
        pytest.param(
            "smirnoff_2_2_1_ccc.py", marks=pytest.mark.skipif(not OEChemIsLicensed(), reason="Need OEChem for ccc FF")
        ),
        "smirnoff_2_0_0_amber_am1ccc.py",
    ],
)
def test_fit_solvent_absolute_hydration_ccc_params(forcefield):
    seed = 2026
    n_frames = 500
    n_eq_steps = 10000
    n_windows = 6
    steps_per_frame = 400

    mols = fetch_freesolv()
    mol = mols[14]
    ff = Forcefield.load_from_file(forcefield)

    ref_bcc_params = ff.q_handle.params
    assert len(ref_bcc_params) > 0
    md_params = MDParams(
        seed=seed,
        n_eq_steps=n_eq_steps,
        n_frames=n_frames,
        steps_per_frame=steps_per_frame,
        local_md_params=LocalMDParams(local_steps=390, min_radius=1.5, max_radius=1.5, k=10_000.0),
    )
    res, host_config = absolute_hydration.run_solvent(mol, ff, None, md_params=md_params, n_windows=n_windows)

    assert len(res.frames) == n_windows
    assert len(res.boxes) == n_windows
    assert len(res.frames[0]) == n_frames
    assert len(res.frames[-1]) == n_frames
    assert len(res.boxes[0]) == n_frames
    assert len(res.boxes[-1]) == n_frames
    assert res.md_params == md_params
    assert host_config.omm_topology is not None
    # The number of waters in the system should stay constant
    assert host_config.num_water_atoms == 6282
    assert host_config.conf.shape == (res.frames[0][0].shape[0] - mol.GetNumAtoms(), 3)
    assert host_config.box.shape == (3, 3)

    env_idxs = np.arange(host_config.conf.shape[0])
    lig_idxs = np.arange(mol.GetNumAtoms()) + host_config.conf.shape[0]

    nb_bp = get_bound_potential_by_type(res.final_result.initial_states[-1].potentials, Nonbonded)
    nb_pot = nb_bp.potential

    charges, _, _, w = nb_bp.params.T

    q_prefactors = coulomb_prefactors_on_traj(
        np.array(res.trajectories[-1].frames),
        np.array(res.trajectories[-1].boxes),
        charges,
        lig_idxs,
        env_idxs,
        beta=nb_pot.beta,
        cutoff=nb_pot.cutoff,
    )

    initial_ligand_charges = charges[lig_idxs]

    kBT = BOLTZ * res.final_result.initial_states[-1].integrator.temperature

    def u_charge(q_ligand):
        return coulomb_interaction_group_energy(q_ligand, q_prefactors)

    def make_reweighter_q(u_batch_fxn_q):
        u_0 = u_batch_fxn_q(initial_ligand_charges)

        def reweight(q_ligand):
            delta_us = (u_batch_fxn_q(q_ligand) - u_0) / kBT
            return one_sided_exp(delta_us)

        return reweight

    reweight_q = jit(make_reweighter_q(u_charge))
    np.testing.assert_allclose(reweight_q(initial_ligand_charges), 0.0, atol=1e-6)
    exp_dg = float(mol.GetProp("dG")) * KCAL_TO_KJ
    pred_dg = np.sum(res.final_result.dGs)

    max_charge_change = 0.1

    def loss_fxn(bcc_params):
        new_charges = ff.q_handle.partial_parameterize(bcc_params, mol)
        delta_dg_term = jnp.abs(exp_dg - (pred_dg + (reweight_q(new_charges) * kBT))) ** 2
        # Total difference in per atom charge should stay similar

        delta = jnp.abs(new_charges - initial_ligand_charges)
        difference_in_charge = jnp.sum((delta > max_charge_change) * ((delta - max_charge_change) ** 2))

        return delta_dg_term * 100.0 + difference_in_charge * 100.0

    @jit
    def fun(charge_params):
        v, g = value_and_grad(loss_fxn)(charge_params)
        return v.astype(float), jnp.asarray(g)

    # minimization successful
    result = minimize(fun, ref_bcc_params, jac=True, tol=0, options={"disp": True})

    adjusted_ccc_params = result.x
    assert np.any(adjusted_ccc_params != ff.q_handle.params)

    # Use the updated CCC charges for intermolecular
    ff = replace(ff, q_handle=type(ff.q_handle)(ff.q_handle.smirks, adjusted_ccc_params, None))
    md_params = replace(md_params, seed=md_params.seed * 2)
    comp_res, host_config = absolute_hydration.run_solvent(mol, ff, None, md_params=md_params, n_windows=n_windows)

    new_pred = np.sum(comp_res.final_result.dGs)
    print("Old prediction", pred_dg / KCAL_TO_KJ, "New", new_pred / KCAL_TO_KJ, "Exp", exp_dg / KCAL_TO_KJ)
    # The new prediction should be closer
    assert np.abs(exp_dg - pred_dg) > np.abs(exp_dg - new_pred)
