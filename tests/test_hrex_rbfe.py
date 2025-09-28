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

from dataclasses import replace
from typing import Optional
from unittest.mock import patch
from warnings import catch_warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from psutil import Process
from scipy import stats

from tmd.constants import DEFAULT_TEMP
from tmd.fe.free_energy import (
    EarlyTerminationParams,
    HostConfig,
    HREXParams,
    HREXSimulationResult,
    LocalMDParams,
    MDParams,
    RESTParams,
    WaterSamplingParams,
    estimate_free_energy_bar,
    generate_pair_bar_ulkns,
    sample_with_context_iter,
)
from tmd.fe.plots import (
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from tmd.fe.rbfe import estimate_relative_free_energy_bisection_hrex
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

DEBUG = False


def get_hif2a_single_topology_leg(host_name: str | None):
    forcefield = Forcefield.load_default()
    host_config: Optional[HostConfig] = None

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    if host_name == "complex":
        with path_to_internal_file("tmd.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_config = builders.build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b], box_margin=0.1
            )
    elif host_name == "solvent":
        host_config = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b], box_margin=0.1)

    return mol_a, mol_b, core, forcefield, host_config


@pytest.fixture(
    scope="module",
    params=[
        None,
        pytest.param("solvent", marks=pytest.mark.nightly(reason="slow")),
        pytest.param("complex", marks=pytest.mark.nightly(reason="slow")),
    ],
)
def hif2a_single_topology_leg(request):
    host_name = request.param
    return host_name, get_hif2a_single_topology_leg(request.param)


@pytest.mark.parametrize("seed", [2024])
def test_hrex_rbfe_hif2a_water_sampling_warning(hif2a_single_topology_leg, seed):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg
    if host_name != "complex":
        pytest.skip("Only relevant in complex")
    md_params = MDParams(
        n_frames=2,
        n_eq_steps=100,
        steps_per_frame=10,
        seed=seed,
        hrex_params=HREXParams(n_frames_bisection=100),
        water_sampling_params=WaterSamplingParams(interval=400, n_proposals=1000) if host_name == "complex" else None,
    )
    # Warning will only be triggered if total steps per window is less than the water sampling interval
    assert md_params.n_frames * md_params.steps_per_frame < md_params.water_sampling_params.interval
    n_windows = 2

    with catch_warnings(record=True) as captured_warnings:
        estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            host_config,
            md_params,
            lambda_interval=(0.0, 0.15),
            n_windows=n_windows,
            min_cutoff=0.7,
        )
    # We have hundreds of warnings thrown by MBAR in this code, so got to sift through
    assert len(captured_warnings) >= 1

    assert any("Not running any water sampling" in str(warn.message) for warn in captured_warnings)


@pytest.mark.parametrize("local_steps_percentage", [0.0, 0.75])
@pytest.mark.parametrize("max_bisection_windows, target_overlap", [(5, None), (5, 0.667)])
@pytest.mark.parametrize("enable_rest", [False, True])
@pytest.mark.parametrize("seed", [2024])
def test_hrex_rbfe_hif2a(
    hif2a_single_topology_leg, seed, max_bisection_windows, target_overlap, enable_rest, local_steps_percentage
):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    assert 0.0 <= local_steps_percentage <= 1.0
    if host_name is None and local_steps_percentage > 0.0:
        pytest.skip("No local MD for vacuum")

    steps_per_frame = 400
    md_params = MDParams(
        n_frames=200,
        n_eq_steps=10_000,
        steps_per_frame=steps_per_frame,
        seed=seed,
        hrex_params=HREXParams(
            n_frames_bisection=100,
            optimize_target_overlap=target_overlap,
            rest_params=(
                RESTParams(max_temperature_scale=3.0, temperature_scale_interpolation="exponential")
                if enable_rest
                else None
            ),
        ),
        water_sampling_params=WaterSamplingParams(interval=steps_per_frame, n_proposals=1000)
        if host_name == "complex"
        else None,
        local_md_params=LocalMDParams(int(steps_per_frame * local_steps_percentage))
        if local_steps_percentage > 0 and host_name is not None
        else None,
    )

    rss_traj = []

    def sample_and_record_rss(*args, **kwargs):
        result = sample_with_context_iter(*args, **kwargs)
        rss_traj.append(Process().memory_info().rss)
        return result

    with patch("tmd.fe.free_energy.sample_with_context_iter", sample_and_record_rss):
        result = estimate_relative_free_energy_bisection_hrex(
            mol_a,
            mol_b,
            core,
            forcefield,
            host_config,
            md_params,
            prefix=host_name if host_name is not None else "vacuum",
            lambda_interval=(0.0, 0.15),
            n_windows=max_bisection_windows,
            min_cutoff=0.7 if host_name == "complex" else None,
        )

    final_windows = len(result.final_result.initial_states)
    # All of the lambda values of the initial states should be different
    assert len(set([s.lamb for s in result.final_result.initial_states])) == final_windows

    if md_params.hrex_params.optimize_target_overlap is not None:
        assert final_windows <= max_bisection_windows
    else:
        # min_overlap is None here, will reach the max number of windows
        assert final_windows == max_bisection_windows

    assert len(rss_traj) > final_windows * md_params.n_frames
    # Check that memory usage is not increasing
    rss_traj = rss_traj[10:]  # discard initial transients
    assert len(rss_traj)
    rss_diff_count = np.sum(np.diff(rss_traj) != 0)
    rss_increase_count = np.sum(np.diff(rss_traj) > 0)
    assert stats.binom.pmf(rss_increase_count, n=rss_diff_count, p=0.5) >= 0.001

    if DEBUG:
        plot_hrex_rbfe_hif2a(result)

    assert result.hrex_diagnostics.cumulative_swap_acceptance_rates.shape[1] == final_windows - 1

    # Swap acceptance rates for all neighboring pairs should be >~ 20%
    final_swap_acceptance_rates = result.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]
    assert np.all(final_swap_acceptance_rates > 0.2)

    # Expect some replicas to visit every state
    final_replica_state_counts = result.hrex_diagnostics.cumulative_replica_state_counts[-1]
    assert np.any(np.all(final_replica_state_counts > 0, axis=0))

    assert isinstance(result.hrex_diagnostics.relaxation_time, float)
    assert result.hrex_diagnostics.normalized_kl_divergence >= 0.0

    if host_name == "complex" and md_params.water_sampling_params is not None:
        assert result.water_sampling_diagnostics.proposals_by_state_by_iter.shape == (
            md_params.n_frames,
            final_windows,
            2,
        )
        if md_params.local_md_params is None:
            proposals_per_frame = (
                md_params.steps_per_frame // md_params.water_sampling_params.interval
            ) * md_params.water_sampling_params.n_proposals
            assert np.all(result.water_sampling_diagnostics.proposals_by_state_by_iter[:, :, 1] == proposals_per_frame)
        else:
            global_steps = md_params.steps_per_frame - md_params.local_md_params.local_steps
            # If the number of global steps is sufficient to trigger water sampling in all the frames, at least one iteration should
            # have proposals.
            if (global_steps / md_params.water_sampling_params.interval) * md_params.n_frames > 0:
                assert np.any(
                    result.water_sampling_diagnostics.proposals_by_state_by_iter[:, :, 1]
                    == md_params.water_sampling_params.n_proposals
                )

        assert np.all(result.water_sampling_diagnostics.proposals_by_state_by_iter[:, :, 0] >= 0)
        assert result.water_sampling_diagnostics.cumulative_proposals_by_state.shape == (final_windows, 2)
    else:
        assert result.water_sampling_diagnostics is None
    assert len(result.hrex_diagnostics.replica_idx_by_state_by_iter) == md_params.n_frames
    assert all(
        len(replica_idx_by_state) == final_windows
        for replica_idx_by_state in result.hrex_diagnostics.replica_idx_by_state_by_iter
    )

    # Initial permutation should be the identity
    np.testing.assert_array_equal(result.hrex_diagnostics.replica_idx_by_state_by_iter[0], np.arange(final_windows))

    # Check that we can extract replica trajectories
    n_atoms = result.final_result.initial_states[0].x0.shape[0]
    rng = np.random.default_rng(seed)
    n_atoms_subset = rng.choice(n_atoms) + 1  # in [1, n_atoms]
    atom_idxs = rng.choice(n_atoms, n_atoms_subset, replace=False)
    trajs_by_replica = result.extract_trajectories_by_replica(atom_idxs)
    assert trajs_by_replica.shape == (final_windows, md_params.n_frames, n_atoms_subset, 3)

    # Check that the frame-to-frame rmsd is lower for replica trajectories versus state trajectories
    def time_lagged_rmsd(traj):
        sds = jnp.sum(jnp.diff(traj, axis=0) ** 2, axis=(1, 2))
        return jnp.sqrt(jnp.mean(sds))

    # (states, frames)
    trajs_by_state = np.array(
        [[np.array(frame)[atom_idxs] for frame in state_traj.frames] for state_traj in result.trajectories]
    )

    replica_traj_rmsds = jax.vmap(time_lagged_rmsd)(trajs_by_replica)
    state_traj_rmsds = jax.vmap(time_lagged_rmsd)(trajs_by_state)

    # should have rmsd(replica trajectory) < rmsd(state trajectory) for all pairs (replica, state)
    assert np.max(replica_traj_rmsds) < np.min(state_traj_rmsds)

    # Check that we can extract ligand trajectories by replica
    ligand_trajs_by_replica = result.extract_ligand_trajectories_by_replica()
    n_ligand_atoms = len(result.final_result.initial_states[0].ligand_idxs)
    assert ligand_trajs_by_replica.shape == (final_windows, md_params.n_frames, n_ligand_atoms, 3)

    # Check plots were generated
    assert result.hrex_plots
    assert result.hrex_plots.transition_matrix_png
    assert result.hrex_plots.replica_state_distribution_heatmap_png

    # Verify that the Bar results match to the reference implementation
    samples = result.trajectories
    initial_states = result.final_result.initial_states
    unbound_impls = [p.potential.to_gpu(np.float32).unbound_impl for p in initial_states[0].potentials]
    temperature = initial_states[0].integrator.temperature
    comp_bar_results = result.final_result.bar_results
    # Compute the reference bar results
    assert len(samples) == len(samples)

    ref_bar_results = []
    # Generate reference pair bar estimates by re-computing the energies.
    for i, initial_states in enumerate(zip(initial_states, initial_states[1:])):
        trajs = [result.trajectories[i], result.trajectories[i + 1]]
        u_kln_by_component = generate_pair_bar_ulkns(initial_states, trajs, temperature, unbound_impls)

        ref_bar_results.append(
            estimate_free_energy_bar(
                u_kln_by_component.squeeze(axis=0).astype(np.float32), temperature, n_bootstrap=100
            )
        )
    for ref_res, comp_res in zip(ref_bar_results, comp_bar_results):
        np.testing.assert_array_equal(ref_res.u_kln_by_component, comp_res.u_kln_by_component)
        np.testing.assert_array_equal(ref_res.overlap, comp_res.overlap)
        np.testing.assert_array_equal(ref_res.dG_err_by_component, comp_res.dG_err_by_component)
        np.testing.assert_array_equal(ref_res.overlap_by_component, comp_res.overlap_by_component)


def plot_hrex_rbfe_hif2a(result: HREXSimulationResult):
    plot_hrex_swap_acceptance_rates_convergence(result.hrex_diagnostics.cumulative_swap_acceptance_rates)
    plot_hrex_transition_matrix(result.hrex_diagnostics.transition_matrix)
    plot_hrex_replica_state_distribution_heatmap(
        result.hrex_diagnostics.cumulative_replica_state_counts,
        [state.lamb for state in result.final_result.initial_states],
    )
    plt.show()


@pytest.mark.parametrize("seed", [2025])
@pytest.mark.parametrize("local_md", [True, False])
def test_hrex_rbfe_early_termination(hif2a_single_topology_leg, local_md: bool, seed):
    """Verify that early termination works correctly"""
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    if local_md and host_name is None:
        pytest.skip("No local MD for vacuum")

    early_termination_params = EarlyTerminationParams(prediction_delta=1e6)
    md_params = MDParams(
        n_frames=2000,
        n_eq_steps=10,
        steps_per_frame=200,
        seed=seed,
        local_md_params=LocalMDParams(local_steps=200) if local_md else None,
        hrex_params=HREXParams(
            n_frames_bisection=10,
            # Make the prediction delta very large, should terminate after getting num_samples
            early_termination_params=early_termination_params,
        ),
    )
    expected_frames = early_termination_params.num_samples * early_termination_params.interval

    res = estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        md_params,
        lambda_interval=(0.0, 0.1),
        n_windows=3,
    )
    for traj in res.trajectories:
        assert len(traj.boxes) == expected_frames


@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("local_md", [True, False])
def test_hrex_rbfe_reproducibility(hif2a_single_topology_leg, local_md: bool, seed):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    if local_md and host_name is None:
        pytest.skip("No local MD for vacuum")

    md_params = MDParams(
        n_frames=10,
        n_eq_steps=10,
        steps_per_frame=400,
        seed=seed,
        local_md_params=LocalMDParams(local_steps=200) if local_md else None,
        hrex_params=HREXParams(n_frames_bisection=1),
    )

    run = lambda seed: estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        replace(md_params, seed=seed),
        lambda_interval=(0.0, 0.1),
        n_windows=3,
    )

    res1 = run(seed)
    res2 = run(seed)
    np.testing.assert_equal(res1.frames, res2.frames)
    np.testing.assert_equal(res1.boxes, res2.boxes)

    res3 = run(seed + 1)

    assert not np.all(res1.frames == res3.frames)

    if host_config:
        # for vacuum leg, boxes are trivially identical
        # If all steps are local, may also be identical since barostat is effectively
        # disabled (unless equil steps is > 15)
        assert not np.all(np.array(res1.boxes) == np.array(res3.boxes))


@pytest.mark.parametrize("seed", [2023])
def test_hrex_rbfe_min_overlap_below_target_overlap(hif2a_single_topology_leg, seed):
    """Test setting the min overlap below target overlap and verify that the results are comparable"""
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg

    target_overlap = 0.667
    overlap_diff = 0.1

    md_params = MDParams(
        n_frames=100,
        n_eq_steps=10000,
        steps_per_frame=400,
        seed=seed,
        hrex_params=HREXParams(optimize_target_overlap=target_overlap),
    )

    ref_res = estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        replace(md_params, seed=seed),
        lambda_interval=(0.0, 0.35),
        min_overlap=target_overlap,
    )

    comp_res = estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        replace(md_params, seed=seed),
        lambda_interval=(0.0, 0.35),
        min_overlap=target_overlap - overlap_diff,
    )

    # Should have fewer intermediates thank to the lower min_overlap
    assert len(ref_res.intermediate_results) > len(comp_res.intermediate_results)
    ref_final_swap_acceptance_rates = ref_res.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]
    comp_final_swap_acceptance_rates = comp_res.hrex_diagnostics.cumulative_swap_acceptance_rates[-1]

    assert ref_final_swap_acceptance_rates.size == comp_final_swap_acceptance_rates.size
    # Accept 5% difference in overlaps, 2x for swaps
    tolerance = 0.05
    np.testing.assert_allclose(ref_final_swap_acceptance_rates, comp_final_swap_acceptance_rates, atol=tolerance * 2)
    # Verify that all swaps are greater than zero
    assert np.all(ref_final_swap_acceptance_rates > tolerance)
    assert np.all(comp_final_swap_acceptance_rates > tolerance)

    # Overlaps should be within 5% of the target overlap or higher than the target overlap (because final neighboring windows can have significantly higher overlap)
    assert np.all(np.array(ref_res.final_result.overlaps) >= target_overlap - tolerance)
    assert np.all(np.array(comp_res.final_result.overlaps) >= target_overlap - tolerance)


@pytest.mark.parametrize("seed", [2023])
@pytest.mark.parametrize("temperature", [DEFAULT_TEMP, 310.0])
def test_hrex_rbfe_adjust_temperature(hif2a_single_topology_leg, seed, temperature):
    host_name, (mol_a, mol_b, core, forcefield, host_config) = hif2a_single_topology_leg
    if host_name is not None:
        pytest.skip("Only runs vacuum legs")

    md_params = MDParams(
        n_frames=10,
        n_eq_steps=10,
        steps_per_frame=400,
        seed=seed,
        hrex_params=HREXParams(n_frames_bisection=1),
    )

    run = lambda seed: estimate_relative_free_energy_bisection_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        host_config,
        replace(md_params, seed=seed),
        lambda_interval=(0.0, 0.1),
        n_windows=3,
        temperature=temperature,
    )

    res = run(seed)
    assert np.isfinite(np.sum(res.final_result.dGs))
