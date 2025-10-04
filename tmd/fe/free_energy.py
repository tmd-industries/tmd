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

import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import cache
from typing import Callable, Optional
from warnings import warn

import jax
import numpy as np
from numpy.typing import NDArray

from tmd._vendored.pymbar.utils import kln_to_kn
from tmd.constants import BOLTZ
from tmd.fe import model_utils, topology
from tmd.fe.bar import (
    MBARConvergence,
    bar_with_pessimistic_uncertainty,
    construct_mbar_from_u_kln,
    df_and_err_from_mbar,
    pair_overlap_from_mbar,
    sanitize_energies_for_bar,
    works_from_ukln,
)
from tmd.fe.plots import (
    plot_as_png_fxn,
    plot_dG_errs_figure,
    plot_overlap_summary_figure,
)
from tmd.fe.protocol_refinement import greedy_bisection_step
from tmd.fe.rest.single_topology import InterpolationFxnName
from tmd.fe.stored_arrays import StoredArrays
from tmd.fe.utils import get_mol_masses, get_romol_conf
from tmd.ff import Forcefield, ForcefieldParams
from tmd.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from tmd.lib.custom_ops import BoundPotential_f32, Context_f32, SummedPotential_f32
from tmd.md.barostat.utils import compute_box_center, get_bond_list, get_group_indices
from tmd.md.builders import HostConfig
from tmd.md.exchange.exchange_mover import WaterSamplingDiagnostics, get_water_idxs
from tmd.md.hrex import HREX, HREXDiagnostics, ReplicaIdx, StateIdx, get_swap_attempts_per_iter_heuristic
from tmd.md.states import CoordsVelBox
from tmd.potentials import (
    BoundPotential,
    HarmonicBond,
    Nonbonded,
    Potential,
    make_summed_potential,
)
from tmd.potentials.potential import get_bound_potential_by_type
from tmd.utils import batches

WATER_SAMPLER_MOVERS = (
    custom_ops.TIBDExchangeMove_f32,
    custom_ops.TIBDExchangeMove_f64,
)


@dataclass(frozen=True)
class RESTParams:
    max_temperature_scale: float
    temperature_scale_interpolation: InterpolationFxnName


@dataclass(frozen=True)
class EarlyTerminationParams:
    """Parameters that define early termination of HREX.

    Parameters
    ----------

    prediction_delta: float
        The difference between past predictions before terminating. Units in kJ/mol

    num_samples: int
        The number of samples to collect before considering early termination. Must be at least 2

    interval: int
        How often to compute an estimate. Reducing this calls MBAR more often, which can slow down HREX. Must be at least 1.

    slope_threshold: float
        What percentage of the max possible slope implied by the prediction delta is tolerated. Must be within range (0, 1.0]
        See tmd.fe.bar.MBARConvergence for more details
    """

    prediction_delta: float
    num_samples: int = 10
    interval: int = 25
    slope_threshold: float = 0.75

    def __post_init__(self):
        assert self.prediction_delta > 0.0
        assert self.num_samples >= 2
        assert self.interval >= 1
        assert 0.0 < self.slope_threshold <= 1.0


@dataclass(frozen=True)
class HREXParams:
    """
    Parameters
    ----------

    n_frames_bisection: int
        Number of frames to sample using MD during the initial bisection phase used to determine lambda spacing

    n_frames_per_iter: int
        DEPRECATED, must be set to 1. Number of frames to sample using MD per HREX iteration.

    max_delta_states: int or None
        If given, number of neighbor states on either side of a given replica's initial state for which to compute
        potentials. This determines the maximum number of states that a replica can move from its initial state during
        a single HREX iteration. Otherwise, compute potentials for all (replica, state) pairs.

    optimize_target_overlap: float or None
        If given, optimize the lambda schedule out of the initial bisection phase to target a specific minimum overlap
        between all adjacent windows. Must be in the interval (0.0, 1.0) if provided.
    """

    n_frames_bisection: int = 100
    max_delta_states: int | None = 4
    optimize_target_overlap: float | None = None
    rest_params: RESTParams | None = None
    early_termination_params: EarlyTerminationParams | None = None

    def __post_init__(self):
        assert self.n_frames_bisection > 0
        assert self.max_delta_states is None or self.max_delta_states > 0
        assert self.optimize_target_overlap is None or 0.0 < self.optimize_target_overlap < 1.0


@dataclass(frozen=True)
class WaterSamplingParams:
    """
    Parameters
    ----------

    interval:
        How many steps of MD between water sampling moves

    n_proposals:
        Number of proposals per make.

    batch_size:
        Internal parameter detailing the parallelism of the mover, typically can be left at default.

    radius:
        Radius, in nanometers, from the centroid of the molecule to treat as the inner target volume
    """

    interval: int = 400
    n_proposals: int = 1000
    batch_size: int = 250
    radius: float = 1.0

    def __post_init__(self):
        assert self.interval > 0
        assert self.n_proposals > 0
        assert self.radius > 0.0
        assert self.batch_size > 0
        assert self.batch_size <= self.n_proposals


@dataclass(frozen=True)
class LocalMDParams:
    local_steps: int
    k: float = 1_000.0  # kJ/mol/nm^4
    min_radius: float = 1.0  # nm
    max_radius: float = 3.0  # nm
    freeze_reference: bool = True

    def __post_init__(self):
        assert 0.1 <= self.min_radius <= self.max_radius
        assert self.local_steps > 0
        assert 1.0 <= self.k <= 1.0e6


@dataclass(frozen=True)
class MDParams:
    n_frames: int
    n_eq_steps: int
    steps_per_frame: int
    seed: int

    # Set to LocalMDParams to enable local MD
    local_md_params: LocalMDParams | None = None
    # Set to HREXParams or None to disable HREX
    hrex_params: HREXParams | None = None
    # Setting water_sampling_params to None disables water sampling.
    water_sampling_params: WaterSamplingParams | None = None

    def __post_init__(self):
        assert self.steps_per_frame > 0
        assert self.n_frames > 0
        assert self.n_eq_steps >= 0
        if self.local_md_params is not None:
            assert self.local_md_params.local_steps <= self.steps_per_frame


@dataclass
class InitialState:
    """
    An initial contains everything that is needed to bitwise reproduce a trajectory given MDParams

    This object can be pickled safely.
    """

    potentials: list[BoundPotential]
    integrator: LangevinIntegrator
    barostat: Optional[MonteCarloBarostat]
    x0: NDArray
    v0: NDArray
    box0: NDArray
    lamb: float
    ligand_idxs: NDArray
    protein_idxs: NDArray
    # The atoms that are in the 4d plane defined by w_coord == 0.0
    interacting_atoms: Optional[NDArray] = None

    def __post_init__(self):
        assert self.ligand_idxs.dtype == np.int32 or self.ligand_idxs.dtype == np.int64
        assert self.protein_idxs.dtype == np.int32 or self.protein_idxs.dtype == np.int64

        self.x0 = self.x0.astype(dtype=np.float32)
        self.v0 = self.v0.astype(dtype=np.float32)
        self.box0 = self.box0.astype(dtype=np.float32)

        for p in self.potentials:
            p.params = p.params.astype(np.float32)

    def to_bound_impl(self):
        return make_summed_potential(self.potentials).to_gpu(np.float32).bound_impl


@dataclass
class BarResult:
    dG: float
    dG_err: float
    dG_err_by_component: NDArray  # (len(U_names),)
    overlap: float
    overlap_by_component: NDArray  # (len(U_names),)
    u_kln_by_component: NDArray  # (len(U_names), 2, 2, N)


@dataclass
class PairBarPlots:
    dG_errs_png: bytes
    overlap_summary_png: bytes


@dataclass
class HREXPlots:
    transition_matrix_png: bytes
    replica_state_distribution_heatmap_png: bytes


@dataclass
class PairBarResult:
    """Results of BAR analysis on L-1 adjacent pairs of states given a sequence of L states."""

    initial_states: list[InitialState]  # length L
    bar_results: list[BarResult]  # length L - 1

    def __post_init__(self):
        assert len(self.bar_results) == len(self.initial_states) - 1

    @property
    def dGs(self) -> list[float]:
        return [r.dG for r in self.bar_results]

    @property
    def dG_errs(self) -> list[float]:
        return [r.dG_err for r in self.bar_results]

    @property
    def dG_err_by_component_by_lambda(self) -> NDArray:
        return np.array([r.dG_err_by_component for r in self.bar_results])

    @property
    def overlaps(self) -> list[float]:
        return [r.overlap for r in self.bar_results]

    @property
    def overlap_by_component_by_lambda(self) -> NDArray:
        return np.array([r.overlap_by_component for r in self.bar_results])

    @property
    def u_kln_by_component_by_lambda(self) -> NDArray:
        return np.array([r.u_kln_by_component for r in self.bar_results])


@dataclass
class Trajectory:
    frames: StoredArrays  # (frame, atom, dim)
    boxes: list[NDArray]  # (frame, dim, dim)
    final_velocities: Optional[NDArray]  # (atom, dim)
    final_barostat_volume_scale_factor: Optional[float] = None

    def __post_init__(self):
        n_frames = len(self.frames)
        assert len(self.boxes) == n_frames
        if n_frames == 0:
            return
        n_atoms, n_dims = self.frames[0].shape
        assert self.boxes[0].shape == (n_dims, n_dims)
        if self.final_velocities is not None:
            assert self.final_velocities.shape == (n_atoms, n_dims)

    def extend(self, other: "Trajectory"):
        """Concatenate another trajectory to the end of this one"""
        self.frames.extend(other.frames)
        self.boxes.extend(other.boxes)
        self.final_velocities = other.final_velocities
        self.final_barostat_volume_scale_factor = other.final_barostat_volume_scale_factor

    @classmethod
    def empty(cls):
        return Trajectory(StoredArrays(), [], None, None)


@dataclass
class SimulationResult:
    final_result: PairBarResult
    plots: PairBarPlots
    trajectories: list[Trajectory]
    md_params: MDParams
    intermediate_results: list[PairBarResult]

    @property
    def frames(self) -> list[StoredArrays]:
        return [traj.frames for traj in self.trajectories]

    @property
    def boxes(self) -> list[NDArray]:
        return [np.array(traj.boxes) for traj in self.trajectories]

    def compute_u_kn(self) -> tuple[NDArray, NDArray]:
        """get MBAR input matrices u_kn and N_k"""

        return compute_u_kn(self.trajectories, self.final_result.initial_states)


@dataclass
class HREXSimulationResult(SimulationResult):
    hrex_diagnostics: HREXDiagnostics
    hrex_plots: HREXPlots
    water_sampling_diagnostics: WaterSamplingDiagnostics | None = None

    def extract_trajectories_by_replica(self, atom_idxs: NDArray) -> NDArray:
        """Returns an array of shape (n_replicas, n_frames, len(atom_idxs), 3) of trajectories for each replica

        Note: This consumes O(n_frames * len(atom_idxs)) memory, and thus may OOM for large systems if len(atom_idxs) is
        a significant fraction of the total number of atoms.

        Parameters
        ----------
        atom_idxs: NDArray
            Indices of atoms to extract
        """

        # (states, frames, atoms, 3)
        # NOTE: chunk[:, atom_idxs] below returns a copy (rather than a view) due to the use of "advanced indexing".
        # This is important because otherwise we would try to store all of the whole-system frames in memory at once.
        trajs_by_state = np.array(
            [
                np.concatenate([chunk[:, atom_idxs] for chunk in state_traj.frames._chunks()], axis=0)
                for state_traj in self.trajectories
            ]
        )

        replica_idx_by_iter_by_state = np.asarray(self.hrex_diagnostics.replica_idx_by_state_by_iter).T
        state_idx_by_iter_by_replica = np.argsort(replica_idx_by_iter_by_state, axis=0)

        # (replicas, frames, atoms, 3)
        trajs_by_replica = np.take_along_axis(trajs_by_state, state_idx_by_iter_by_replica[:, :, None, None], axis=0)

        return trajs_by_replica

    def extract_ligand_trajectories_by_replica(self):
        """Returns an array of shape (n_replicas, n_frames, n_ligand_atoms, 3) of ligand trajectories for each replica"""
        ligand_idxs = self.final_result.initial_states[0].ligand_idxs
        assert all(np.all(s.ligand_idxs == ligand_idxs) for s in self.final_result.initial_states)
        return self.extract_trajectories_by_replica(ligand_idxs)


def trajectories_by_replica_to_by_state(
    trajectory_by_iter_by_replica: NDArray,
    replica_idx_by_state_by_iter: Sequence[Sequence[ReplicaIdx]],
) -> NDArray:
    """Utility function to convert the output of `extract_trajectories_by_replica` from (replica, iters, ...) to
    (state, iters, ...). This is useful for evaluating the trajectories of states.
    """
    assert len(trajectory_by_iter_by_replica.shape) == 4
    replica_idx_by_iter_by_state = np.asarray(replica_idx_by_state_by_iter).T
    assert replica_idx_by_iter_by_state.shape == trajectory_by_iter_by_replica.shape[:2]

    trajectory_by_iter_by_state = np.take_along_axis(
        trajectory_by_iter_by_replica, replica_idx_by_iter_by_state[:, :, None, None], axis=0
    )

    return trajectory_by_iter_by_state


def image_frames(initial_state: InitialState, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Images a sequence of frames within the periodic box given an Initial state. Recenters the simulation around the
    centroid of the coordinates specified by initial_state.ligand_idxs prior to imaging.

    Calling this function on a sequence of frames will NOT produce identical energies/du_dp/du_dx. Should only be used
    for visualization convenience.

    Parameters
    ----------

    initial_state: InitialState
        State that the frames came from

    frames: sequence of np.ndarray of coordinates
        Coordinates to image, sequence of K arrays with shape (N, 3)

    boxes: list of boxes
        Boxes to image coordinates into, list of K arrays with shape (3, 3)

    Returns
    -------
        imaged_coordinates
    """
    assert np.array(boxes).shape[1:] == (3, 3), "Boxes are not 3x3"
    assert len(frames) == len(boxes), "Number of frames and boxes don't match"

    hb_potential = get_bound_potential_by_type(initial_state.potentials, HarmonicBond).potential
    group_indices = get_group_indices(get_bond_list(hb_potential), len(initial_state.integrator.masses))
    imaged_frames = np.empty_like(frames)
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        assert frame.ndim == 2 and frame.shape[-1] == 3, "frames must have shape (N, 3)"
        # Recenter the frame around the centroid of the ligand
        ligand_centroid = np.mean(frame[initial_state.ligand_idxs], axis=0)
        center = compute_box_center(box)
        offset = ligand_centroid + center
        centered_frames = frame - offset

        imaged_frames[i] = model_utils.image_frame(group_indices, centered_frames, box)
    return np.array(imaged_frames)


class BaseFreeEnergy:
    @staticmethod
    def _get_system_params_and_potentials(ff_params: ForcefieldParams, topology, lamb: float):
        params_potential_pairs = [
            topology.parameterize_harmonic_bond(ff_params.hb_params),
            topology.parameterize_harmonic_angle(ff_params.ha_params),
            topology.parameterize_proper_torsion(ff_params.pt_params),
            topology.parameterize_improper_torsion(ff_params.it_params),
            topology.parameterize_nonbonded_pairlist(
                ff_params.q_params,
                ff_params.q_params_intra,
                ff_params.lj_params,
                ff_params.lj_params_intra,
            ),
            topology.parameterize_nonbonded(
                ff_params.q_params,
                ff_params.q_params_intra,
                ff_params.lj_params,
                ff_params.lj_params_intra,
                lamb,
            ),
        ]

        params, potentials = zip(*params_potential_pairs)
        return params, potentials


# this class is serializable.
class AbsoluteFreeEnergy(BaseFreeEnergy):
    def __init__(self, mol, top):
        """
        Compute the absolute free energy of a molecule via 4D decoupling.

        Parameters
        ----------
        mol: rdkit mol
            Ligand to be decoupled

        top: Topology
            topology.Topology to use

        """
        self.mol = mol
        self.top = top

    def prepare_host_edge(
        self, ff: Forcefield, host_config: HostConfig, lamb: float
    ) -> tuple[tuple[Potential, ...], tuple, NDArray]:
        """
        Prepares the host-guest system

        Parameters
        ----------
        ff: Forcefield
            forcefield to use

        host_config: HostConfig
            HostConfig containing openmm System object to be deserialized.

        lamb: float
            alchemical parameter controlling 4D decoupling

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ligand_masses = get_mol_masses(self.mol)
        ff_params = ff.get_params()
        hgt = topology.HostGuestTopology(
            host_config.host_system.get_U_fns(), self.top, host_config.num_water_atoms, ff, host_config.omm_topology
        )

        combined_params, combined_potentials = self._get_system_params_and_potentials(ff_params, hgt, lamb)
        combined_masses = self._combine(ligand_masses, np.array(host_config.masses))
        return combined_potentials, combined_params, combined_masses

    def prepare_vacuum_edge(self, ff: Forcefield) -> tuple[tuple[Potential, ...], tuple, NDArray]:
        """
        Prepares the vacuum system

        Parameters
        ----------
        ff: Forcefield
            forcefield to use

        Returns
        -------
        3-tuple
            unbound_potentials, system_params, combined_masses

        """
        ff_params = ff.get_params()
        ligand_masses = get_mol_masses(self.mol)
        params, pots = self._get_system_params_and_potentials(ff_params, self.top, 0.0)
        final_pots = []
        final_params = []
        for pot, param in zip(pots, params):
            # Drop the Nonbonded potential in the vacuum case. Makes vacuum significantly slower than it needs to be
            if isinstance(pot, Nonbonded):
                continue
            final_pots.append(pot)
            final_params.append(param)
        return tuple(final_pots), tuple(final_params), ligand_masses

    def prepare_combined_coords(self, host_coords: Optional[NDArray] = None) -> NDArray:
        """
        Returns the combined coordinates.

        Parameters
        ----------
        host_coords: np.array
            Nx3 array of atomic coordinates
            If None, return just the ligand coordinates.

        Returns
        -------
            combined_coordinates
        """
        ligand_coords = get_romol_conf(self.mol)
        return self._combine(ligand_coords, host_coords)

    def _combine(self, ligand_values: NDArray, host_values: Optional[NDArray] = None) -> NDArray:
        """
        Combine the values along the 0th axis.
        The host values will be first, if given.
        Then ligand values.

        Parameters
        ----------
        ligand_values: np.array
        host_values: Optional[np.array]

        Returns
        -------
            combined_values
        """
        if host_values is None:
            return ligand_values
        return np.concatenate([host_values, ligand_values])


def get_water_sampler_params(initial_state: InitialState) -> NDArray:
    """Given an initial state, return a copy of the parameters that define the nonbonded parameters of water with respect to the
    entire system.

    Water Interactions (with other Waters, Ligand, Protein) need to be consistent with the MD potentials.

    This is only valid if the Nonbonded potential contains all host-host and host-guest interactions. No longer
    necessarily valid if we add back in a NonbondedInteractionGroup
    """
    nb_ixn_pot = get_bound_potential_by_type(initial_state.potentials, Nonbonded)
    water_sampler_params = np.array(nb_ixn_pot.params)

    assert water_sampler_params.shape[1] == 4
    return water_sampler_params


def get_summed_potential_from_bps(bps: Sequence[BoundPotential_f32]) -> SummedPotential_f32:
    return SummedPotential_f32([bp.get_potential() for bp in bps], [bp.size() for bp in bps])


def get_context(initial_state: InitialState, md_params: Optional[MDParams] = None) -> Context_f32:
    """
    Construct a Context from the potentials defined by the initial state
    """

    bound_impls = [bp.to_gpu(np.float32).bound_impl for bp in initial_state.potentials]
    intg_impl = initial_state.integrator.impl()
    movers = []
    if initial_state.barostat:
        movers.append(initial_state.barostat.impl(bound_impls))
    if md_params is not None and md_params.water_sampling_params is not None:
        # Setup the water indices
        hb_potential = get_bound_potential_by_type(initial_state.potentials, HarmonicBond).potential
        group_indices = get_group_indices(get_bond_list(hb_potential), len(initial_state.integrator.masses))

        water_idxs = get_water_idxs(group_indices, ligand_idxs=initial_state.ligand_idxs)

        # Select a Nonbonded Potential to get the the cutoff/beta, assumes all have same cutoff/beta.
        nb = get_bound_potential_by_type(initial_state.potentials, Nonbonded).potential

        water_params = get_water_sampler_params(initial_state)

        # Generate a new random seed based on the integrator seed, MDParams seed is constant across states
        rng = np.random.default_rng(initial_state.integrator.seed)
        water_sampler_seed = rng.integers(np.iinfo(np.int32).max)

        water_sampler = custom_ops.TIBDExchangeMove_f32(
            initial_state.x0.shape[0],
            initial_state.ligand_idxs.tolist(),  # type: ignore
            water_idxs,
            water_params,
            initial_state.integrator.temperature,
            nb.beta,
            nb.cutoff,
            md_params.water_sampling_params.radius,
            water_sampler_seed,
            md_params.water_sampling_params.n_proposals,
            md_params.water_sampling_params.interval,
            batch_size=md_params.water_sampling_params.batch_size,
        )
        movers.append(water_sampler)

    return Context_f32(initial_state.x0, initial_state.v0, initial_state.box0, intg_impl, bound_impls, movers=movers)


def sample_with_context_iter(
    ctxt: Context_f32, md_params: MDParams, temperature: float, ligand_idxs: NDArray, batch_size: int
) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
    """Sample a context using MDParams returning batches of frames up to `batch_size`. All results are returned
    as numpy arrays that are in memory, and it is left to the user to act accordingly.

    For getting a Trajectory object that stores the frames to disk, refer to `sample_with_context`.

    Parameters
    ----------
    ctxt: Context_f32
        The context to use to generate samples

    md_params: MDParams
        The parameters that define the sampling of frames from the context

    temperature: float
        The temperature, in kelvin, used when running Local MD moves

    ligand_idxs: np.ndarray
        Array representing the indices of atoms that make up the ligand, determines the atoms considered as the center
        of local MD.

    batch_size: int
        The most number of frames (coords and boxes) that will be kept in memory at one time.

    Returns
    -------
    Iterator of 3-tuples
        coords, boxes, final_velocities

    Notes
    -----
    * If md_params.n_eq_steps is greater than 0, the barostat will be set to run every 15 steps regardless of what
      the context defined. Will be reset to the original interval for production steps.
    """
    rng = np.random.default_rng(md_params.seed)

    if md_params.local_md_params is not None:
        ctxt.setup_local_md(temperature, md_params.local_md_params.freeze_reference)

    def run_global_steps(n_steps: int, store_frames: bool) -> tuple[NDArray, NDArray, NDArray]:
        coords, boxes = ctxt.multiple_steps(
            n_steps=n_steps,
            store_x_interval=md_params.steps_per_frame if store_frames else 0,
        )
        final_velocities = ctxt.get_v_t()

        return coords, boxes, final_velocities

    def run_local_steps(n_steps: int, store_frames: bool) -> tuple[NDArray, NDArray, NDArray]:
        coords = []
        boxes = []
        assert md_params.local_md_params is not None
        for steps in batches(n_steps, md_params.steps_per_frame):
            if steps < md_params.steps_per_frame:
                warn(
                    f"Batch of sample has {steps} steps, less than batch size {md_params.steps_per_frame}. Setting to {md_params.steps_per_frame}"
                )
                steps = md_params.steps_per_frame
            local_steps = md_params.local_md_params.local_steps
            global_steps = steps - local_steps
            if global_steps > 0:
                ctxt.multiple_steps(n_steps=global_steps)
            x_t, box_t = ctxt.multiple_steps_local(
                local_steps,
                ligand_idxs.astype(np.int32),
                k=md_params.local_md_params.k,
                radius=rng.uniform(md_params.local_md_params.min_radius, md_params.local_md_params.max_radius),
                seed=rng.integers(np.iinfo(np.int32).max),
            )
            if store_frames:
                coords.append(x_t[-1])
                boxes.append(box_t[-1])

        final_velocities = ctxt.get_v_t()

        return np.asarray(coords), np.asarray(boxes), final_velocities

    steps_func = run_global_steps
    if md_params.local_md_params is not None:
        steps_func = run_local_steps

    # burn-in
    if md_params.n_eq_steps:
        # Set barostat interval to 15 for equilibration, then back to the original interval for production
        # If running local MD, the barostat is disabled at the C++ level
        barostat = ctxt.get_barostat()
        original_interval = 0 if barostat is None else barostat.get_interval()
        equil_barostat_interval = 15
        if barostat is not None:
            barostat.set_interval(equil_barostat_interval)
        # Don't store the frames, since the equilibration frames are unused
        steps_func(md_params.n_eq_steps, False)
        if barostat is not None:
            barostat.set_interval(original_interval)

    assert np.all(np.isfinite(ctxt.get_x_t())), "Equilibration resulted in a nan"

    for n_frames in batches(md_params.n_frames, batch_size):
        yield steps_func(n_frames * md_params.steps_per_frame, True)


def sample_with_context(
    ctxt: Context_f32, md_params: MDParams, temperature: float, ligand_idxs: NDArray, max_buffer_frames: int
) -> Trajectory:
    """Wrapper for `sample_with_context_iter` that stores the frames to disk and returns a Trajectory result.
    Stores up to `max_buffer_frames` frames in memory before writing to disk.

    Refer to `sample_with_context_iter` for parameter documentation
    """
    all_coords = StoredArrays()
    all_boxes: list[NDArray] = []
    final_velocities: NDArray = None  # type: ignore # work around "possibly unbound" error
    for batch_coords, batch_boxes, final_velocities in sample_with_context_iter(
        ctxt, md_params, temperature, ligand_idxs, max_buffer_frames
    ):
        all_coords.extend(batch_coords)
        all_boxes.extend(batch_boxes)

    assert len(all_coords) == md_params.n_frames
    assert len(all_boxes) == md_params.n_frames

    assert np.all(np.isfinite(all_coords[-1])), "Production resulted in a nan"

    final_barostat_volume_scale_factor = ctxt.get_barostat().get_volume_scale_factor() if ctxt.get_barostat() else None

    return Trajectory(all_coords, all_boxes, final_velocities, final_barostat_volume_scale_factor)


def sample(initial_state: InitialState, md_params: MDParams, max_buffer_frames: int) -> Trajectory:
    """Generate a trajectory given an initial state and a simulation protocol

    Parameters
    ----------
    initial_state: InitialState
        (contains potentials, integrator, optional barostat)

    md_params: MDParams
        MD parameters

    max_buffer_frames: int
        number of frames to store in memory before dumping to disk

    Returns
    -------
    Trajectory

    Notes
    -----
    * Assertion error if coords become NaN
    """

    ctxt = get_context(initial_state, md_params)

    return sample_with_context(
        ctxt, md_params, initial_state.integrator.temperature, initial_state.ligand_idxs, max_buffer_frames
    )


def estimate_free_energy_bar(u_kln_by_component: NDArray, temperature: float, n_bootstrap: int = 0) -> BarResult:
    """
    Estimate free energy difference for a pair of states given pre-generated samples.

    Parameters
    ----------
    u_kln_by_component: array
        u_kln in pymbar format (k = l = 2) for each energy component

    temperature: float
        Temperature

    n_bootstrap: int
        Number of bootstrap iterations for Bar error estimation.
        If n_bootstrap == 0, will return Bar uncertainty without bootstrapping.

    Return
    ------
    PairBarResult
        results from BAR computation

    """
    assert n_bootstrap >= 0

    u_kln_by_component = sanitize_energies_for_bar(u_kln_by_component)

    u_kln = u_kln_by_component.sum(0)

    mbar = construct_mbar_from_u_kln(u_kln)

    if n_bootstrap > 0:
        df, df_err = bar_with_pessimistic_uncertainty(u_kln)  # reduced units
    else:
        df, df_err = df_and_err_from_mbar(mbar)

    kBT = BOLTZ * temperature
    dG, dG_err = df * kBT, df_err * kBT  # kJ/mol

    overlap = pair_overlap_from_mbar(mbar)

    # Componentwise calculations
    w_fwd_by_component, w_rev_by_component = jax.vmap(works_from_ukln)(u_kln_by_component)
    dG_err_by_component_ = []
    overlap_by_component_ = []
    for u_kln in u_kln_by_component:
        component_mbar = construct_mbar_from_u_kln(u_kln)
        dG_err_by_component_.append(df_and_err_from_mbar(component_mbar)[1] * kBT)
        overlap_by_component_.append(pair_overlap_from_mbar(component_mbar))

    dG_err_by_component = np.array(dG_err_by_component_)
    overlap_by_component = np.array(overlap_by_component_)

    # When forward and reverse works are identically zero (usually because a given energy term does not depend on
    # lambda, e.g. host-host nonbonded interactions), BAR error is undefined; we return 0.0 by convention.
    dG_err_by_component = np.where(
        np.all(np.isclose(w_fwd_by_component, 0.0), axis=1) & np.all(np.isclose(w_rev_by_component, 0.0), axis=1),
        0.0,
        dG_err_by_component,
    )

    return BarResult(dG, dG_err, dG_err_by_component, overlap, overlap_by_component, u_kln_by_component)


def make_pair_bar_plots(res: PairBarResult, temperature: float, prefix: str) -> PairBarPlots:
    U_names = [type(p.potential).__name__ for p in res.initial_states[0].potentials]
    lambdas = [s.lamb for s in res.initial_states]

    dG_errs_png = plot_as_png_fxn(plot_dG_errs_figure, U_names, lambdas, res.dG_errs, res.dG_err_by_component_by_lambda)

    overlap_summary_png = plot_as_png_fxn(
        plot_overlap_summary_figure, U_names, lambdas, res.overlaps, res.overlap_by_component_by_lambda
    )

    return PairBarPlots(dG_errs_png, overlap_summary_png)


def assert_deep_eq(obj1, obj2, custom_assertion=lambda path, x1, x2: False):
    def is_dataclass_instance(obj):
        return is_dataclass(obj) and not isinstance(obj, type)

    def go(x1, x2, path=("$",)):
        def assert_(cond, reason):
            assert cond, f"objects differ in field {'.'.join(path)}: {reason}"

        if custom_assertion(path, x1, x2):
            pass
        elif type(x1) is not type(x2):
            assert_(False, f"types differ (left={type(x1)}, right={type(x2)})")
        elif is_dataclass_instance(x1) and is_dataclass_instance(x2):
            go(asdict(x1), asdict(x2), path)
        elif isinstance(x1, (np.ndarray, jax.Array)):
            assert_(np.array_equal(x1, x2), "arrays not equal")
        elif isinstance(x1, dict):
            assert_(x1.keys() == x2.keys(), "dataclass fields or dictionary keys differ")
            for k in x1.keys():
                go(x1[k], x2[k], (*path, str(k)))
        elif isinstance(x1, Sequence):
            assert_(len(x1) == len(x2), f"lengths differ (left={len(x1)}, right={len(x2)})")
            for idx, (v1, v2) in enumerate(zip(x1, x2)):
                go(v1, v2, (*path, f"[{idx}]"))
        else:
            assert_(x1 == x2, "left != right")

    return go(obj1, obj2, ("$",))


def assert_potentials_compatible(bps1: Sequence[BoundPotential], bps2: Sequence[BoundPotential]):
    """Asserts that two sequences of bound potentials are equivalent except for their parameters"""

    ps1 = [bp.potential for bp in bps1]
    ps2 = [bp.potential for bp in bps2]

    # We override the default deep equality check to allow SummedPotentials to differ in the values of the initial
    # parameters, as long as the shapes are consistent

    def custom_assertion(path, x1, x2):
        if len(path) >= 2 and path[-2] == "params_init":
            assert x1.shape == x2.shape, f"shape mismatch in field {'.'.join(path)}"
            return True
        return False

    assert_deep_eq(ps1, ps2, custom_assertion)


def run_sims_sequential(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    temperature: float,
) -> tuple[PairBarResult, list[Trajectory]]:
    """Sequentially run simulations at each state in initial_states,
    returning summaries that can be used for pair BAR, energy decomposition, and other diagnostics

    Returns
    -------
    PairBarResult
        Results of pair BAR analysis

    list of Trajectory
        Trajectory for each state in initial_states

    Notes
    -----
    * Memory complexity:
        Memory demand should be no more than that of 2 states worth of frames.
        Disk demand is proportional to number of initial states.

        This restriction may need to be relaxed in the future if:
        * We decide to use MBAR(states) rather than sum_i BAR(states[i], states[i+1])
        * We use online protocol optimization approaches that require more states to be kept on-hand
    """
    stored_trajectories = []

    # Ensure that states differ only in their parameters so that we can safely instantiate potentials from the first
    # state and use set_params for efficiency
    for s in initial_states[1:]:
        assert_potentials_compatible(initial_states[0].potentials, s.potentials)

    unbound_impls = [p.potential.to_gpu(np.float32).unbound_impl for p in initial_states[0].potentials]
    for initial_state in initial_states:
        # run simulation
        traj = sample(initial_state, md_params, max_buffer_frames=100)
        print(f"completed simulation at lambda={initial_state.lamb}!")

        # keep samples from any requested states in memory
        stored_trajectories.append(traj)

    neighbor_ulkns_by_component = generate_pair_bar_ulkns(
        initial_states, stored_trajectories, temperature, unbound_impls=unbound_impls
    )

    pair_bar_results = [
        estimate_free_energy_bar(u_kln_by_component, temperature) for u_kln_by_component in neighbor_ulkns_by_component
    ]

    return PairBarResult(list(initial_states), pair_bar_results), stored_trajectories


class MinOverlapWarning(UserWarning):
    pass


def run_sims_bisection(
    initial_lambdas: Sequence[float],
    make_initial_state: Callable[[float], InitialState],
    md_params: MDParams,
    n_bisections: int,
    temperature: float,
    min_overlap: Optional[float] = None,
    verbose: bool = True,
) -> tuple[list[PairBarResult], list[Trajectory]]:
    r"""Starting from a specified lambda schedule, successively bisect the lambda interval between the pair of states
    with the lowest BAR overlap and sample the new state with MD.

    Parameters
    ----------
    initial_lambdas: sequence of float, length >= 2, monotonically increasing
        Initial protocol; starting point for bisection.

    make_initial_state: callable
        Function returning an InitialState (i.e., starting point for MD) given lambda

    md_params: MDParams
        Parameters used to simulate new states

    n_bisections: int
        Number of bisection steps to perform

    temperature: float
        Temperature in K

    min_overlap: float or None, optional
        If not None, return early when the BAR overlap between all neighboring pairs of states exceeds this value

    verbose: bool, optional
        Whether to print diagnostic information

    Returns
    -------
    list of IntermediateResult
        For each iteration of bisection, object containing the current list of states and array of energy-decomposed
        u_kln matrices.

    list of Trajectory
        Trajectory for each state
    """

    assert len(initial_lambdas) >= 2
    assert np.all(np.diff(initial_lambdas) > 0), "initial lambda schedule must be monotonically increasing"

    lambdas = list(initial_lambdas)

    get_initial_state = cache(make_initial_state)
    rng = np.random.default_rng(md_params.seed)

    @cache
    def get_samples(lamb: float) -> Trajectory:
        initial_state = get_initial_state(lamb)
        # Update the seed of the MDParams to ensure Local MD doesn't select the same sequence of reference atoms
        traj = sample(
            initial_state, replace(md_params, seed=rng.integers(np.iinfo(np.int32).max)), max_buffer_frames=100
        )
        return traj

    # Set up a single set of unbound potentials for generating the BarResult objects
    potentials_0 = get_initial_state(lambdas[0]).potentials
    unbound_impls = [p.potential.to_gpu(np.float32).unbound_impl for p in potentials_0]

    @cache
    def get_bar_result(lamb1: float, lamb2: float) -> BarResult:
        initial_states = [get_initial_state(lamb1), get_initial_state(lamb2)]
        trajs = [get_samples(lamb1), get_samples(lamb2)]
        u_kln_by_component = generate_pair_bar_ulkns(initial_states, trajs, temperature, unbound_impls)

        # Trim off the first dimension
        assert u_kln_by_component.shape[0] == 1
        u_kln_by_component = u_kln_by_component.reshape(u_kln_by_component.shape[1:])
        assert len(u_kln_by_component.shape) == 4
        return estimate_free_energy_bar(u_kln_by_component, temperature, n_bootstrap=0)

    def overlap_to_cost(overlap: float) -> float:
        """Use -log(overlap) as the cost function for bisection; i.e., bisect the pair of states with lowest overlap."""
        return -np.log(overlap) if overlap != 0.0 else float("inf")

    def cost_to_overlap(cost: float) -> float:
        return np.exp(-cost)

    def cost_fn(lamb1: float, lamb2: float) -> float:
        overlap = get_bar_result(lamb1, lamb2).overlap
        return overlap_to_cost(overlap)

    def midpoint(x1: float, x2: float) -> float:
        return (x1 + x2) / 2.0

    def compute_intermediate_result(lambdas: Sequence[float]) -> PairBarResult:
        refined_initial_states = [get_initial_state(lamb) for lamb in lambdas]
        bar_results = [get_bar_result(lamb1, lamb2) for lamb1, lamb2 in zip(lambdas, lambdas[1:])]
        return PairBarResult(refined_initial_states, bar_results)

    result = compute_intermediate_result(lambdas)
    results = [result]

    for iteration in range(n_bisections):
        if min_overlap is not None and np.all(np.array(result.overlaps) > min_overlap):
            if verbose:
                print(f"All BAR overlaps exceed min_overlap={min_overlap}. Returning after {iteration} iterations.")
            break

        lambdas_new, info = greedy_bisection_step(lambdas, cost_fn, midpoint)
        if verbose:
            costs, left_idx, lamb_new = info
            lamb1 = lambdas[left_idx]
            lamb2 = lambdas[left_idx + 1]

            if min_overlap is not None:
                overlap_info = f"Current minimum BAR overlap {cost_to_overlap(max(costs)):.3g} <= {min_overlap:.3g} "
            else:
                overlap_info = f"Current minimum BAR overlap {cost_to_overlap(max(costs)):.3g} (min_overlap == None) "

            print(
                f"Bisection iteration {iteration + 1} (of {n_bisections}): "
                + overlap_info
                + f"between states at λ={lamb1:.3g} and λ={lamb2:.3g}. "
                f"Sampling new state at λ={lamb_new:.3g}…"
            )

        lambdas = lambdas_new
        result = compute_intermediate_result(lambdas)
        results.append(result)
    else:
        if min_overlap is not None and np.min(result.overlaps) < min_overlap:
            warn(
                f"Reached n_bisections={n_bisections} iterations without achieving min_overlap={min_overlap}. "
                f"The minimum BAR overlap was {np.min(result.overlaps)}.",
                MinOverlapWarning,
            )

    trajectories = [get_samples(lamb) for lamb in lambdas]

    return results, trajectories


def compute_potential_matrix(
    potential: custom_ops.Potential_f32,
    hrex: HREX[CoordsVelBox],
    params_by_state: NDArray,
    max_delta_states: Optional[int] = None,
) -> NDArray[np.float32]:
    """Computes the (n_replicas, n_states) sparse matrix of potential energies, where a given element $(k, l)$ is
    computed if and only if state $l$ is within `max_delta_states` of the current state of replica $k$, and is otherwise
    set to `np.inf`.

    Parameters
    ----------
    potential : custom_ops.Potential_f32
        potential to evaluate

    hrex : HREX
        HREX state (containing replica states and permutation)

    params_by_state : NDArray
        (n_states, ...) array of potential parameters for each state

    max_delta_states : int or None, optional
        If given, number of neighbor states on either side of a given replica's initial state for which to compute
        potentials. Otherwise, compute potentials for all (replica, state) pairs.
    """

    coords = np.array([xvb.coords for xvb in hrex.replicas])
    boxes = np.array([xvb.box for xvb in hrex.replicas])

    def compute_sparse(k: int):
        n_states = len(hrex.replicas)
        state_idx = np.argsort(hrex.replica_idx_by_state)
        neighbor_state_idxs = state_idx[:, None] + np.arange(-k, k + 1)[None, :]
        valid_idxs: tuple = np.nonzero((0 <= neighbor_state_idxs) & (neighbor_state_idxs < n_states))
        coords_batch_idxs = valid_idxs[0].astype(np.uint32)
        params_batch_idxs = neighbor_state_idxs[valid_idxs].astype(np.uint32)

        _, _, U = potential.execute_batch_sparse(
            coords, params_by_state, boxes, coords_batch_idxs, params_batch_idxs, False, False, True
        )

        U_kl = np.full((n_states, n_states), np.inf, dtype=np.float32)
        U_kl[coords_batch_idxs, params_batch_idxs] = U

        return U_kl

    def compute_dense():
        _, _, U_kl = potential.execute_batch(coords, params_by_state, boxes, False, False, True)
        return U_kl

    U_kl = compute_sparse(max_delta_states) if max_delta_states is not None else compute_dense()

    return U_kl


def batch_compute_potential_matrix(
    potentials: list[custom_ops.Potential_f32],
    hrex: HREX[CoordsVelBox],
    params_by_state_by_potential: list[NDArray],
    max_delta_states: Optional[int] = None,
) -> NDArray[np.float32]:
    """Computes the (n_potentials, n_replicas, n_states) sparse matrix of potential energies, where a given element $(k, l)$ is
    computed if and only if state $l$ is within `max_delta_states` of the current state of replica $k$, and is otherwise
    set to `np.inf`.

    A batched variant of `compute_potential_matrix`.

    Parameters
    ----------
    potentials : list[custom_ops.Potential_f32]
        list of potentials to evaluate

    hrex : HREX
        HREX state (containing replica states and permutation)

    params_by_state : NDArray
        (n_states, ...) array of potential parameters for each state

    max_delta_states : int or None, optional
        If given, number of neighbor states on either side of a given replica's initial state for which to compute
        potentials. Otherwise, compute potentials for all (replica, state) pairs.
    """

    potential_runner = custom_ops.PotentialExecutor_f32()
    coords = np.array([xvb.coords for xvb in hrex.replicas])
    boxes = np.array([xvb.box for xvb in hrex.replicas])

    def compute_sparse(k: int):
        n_states = len(hrex.replicas)
        state_idx = np.argsort(hrex.replica_idx_by_state)
        neighbor_state_idxs = state_idx[:, None] + np.arange(-k, k + 1)[None, :]
        valid_idxs: tuple = np.nonzero((0 <= neighbor_state_idxs) & (neighbor_state_idxs < n_states))
        coords_batch_idxs = valid_idxs[0].astype(np.uint32)
        params_batch_idxs = neighbor_state_idxs[valid_idxs].astype(np.uint32)

        _, _, U = potential_runner.execute_batch_sparse(
            potentials,
            coords,
            params_by_state_by_potential,
            boxes,
            coords_batch_idxs,
            params_batch_idxs,
            False,
            False,
            True,
        )

        U_kl = np.full((len(potentials), n_states, n_states), np.inf, dtype=np.float32)
        U_kl[:, coords_batch_idxs, params_batch_idxs] = U

        return U_kl

    def compute_dense():
        _, _, U_kl = potential_runner.execute_batch(
            potentials, coords, params_by_state_by_potential, boxes, False, False, True
        )
        return U_kl

    U_kl = compute_sparse(max_delta_states) if max_delta_states is not None else compute_dense()

    return U_kl


def verify_and_sanitize_potential_matrix(
    U_kl: NDArray, replica_idx_by_state: Sequence[int], abs_energy_threshold: float = 1e9
) -> NDArray:
    """Ensure energies in the diagonal are finite and below some threshold and sanitizes NaNs to infs."""
    # Verify that the energies that the energies of the replica in the same state are finite, else a replica is no longer valid
    replica_energies = np.diagonal(U_kl[replica_idx_by_state])
    assert np.all(np.isfinite(replica_energies)), "Replicas have non-finite energies"
    assert np.all(np.abs(replica_energies) < abs_energy_threshold), "Energies larger in magnitude than tolerated"
    return sanitize_energies_for_bar(U_kl)


def assert_ensembles_compatible(state_a: InitialState, state_b: InitialState):
    """check that xvb from state_a can be swapped with xvb from state_b (up to timestep error),
    with swap acceptance probability that depends only on U_a, U_b, kBT"""

    # assert (A, B) have identical masses, temperature
    intg_a = state_a.integrator
    intg_b = state_b.integrator

    assert (intg_a.masses == intg_b.masses).all()
    assert intg_a.temperature == intg_b.temperature

    # assert same pressure (or same volume)
    assert (state_a.barostat is None) == (state_b.barostat is None), "should both be NVT or both be NPT"

    if state_a.barostat and state_b.barostat:
        # assert (A, B) are compatible NPT ensembles
        baro_a: MonteCarloBarostat = state_a.barostat
        baro_b: MonteCarloBarostat = state_b.barostat

        assert baro_a.pressure == baro_b.pressure
        assert baro_a.temperature == baro_b.temperature

        # also, assert barostat and integrator are self-consistent
        assert intg_a.temperature == baro_a.temperature

        # The protein and water parameters should match exactly for the water sampling parameters
        water_sampler_params_a = get_water_sampler_params(state_a)
        water_sampler_params_b = get_water_sampler_params(state_b)
        assert (state_a.ligand_idxs == state_b.ligand_idxs).all()
        non_ligand_idxs = np.delete(np.arange(state_a.x0.shape[0]), state_a.ligand_idxs)
        assert (water_sampler_params_a[non_ligand_idxs] == water_sampler_params_b[non_ligand_idxs]).all()
    else:
        # assert (A, B) are compatible NVT ensembles
        assert (state_a.box0 == state_b.box0).all()


def compute_u_kn(trajs, initial_states) -> tuple[NDArray, NDArray]:
    """makes K calls to execute_batch


    Notes
    -----
    * Will load each trajectory into memory along with all parameters for each evaluation. This
      may result in OOMs if called with large numbers of frames and initial states.
    """

    for s in initial_states[1:]:
        assert_potentials_compatible(initial_states[0].potentials, s.potentials)
        assert_ensembles_compatible(initial_states[0], s)

    N_k = [len(traj.frames) for traj in trajs]
    kBTs = [BOLTZ * state.integrator.temperature for state in initial_states]
    assert len(set(kBTs)) == 1
    summed_pot = make_summed_potential(initial_states[0].potentials)
    K = len(initial_states)
    P = len(summed_pot.params)
    all_params = np.zeros((K, P), dtype=np.float32)
    for i in range(K):
        s = initial_states[i]
        all_params[i] = make_summed_potential(s.potentials).params
    K = len(N_k)

    unbound_impl = summed_pot.potential.to_gpu(np.float32).unbound_impl
    assert len(initial_states) == K
    u_kln = np.nan * np.zeros((K, K, max(N_k)))
    for i, traj in enumerate(trajs):
        # The computations of the energies are computed serially on the GPU
        # TBD: Compute the parameters sets in parallel for improved throughput
        _, _, Us = unbound_impl.execute_batch(
            traj.frames, all_params, traj.boxes, compute_du_dx=False, compute_du_dp=False, compute_u=True
        )
        Us = Us.T  # Transpose to get energies by params
        us = Us.reshape(K, N_k[i]) / kBTs[i]
        u_kln[i, :, : N_k[i]] = np.nan_to_num(us, nan=+np.inf)

    u_kn = kln_to_kn(u_kln, N_k)
    return u_kn, np.array(N_k)


def generate_pair_bar_ulkns(
    initial_states: Sequence[InitialState],
    samples_by_state: Sequence[Trajectory],
    temperature: float,
    unbound_impls: list[custom_ops.Potential_f32] | None,
) -> NDArray:
    """Generate pair bair u_klns.
    This is a specialized variant of generating u_klns, only loading each set of frames into memory once.
    Each set of frames is loaded once then all of the parameters of interest are run in a batch.
    This improves throughput for potentials that use Neighborlists, as there are at most len(frames) neighborlist
    rebuilds, rather than 3 * len(frames).

    Returns
    -------
        u_klns: np.array[len(initial_states) - 1, len(unbound_impls), 2, 2, n_frames]
    """

    assert len(initial_states) > 0
    assert len(initial_states) == len(samples_by_state)
    num_samples = len(samples_by_state[0].boxes)
    assert all([len(traj.boxes) == num_samples for traj in samples_by_state])
    if unbound_impls is None:
        unbound_impls = [pot.potential.to_gpu(np.float32).unbound_impl for pot in initial_states[0].potentials]
    assert len(unbound_impls) == len(initial_states[0].potentials)
    kBT = temperature * BOLTZ
    # Construct an empty array
    energies_by_frames_by_params = np.zeros(
        (len(initial_states), len(initial_states), len(unbound_impls)), dtype=object
    )
    params_by_state = [[bp.params for bp in initial_state.potentials] for initial_state in initial_states]
    executor = custom_ops.PotentialExecutor_f32()
    for i, state in enumerate(initial_states):
        frames = np.array(samples_by_state[i].frames)
        boxes = np.asarray(samples_by_state[i].boxes)

        state_idxs = []
        if i > 0:
            state_idxs.append(i - 1)
        state_idxs.append(i)
        if i < len(initial_states) - 1:
            state_idxs.append(i + 1)
        state_params = [params_by_state[idx] for idx in state_idxs]
        params_by_state_by_pot = [np.asarray([params[i] for params in state_params]) for i in range(len(unbound_impls))]
        _, _, Us = executor.execute_batch(
            unbound_impls,
            frames,
            params_by_state_by_pot,
            boxes,
            compute_du_dx=False,
            compute_du_dp=False,
            compute_u=True,
        )
        us = Us / kBT
        for j in range(len(unbound_impls)):
            # Transpose to get energies by params
            per_pot_us = us[j].T
            for state_idx, p_us in zip(state_idxs, per_pot_us):
                energies_by_frames_by_params[i, state_idx, j] = p_us

    u_kln_by_component_by_lambda = np.empty(
        (len(initial_states) - 1, len(unbound_impls), 2, 2, num_samples), dtype=np.float32
    )
    for i, states in enumerate(zip(range(len(initial_states)), range(1, len(initial_states)))):
        assert len(states) == 2
        for j in range(len(unbound_impls)):
            for l in range(2):
                for k in range(2):
                    # energies_by_frames_by_params is frames of state k to params of l
                    u_kln_by_component_by_lambda[i, j, k, l] = energies_by_frames_by_params[states[k]][states[l]][j]
    return u_kln_by_component_by_lambda


def run_sims_hrex(
    initial_states: Sequence[InitialState],
    md_params: MDParams,
    n_swap_attempts_per_iter: Optional[int] = None,
    print_diagnostics_interval: Optional[int] = 10,
) -> tuple[PairBarResult, list[Trajectory], HREXDiagnostics, WaterSamplingDiagnostics | None]:
    r"""Sample from a sequence of states using nearest-neighbor Hamiltonian Replica EXchange (HREX).

    See documentation for :py:func:`tmd.md.hrex.run_hrex` for details of the algorithm and implementation.

    Parameters
    ----------
    initial_states: sequence of InitialState
        States to sample. Should be ordered such that adjacent states have significant overlap for good mixing
        performance

    md_params: MDParams
        MD parameters

    n_swap_attempts_per_iter: int or None, optional
        Number of nearest-neighbor swaps to attempt per iteration. Defaults to len(initial_states) ** 4.

    print_diagnostics_interval: int or None, optional
        If not None, print diagnostics every N iterations

    Returns
    -------
    PairBarResult
        results of pair BAR free energy analysis

    list of Trajectory
        Trajectory for each state

    HREXDiagnostics
        HREX statistics (e.g. swap rates, replica-state distribution)
    """

    assert md_params.hrex_params is not None

    # TODO: to support replica exchange with variable temperatures,
    #  consider modifying sample fxn to rescale velocities by sqrt(T_new/T_orig)
    for s in initial_states[1:]:
        assert_ensembles_compatible(initial_states[0], s)

    if n_swap_attempts_per_iter is None:
        n_swap_attempts_per_iter = get_swap_attempts_per_iter_heuristic(len(initial_states))

    # Ensure that states differ only in their parameters so that we can safely instantiate potentials from the first
    # state and use set_params for efficiency
    for s in initial_states[1:]:
        assert_potentials_compatible(initial_states[0].potentials, s.potentials)

    # Set up overall potential and context using the first state.
    context = get_context(initial_states[0], md_params=md_params)
    bound_potentials = context.get_potentials()
    assert len(bound_potentials) == len(initial_states[0].potentials)
    temperature = initial_states[0].integrator.temperature
    ligand_idxs = initial_states[0].ligand_idxs

    def get_state_params(initial_state: InitialState) -> list[NDArray]:
        return [bp.params for bp in initial_state.potentials]

    params_by_state = [get_state_params(initial_state) for initial_state in initial_states]

    params_by_state_by_pot = [
        np.asarray([params[i] for params in params_by_state]) for i in range(len(bound_potentials))
    ]

    water_params_by_state: Optional[NDArray] = None
    if md_params.water_sampling_params is not None:
        water_params_by_state = np.array([get_water_sampler_params(initial_state) for initial_state in initial_states])

    state_idxs = [StateIdx(i) for i, _ in enumerate(initial_states)]
    neighbor_pairs = list(zip(state_idxs, state_idxs[1:]))

    if len(initial_states) == 2:
        # Add an identity move to the mixture to ensure aperiodicity
        neighbor_pairs = [(StateIdx(0), StateIdx(0)), *neighbor_pairs]

    barostat = context.get_barostat()
    water_sampler: custom_ops.TIBDExchangeMove_f32 | custom_ops.TIBDExchangeMove_f64 | None = None

    if barostat is not None and md_params.water_sampling_params is not None:
        # Should have a barostat and a water sampler
        assert len(context.get_movers()) == 2
        water_sampler = next(mover for mover in context.get_movers() if isinstance(mover, WATER_SAMPLER_MOVERS))

    hrex = HREX.from_replicas([CoordsVelBox(s.x0, s.v0, s.box0) for s in initial_states])

    samples_by_state: list[Trajectory] = [Trajectory.empty() for _ in initial_states]
    replica_idx_by_state_by_iter: list[list[ReplicaIdx]] = []
    water_sampler_proposals_by_state_by_iter: list[list[tuple[int, int]]] = []
    fraction_accepted_by_pair_by_iter: list[list[tuple[int, int]]] = []

    if (
        md_params.water_sampling_params is not None
        and md_params.steps_per_frame * md_params.n_frames < md_params.water_sampling_params.interval
    ):
        warn("Not running any water sampling, too few steps of MD per window for the water sampling interval")

    begin_loop_time = time.perf_counter()
    last_update_time = begin_loop_time

    kBT = temperature * BOLTZ

    iterated_u_kln = np.full(
        (len(bound_potentials), len(initial_states), len(initial_states), md_params.n_frames), np.inf, dtype=np.float32
    )

    convergence: MBARConvergence | None = None
    if md_params.hrex_params.early_termination_params is not None:
        convergence = MBARConvergence(
            md_params.hrex_params.early_termination_params.num_samples,
            md_params.hrex_params.early_termination_params.prediction_delta,
            slope_threshold=md_params.hrex_params.early_termination_params.slope_threshold,
            kBT=kBT,
        )

    for current_frame in range(md_params.n_frames):
        water_sampling_acceptance_proposal_counts_by_state = [(0, 0) for _ in range(len(initial_states))]

        def sample_replica(xvb: CoordsVelBox, state_idx: StateIdx) -> tuple[NDArray, NDArray, NDArray, Optional[float]]:
            context.set_x_t(xvb.coords)
            context.set_v_t(xvb.velocities)
            context.set_box(xvb.box)

            params_by_bp = params_by_state[state_idx]
            for bp, params in zip(bound_potentials, params_by_bp):
                bp.set_params(params)

            # Movers are not called during local steps, so if local moves are mixed in need to account only for global steps
            current_mover_step = current_frame * md_params.steps_per_frame
            if md_params.local_md_params is not None:
                current_mover_step = current_frame * (md_params.steps_per_frame - md_params.local_md_params.local_steps)
            # Setup the MC movers of the Context
            starting_water_acceptances = 0
            starting_water_proposals = 0
            if water_sampler is not None:
                assert water_params_by_state is not None
                water_sampler.set_params(water_params_by_state[state_idx])
                water_sampler.set_step(current_mover_step)
                starting_water_proposals = water_sampler.n_proposed()
                starting_water_acceptances = water_sampler.n_accepted()
            if barostat is not None:
                barostat.set_step(current_mover_step)

            md_params_replica = replace(
                md_params,
                n_frames=1,
                # Run equilibration as part of the first frame
                n_eq_steps=md_params.n_eq_steps if current_frame == 0 else 0,
                seed=state_idx + current_frame,
            )

            assert md_params_replica.n_frames == 1
            # Get the next set of frames from the iterator, which will be the only value returned
            frame, box, final_velos = next(
                sample_with_context_iter(context, md_params_replica, temperature, ligand_idxs, batch_size=1)
            )
            assert frame.shape[0] == 1

            if water_sampler is not None:
                final_water_proposals = water_sampler.n_proposed() - starting_water_proposals
                final_water_acceptances = water_sampler.n_accepted() - starting_water_acceptances
                water_sampling_acceptance_proposal_counts_by_state[state_idx] = (
                    final_water_acceptances,
                    final_water_proposals,
                )

            final_barostat_volume_scale_factor = barostat.get_volume_scale_factor() if barostat is not None else None

            return frame[-1], box[-1], final_velos, final_barostat_volume_scale_factor

        def replica_from_samples(last_sample: tuple[NDArray, NDArray, NDArray, Optional[float]]) -> CoordsVelBox:
            frame, box, velos, _ = last_sample
            return CoordsVelBox(frame, velos, box)

        hrex, samples_by_state_iter = hrex.sample_replicas(sample_replica, replica_from_samples)
        water_sampler_proposals_by_state_by_iter.append(water_sampling_acceptance_proposal_counts_by_state)
        # Generate the per-potential U_kl, avoids the need to re-compute the U_kln at the end
        U_kl_raw = batch_compute_potential_matrix(
            [bp.get_potential() for bp in bound_potentials],
            hrex,
            params_by_state_by_pot,
            md_params.hrex_params.max_delta_states,
        )
        # Sum the per-potential components for performing swaps
        U_kl = verify_and_sanitize_potential_matrix(U_kl_raw.sum(0), hrex.replica_idx_by_state)

        # Re-order energies by state
        iterated_u_kln[:, :, :, current_frame] = (
            sanitize_energies_for_bar(np.array(U_kl_raw[:, hrex.replica_idx_by_state])) / kBT
        )
        log_q_kl = -U_kl / kBT

        replica_idx_by_state_by_iter.append(hrex.replica_idx_by_state)

        hrex, fraction_accepted_by_pair = hrex.attempt_neighbor_swaps_fast(
            neighbor_pairs,
            log_q_kl,
            n_swap_attempts_per_iter,
            md_params.seed + current_frame + 1,  # NOTE: "+ 1" is for bitwise compatibility with previous version
        )

        if len(initial_states) == 2:
            fraction_accepted_by_pair = fraction_accepted_by_pair[1:]  # remove stats for identity move

        for samples, (xs, boxes, velos, final_barostat_volume_scale_factor) in zip(
            samples_by_state, samples_by_state_iter
        ):
            samples.frames.extend([xs])
            samples.boxes.extend([boxes])
            samples.final_velocities = velos
            samples.final_barostat_volume_scale_factor = final_barostat_volume_scale_factor

        fraction_accepted_by_pair_by_iter.append(fraction_accepted_by_pair)

        if (
            md_params.hrex_params.early_termination_params is not None
            and (current_frame + 1) % md_params.hrex_params.early_termination_params.interval == 0
        ):
            assert convergence is not None
            convergence.add_estimate_from_u_kln(iterated_u_kln.sum(0)[:, :, : current_frame + 1])
            if convergence.converged():
                print(
                    f"HREX terminating after {current_frame + 1} frames, the MBAR estimates converged.  The max difference in the last {md_params.hrex_params.early_termination_params.num_samples} estimates is {convergence.max_difference():.2f} kJ/mol"
                )
                break

        if print_diagnostics_interval and (current_frame + 1) % print_diagnostics_interval == 0:
            current_time = time.perf_counter()

            def get_swap_acceptance_rates(fraction_accepted_by_pair):
                return [
                    n_accepted / n_proposed if n_proposed else np.nan
                    for n_accepted, n_proposed in fraction_accepted_by_pair
                ]

            instantaneous_swap_acceptance_rates = get_swap_acceptance_rates(fraction_accepted_by_pair)
            average_swap_acceptance_rates = get_swap_acceptance_rates(np.sum(fraction_accepted_by_pair_by_iter, axis=0))

            wall_time_per_frame_current = (current_time - last_update_time) / print_diagnostics_interval
            wall_time_per_frame_average = (current_time - begin_loop_time) / (current_frame + 1)
            estimated_wall_time_remaining = wall_time_per_frame_average * (md_params.n_frames - (current_frame + 1))

            def format_rate(r):
                return f"{r * 100.0:5.1f}%"

            def format_rates(rs):
                return " |".join(format_rate(r) for r in rs)

            print("Frame", current_frame + 1)
            print(
                f"{estimated_wall_time_remaining:.1f} s remaining at "
                f"{wall_time_per_frame_average:.2f} s/frame "
                f"({wall_time_per_frame_current:.2f} s/frame since last message)"
            )
            print("HREX acceptance rates, current:", format_rates(instantaneous_swap_acceptance_rates))
            print("HREX acceptance rates, average:", format_rates(average_swap_acceptance_rates))
            print("HREX replica permutation      :", hrex.replica_idx_by_state)
            # (fey): Easily compute MBAR estimate from the u_kln. Could be used in the future to terminate early
            # in the case that estimates have converged
            # assert current_frame > 0
            # df, df_err = df_and_err_from_u_kln(iterated_u_kln.sum(0)[:, :, : current_frame + 1])
            # print(f"HREX dG (kJ/mol) Estimate {df * kBT:.2f} dG Error {df_err * kBT:.2f}")
            print()

            last_update_time = current_time

    neighbor_ulkns_by_component = [
        iterated_u_kln[:, i : i + 2, i : i + 2, : current_frame + 1] for i in range(len(initial_states) - 1)
    ]

    pair_bar_results = [
        estimate_free_energy_bar(u_kln_by_component, temperature, n_bootstrap=100)
        for u_kln_by_component in neighbor_ulkns_by_component
    ]

    hrex_diagnostics = HREXDiagnostics(
        replica_idx_by_state_by_iter,
        fraction_accepted_by_pair_by_iter,
        None if convergence is None else convergence.estimates(),
    )
    ws_diagnostics: WaterSamplingDiagnostics | None = None
    if md_params.water_sampling_params is not None:
        ws_diagnostics = WaterSamplingDiagnostics(np.array(water_sampler_proposals_by_state_by_iter, dtype=np.int32))

    return PairBarResult(list(initial_states), pair_bar_results), samples_by_state, hrex_diagnostics, ws_diagnostics
