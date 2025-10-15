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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from rdkit import Chem

from tmd.constants import BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP, MAX_FORCE_NORM, NBParamIdx
from tmd.fe import topology
from tmd.fe.model_utils import apply_hmr
from tmd.fe.utils import get_romol_conf, set_romol_conf
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, MonteCarloBarostat, custom_ops
from tmd.md.barker import BarkerProposal
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.md.builders import HostConfig
from tmd.md.exchange.exchange_mover import get_water_idxs
from tmd.md.fire import fire_descent
from tmd.md.thermostat.utils import sample_velocities
from tmd.potentials import (
    BoundPotential,
    HarmonicBond,
    Nonbonded,
    Potential,
    SummedPotential,
    make_summed_potential,
)
from tmd.potentials.bonded import harmonic_positional_restraint
from tmd.potentials.jax_utils import pairwise_distances
from tmd.potentials.potential import get_bound_potential_by_type, get_potential_by_type


class MinimizationWarning(UserWarning):
    pass


class MinimizationError(Exception):
    pass


@dataclass(frozen=True)
class FireMinimizationConfig:
    """Refer to tmd.md.fire.fire_descent for documentation of each parameter"""

    n_steps: int
    dt_start: float = 1e-5
    dt_max: float = 1e-3
    n_min: float = 5
    f_inc: float = 1.1
    f_dec: float = 0.5
    alpha_start: float = 0.1
    f_alpha: float = 0.99


@dataclass(frozen=True)
class ScipyMinimizationConfig:
    """Allows for using any scipy.optimize.minimize method that supports jac=True.

    Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize for
    documentation
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)
    bounds: Optional[Sequence | scipy.optimize.Bounds] = None


MinimizationConfig: TypeAlias = FireMinimizationConfig | ScipyMinimizationConfig


def check_force_norm(forces: NDArray, threshold: float = MAX_FORCE_NORM):
    """raise MinimizationError if the force on any atom exceeds MAX_FORCE_NORM"""
    per_atom_force_norms = np.linalg.norm(forces, axis=-1)

    if (per_atom_force_norms > threshold).any():
        bad_inds = np.where(per_atom_force_norms > threshold)[0]
        max_atom_force_norm = np.max(per_atom_force_norms)
        message = f"""
        Minimization failed to reduce large forces below threshold:
            max |frc| = {max_atom_force_norm} > {threshold}
            {len(bad_inds)} / {len(forces)} atoms exceed threshold
        """
        raise MinimizationError(message)


def _adjust_mol_parameters_for_minimization(bps: list[BoundPotential], num_host_atoms: int, mols: list[Chem.Mol]):
    """Logic for adjusting bound potentials when minimizing. Typically done to improve the efficiency and to produce
    a fully minimized system that is consistent with the downstream simulations.

    Note that this is because MultiTopology (and DualTopology) have all molecules fully interacting with the environment.
    """
    # Scale down the charges by the number of mols
    nb = get_bound_potential_by_type(bps, Nonbonded)
    nb.params = jnp.asarray(nb.params).at[num_host_atoms:, NBParamIdx.Q_IDX].divide(len(mols))


def parameterize_system(topo, ff: Forcefield, lamb: float) -> tuple[list[Potential], list[NDArray]]:
    # setup the parameter handlers for the ligand
    ff_params = ff.get_params()
    params_potential_pairs = [
        topo.parameterize_harmonic_bond(ff_params.hb_params),
        topo.parameterize_harmonic_angle(ff_params.ha_params),
        topo.parameterize_proper_torsion(ff_params.pt_params),
        topo.parameterize_improper_torsion(ff_params.it_params),
        topo.parameterize_nonbonded_pairlist(
            ff_params.q_params,
            ff_params.q_params_intra,
            ff_params.lj_params,
            ff_params.lj_params_intra,
        ),
        topo.parameterize_nonbonded(
            ff_params.q_params,
            ff_params.q_params_intra,
            ff_params.lj_params,
            ff_params.lj_params_intra,
            lamb,
        ),
    ]
    return [pot for (_, pot) in params_potential_pairs], [params for (params, _) in params_potential_pairs]


def flatten_params(params: list[NDArray]) -> NDArray:
    return np.concatenate([p.reshape(-1) for p in params])


def summed_potential_bound_impl_from_potentials_and_params(
    potentials: list[Potential], params: list[NDArray]
) -> custom_ops.BoundPotential_f32:
    flat_params = flatten_params(params)
    return SummedPotential(potentials, params).bind(flat_params).to_gpu(precision=np.float32).bound_impl


def fire_minimize(
    x0: NDArray,
    du_dx_fxn: Callable[[NDArray], NDArray],
    config: FireMinimizationConfig,
) -> NDArray:
    """
    Minimize coordinates using the FIRE algorithm

    Parameters
    ----------
    coords: np.ndarray
        N x 3 coordinates. units of nanometers.

    du_dx_fxn: Function that given coordinates returns the du_dx

    config: FireMinimizationConfig
        Fire configuration for minimization

    Returns
    -------
    np.ndarray
        Minimized coords.

    """

    def force(coords):
        forces = -du_dx_fxn(coords)
        return forces

    def shift(d, dr):
        return d + dr

    init, f = fire_descent(
        force,
        shift,
        dt_start=config.dt_start,
        dt_max=config.dt_max,
        n_min=config.n_min,
        f_inc=config.f_inc,
        f_dec=config.f_dec,
        alpha_start=config.alpha_start,
        f_alpha=config.f_alpha,
    )
    opt_state = init(x0)
    for _ in range(config.n_steps):
        opt_state = f(opt_state)
    return np.asarray(opt_state.position)


def local_equilibrate(
    x0: NDArray,
    box0: NDArray,
    masses: NDArray,
    potentials: list[BoundPotential],
    local_idxs: list[int] | NDArray,
    steps: int,
    dt: float = 1.5e-3,
    seed: int = 2025,
    temperature: float = DEFAULT_TEMP,
):
    """Equilibrate a subset of a system using a LangevinIntegrator.

    Parameters
    ----------
    x0: np.array (N,3)
        Coordinates

    box0: np.array (3,3)
        Box

    masses: np.array (N,)
        Masses

    potentials: list of BoundPotentials
        Bound potentials for the system

    local_idxs: list of int
        Unique idxs we allow to move.

    steps: int
        Number of steps to equilibrate for

    dt: float
        Step size

    seed: integer
        integrator uses `seed`

    temperature: float
        in kelvin (used by integrator)

    Returns
    -------
    Return equilibrated coords
    """

    free_idxs = np.asarray(local_idxs, dtype=np.int32)
    frozen_idxs = np.delete(np.arange(len(x0)), free_idxs)

    v0 = sample_velocities(masses, temperature, seed)

    # Set the masses to inf to freeze the atoms in place
    frozen_masses = np.array(masses)
    frozen_masses[frozen_idxs] = np.inf

    # Zero out the velocities of the frozen atoms, to prevent movement
    v0[frozen_idxs, :] = 0.0

    bound_impls = [bp.to_gpu(np.float32).bound_impl for bp in potentials]

    # Verify that the forces are reasonable to begin with
    for impl in bound_impls:
        du_dx, _ = impl.execute(x0, box0, compute_u=False)
        check_force_norm(-du_dx)

    friction = 1.0

    intg = LangevinIntegrator(temperature, dt, friction, frozen_masses, seed).impl()

    ctxt = custom_ops.Context_f32(x0, v0, box0, intg, bound_impls)
    xs, boxes = ctxt.multiple_steps(steps)
    x = xs[-1]
    v = ctxt.get_v_t()

    assert np.all(x[frozen_idxs] == x0[frozen_idxs]), "Frozen atoms unexpectedly moved"
    assert np.all(x[free_idxs] != x0[free_idxs]), "Free atoms didn't move"
    assert np.all(v[free_idxs] != v0[free_idxs]), "Free atoms didn't change velocity"

    # Verify that the system has reasonable forces after equilibration
    for impl in bound_impls:
        du_dx, _ = impl.execute(x, box0, compute_u=False)
        check_force_norm(-du_dx)

    return x


def pre_equilibrate_host(
    mols: list[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[list[NDArray]] = None,
    minimizer_steps_per_window: int = 500,
    minimizer_windows: int = 2,
    minimizer_max_lambda: float = 0.1,
    equilibration_steps: int = 1_000,
    pressure: float = DEFAULT_PRESSURE,
    temperature: float = DEFAULT_TEMP,
    barostat_interval: int = 10,
    seed: int = 2024,
    use_hmr: bool = True,
    water_sampling_interval: int = 600,
) -> tuple[NDArray, NDArray]:
    """pre_equilibrate_host is a utility function that performs minimization than some amount of equilibration.

    The intention of this function is to resolve any potential clashes in the system, then to equilibrate the box size, all while
    keeping the ligand fixed. This helps set up up simulations where the local region around the ligand is optimized over a set of windows.

    Note that the ligand charges will be scaled down by the number of molecules.

    Parameters
    ----------
    mols: list of Chem.Mol
        Ligands to be inserted. This must be of length 1 or 2 for now.

    host_config: HostConfig
        Represents the host system.

    ff: ff.Forcefield
        Wrapper class around a list of handlers

    mol_coords: list of np.ndarray, optional
        Pre-specify a list of mol coords. Else use the mol.GetConformer(0)

    minimizer_steps_per_window: integer
        Number of steps to run each window of the FIRE minimizer with

    minimizer_windows: integer
        Number of windows to run FIRE minimizer over

    minimizer_max_lambda: float
        The largest lambda value to run the minimizer with. Refer to docstring of fire_minimize_host for more
        details.

    equilibration_steps: integer
        Number of steps to run MD with

    pressure: float
        in bar (used by barostat)

    temperature: float
        in kelvin (used by integrator and barostat)

    barostat_interval: integer
        # of MD steps between barostat moves

    seed: integer
        integrator uses `seed`, barostat uses `seed+1`

    use_hmr: bool
        Whether or not apply hmr. If use_hmr=True, dt is increased from 1.5fs to 2.5fs.

    water_sampling_interval: int
        If >0 then enables water sampling around the mols provided. Assumes molecules are in the same region.
        Disabled for solvent hosts regardless of setting.

    Returns
    -------
    2-tuple of host coordinates and box
        the minimized host_coords and the new box

    """
    assert len(mols) > 0
    box = host_config.box
    assert box.shape == (3, 3)

    minimized_host_coords = fire_minimize_host(
        mols,
        host_config,
        ff,
        mol_coords=mol_coords,
        n_windows=minimizer_windows,
        n_steps_per_window=minimizer_steps_per_window,
        max_lambda=minimizer_max_lambda,
    )

    num_host_atoms = host_config.conf.shape[0]

    top = topology.MultiTopology(mols, ff)

    mass_list = [np.array(host_config.masses)]
    conf_list = [minimized_host_coords]
    for mol in mols:
        # Set ligand masses to inf to ensure ligands don't move
        mass_list.append(np.ones(mol.GetNumAtoms()) * np.inf)

    if mol_coords is None:
        mol_coords = [get_romol_conf(mol) for mol in mols]
    for mc in mol_coords:
        conf_list.append(mc)

    combined_masses = np.concatenate(mass_list)
    combined_coords = np.concatenate(conf_list)

    hgt = topology.HostGuestTopology(
        host_config.host_system.get_U_fns(), top, host_config.num_water_atoms, ff, host_config.omm_topology
    )

    potentials, params = parameterize_system(hgt, ff, 0.0)

    bps = [pot.bind(p) for pot, p in zip(potentials, params)]

    _adjust_mol_parameters_for_minimization(bps, num_host_atoms, mols)

    bond_pot = get_potential_by_type(potentials, HarmonicBond)
    bond_list = get_bond_list(bond_pot)
    group_idxs = get_group_indices(bond_list, combined_coords.shape[0])

    dt = 1.5e-3
    if use_hmr:
        dt = 2.5e-3
        combined_masses = np.array(apply_hmr(combined_masses, bond_list))
    friction = 1.0

    intg = LangevinIntegrator(temperature, dt, friction, combined_masses, seed).impl()

    x0 = combined_coords
    v0 = np.zeros_like(x0)

    num_host_atoms = minimized_host_coords.shape[0]

    # Disallow the barostat from scaling the ligand coords, scale all of the other molecules to
    # reduce 'air bubbles' within the system. Less efficient than scaling the entire system, but
    # don't want to adjust the ligand coordinates at all.
    non_ligand_group_idxs = [group for group in group_idxs if np.all(group < num_host_atoms)]

    bound_impls = [bp.to_gpu(np.float32).bound_impl for bp in bps]

    movers = []
    if barostat_interval > 0:
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            non_ligand_group_idxs,
            barostat_interval,
            seed + 1,
        )
        baro_impl = baro.impl(bound_impls)

        movers.append(baro_impl)

    if water_sampling_interval > 0 and host_config.num_water_atoms != num_host_atoms:
        ligand_idxs = np.arange(num_host_atoms, len(combined_coords), dtype=np.int32)
        water_idxs = get_water_idxs(group_idxs, ligand_idxs=ligand_idxs)

        diameter = np.max(pairwise_distances(combined_coords[num_host_atoms:]))
        radius = diameter / 2

        nb = get_bound_potential_by_type(bps, Nonbonded)

        water_sampler = custom_ops.TIBDExchangeMove_f32(
            combined_coords.shape[0],
            ligand_idxs.tolist(),
            water_idxs,
            np.array(nb.params, dtype=np.float32),
            temperature,
            nb.potential.beta,  # type: ignore
            nb.potential.cutoff,  # type: ignore
            float(radius),
            seed + 2,
            1000,
            water_sampling_interval,
        )
        movers.append(water_sampler)

    ctxt = custom_ops.Context_f32(x0, v0, box, intg, bound_impls, movers=movers)
    xs, boxes = ctxt.multiple_steps(equilibration_steps)
    x = xs[-1]
    box = boxes[-1]

    assert np.all(x[num_host_atoms:] == np.concatenate(mol_coords)), "Ligand atoms unexpectedly moved"

    # Only evaluate the host forces, the mols may be strained at this stage. No change is made to the mols
    # which means evaluating the mols forces may trigger spurious failures
    for impl in bound_impls:
        du_dx, _ = impl.execute(x, box, compute_u=False)
        check_force_norm(-du_dx[:num_host_atoms])

    return x[:num_host_atoms], box


def fire_minimize_host(
    mols: list[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[list[NDArray]] = None,
    n_steps_per_window: int = 500,
    max_lambda: float = 0.1,
    n_windows: int = 2,
) -> NDArray:
    """
    Minimize a host system using the Fire minimizer.

    The ligand coordinates are fixed during minimization, and only host_coords are minimized.

    Parameters
    ----------
    mols: list of Chem.Mol
        Ligands to be inserted. This must be of length 1 or 2 for now.

    host_config: HostConfig
        Represents the host system.

    ff: ff.Forcefield
        Wrapper class around a list of handlers

    mol_coords: list of np.ndarray
        Pre-specify a list of mol coords. Else use the mol.GetConformer(0)

    n_steps_per_window: integer
        Number of steps to run the FIRE minimizer with at each window

    max_lambda: float
        The largest lambda value to run a window at. If the value is 1.0 the mols will be
        completely decoupled. Between lambda 1.0 and 0.0 the 4D coordinate of the mols are linearly
        interpolated from nonbonded cutoff to 0.0, no others parameters are modified.

    n_windows: integer
        The number of windows to linearly interpolate between the max_lambda value and 0.0.

    Returns
    -------
    np.ndarray
        the minimized host_coords.

    """

    assert 1.0 >= max_lambda > 0.0, "Max lambda must be greater than 0.0 and less than or equal to 1.0"
    x_host = np.asarray(host_config.conf, dtype=np.float32)

    config = FireMinimizationConfig(n_steps_per_window)

    for lamb in np.linspace(max_lambda, 0.0, n_windows):
        du_dx_fxn = make_host_du_dx_fxn(mols, host_config, ff, mol_coords=mol_coords, lamb=lamb)
        x_host = fire_minimize(x_host, du_dx_fxn, config)

    du_dx = du_dx_fxn(x_host)
    check_force_norm(-du_dx)

    return x_host


def make_host_du_dx_fxn(
    mols: list[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[list[NDArray]] = None,
    lamb: float = 0.0,
):
    """construct function to compute du_dx w.r.t. host coords, given fixed mols and box

    Scales down the charges of the provided molecules by the number of molecules. IE 2 mols will result in the charges being scaled by half."""
    assert len(mols) > 0
    assert host_config.box.shape == (3, 3)

    num_host_atoms = len(host_config.conf)

    top = topology.MultiTopology(mols, ff)

    hgt = topology.HostGuestTopology(
        host_config.host_system.get_U_fns(), top, host_config.num_water_atoms, ff, host_config.omm_topology
    )

    # read conformers from mol_coords if given, or each mol's conf0 otherwise
    conf_list = [np.asarray(host_config.conf)]

    if mol_coords is not None:
        for mc in mol_coords:
            conf_list.append(mc)
    else:
        for mol in mols:
            conf_list.append(get_romol_conf(mol))

    # check conf_list consistent with mols
    assert len(conf_list[1:]) == len(mols)
    for conf, mol in zip(conf_list[1:], mols):
        assert conf.shape == (mol.GetNumAtoms(), 3)

    combined_coords = np.concatenate(conf_list)

    # bound impls of potentials @ lam
    potentials, params = parameterize_system(hgt, ff, lamb)

    bps = [pot.bind(p) for pot, p in zip(potentials, params)]

    _adjust_mol_parameters_for_minimization(bps, num_host_atoms, mols)

    gpu_impl = make_summed_potential(bps).to_gpu(np.float32).bound_impl

    # wrap gpu_impl, partially applying box, mol coords
    def du_dx_host_fxn(x_host):
        x = np.array(combined_coords)
        x[:num_host_atoms] = x_host

        du_dx, _ = gpu_impl.execute(x, host_config.box, compute_u=False)
        du_dx_host = du_dx[:num_host_atoms]
        return du_dx_host

    return du_dx_host_fxn


def equilibrate_host_barker(
    mols: list[Chem.Mol],
    host_config: HostConfig,
    ff: Forcefield,
    mol_coords: Optional[list[NDArray]] = None,
    temperature: float = DEFAULT_TEMP,
    proposal_stddev: float = 0.0001,
    n_steps: int = 1000,
    seed: Optional[int] = None,
) -> NDArray:
    """Possible alternative to fire_minimize_host, for purposes of clash resolution and initial pre-equilibration

    Notes
    -----
    * Applies a robust proposal targeting lam = 0, and omits Metropolis correction
        * For sufficiently small proposal_stddev, can be expected to sample from approximately the correct distribution
        * Not expected to outperform BAOAB in terms of improved sampling efficiency or reduced sampling bias
        * Possible advantage: robustness w.r.t. clashy initialization
    * Can make progress even when |force| = +inf
    * At proposal_stddev = 0.0001 nm:
        * appears to resolve steric clashes
        * appears stable even with Metropolis correction omitted
    * Can be run as an approximate minimizer by setting temperature == 0.0
    """

    assert 0 < proposal_stddev <= 0.0001, "not tested with Metropolis correction omitted for larger proposal_stddevs"

    du_dx_host_fxn = make_host_du_dx_fxn(mols, host_config, ff, mol_coords)
    grad_log_q = lambda x_host: -du_dx_host_fxn(x_host) / (BOLTZ * temperature)

    # TODO: if needed, revisit choice to omit Metropolis correction
    barker_prop = BarkerProposal(grad_log_q, proposal_stddev, seed=seed)

    x_host = np.array(host_config.conf)

    for t in range(n_steps):
        x_host = barker_prop.sample(x_host)

    final_forces = -du_dx_host_fxn(x_host)
    check_force_norm(final_forces)

    return x_host


def get_val_and_grad_fn(bps: Sequence[BoundPotential], box: NDArray, precision=np.float32):
    """
    Convert impls, box into a function that only takes in coords.

    Parameters
    ----------
    bps: List of BoundPotentials

    box: np.array (3,3)

    precision: np.float64 or np.float32

    Returns
    -------
    Energy function with gradient
        f: R^(Nx3) -> (R^1, R^Nx3)
    """
    summed_pot = make_summed_potential(bps)
    impl = summed_pot.to_gpu(precision).bound_impl

    def val_and_grad_fn(coords):
        g_bp, u_bp = impl.execute(coords, box)
        assert len(u_bp) == 1
        return u_bp[0], g_bp

    return val_and_grad_fn


def wrap_val_and_grad_with_positional_restraint(
    val_and_grad_fn: Callable[[NDArray], tuple[float, NDArray]],
    x0: NDArray,
    box0: NDArray,
    free_idxs: NDArray,
    k: float,
) -> Callable[[NDArray], tuple[float, NDArray]]:
    restraint_val_and_grad = jax.value_and_grad(harmonic_positional_restraint, argnums=1)

    starting_free = np.array(x0[free_idxs])

    def wrapped_val_and_grad(x):
        u, grad = val_and_grad_fn(x)
        restraint_u, restraint_grad = restraint_val_and_grad(starting_free, x[free_idxs], box0, k=k)
        u += restraint_u
        grad = jnp.asarray(grad).at[free_idxs].add(restraint_grad)
        return u, grad

    return wrapped_val_and_grad


def scipy_minimize(
    x0: NDArray, val_and_grad_fn: Callable[[NDArray], tuple[float, NDArray]], config: ScipyMinimizationConfig
):
    final_shape = x0.shape

    # deals with reshaping from (L,3) -> (Lx3,)
    def val_and_grad_fn_bfgs(x_flattened):
        x = x_flattened.reshape(final_shape)
        u, grad_full = val_and_grad_fn(x)
        return u, np.asarray(grad_full.reshape(-1)).astype(np.float64)

    x_flat = x0.reshape(-1)

    res = scipy.optimize.minimize(
        val_and_grad_fn_bfgs,
        x_flat,
        method=config.method,
        jac=True,
        bounds=config.bounds,
        options=config.options,
    )

    return res.x.reshape(final_shape)


def local_minimize(
    x0: NDArray,
    box0: NDArray,
    val_and_grad_fn: Callable[[NDArray], tuple[float, NDArray]],
    local_idxs: list[int] | NDArray,
    minimizer_config: Sequence[MinimizationConfig],
    verbose: bool = True,
    assert_energy_decreased: bool = True,
    restraint_k: float = 0.0,
    restrained_idxs: Optional[NDArray] = None,
) -> NDArray:
    """
    Minimize a local region given selected idxs.

    Parameters:
    -----------
    x0: np.array (N,3)
        Coordinates

    box0: np.array (3,3)
        Box

    val_and_grad_fn: f: R^(Nx3) -> (R^1, R^Nx3)
        Energy function

    local_idxs: list of int
        Unique idxs we allow to move.

    minimizer_config: Sequence[ScipyMinimizationConfig | FireMinimizationConfig]
        A sequence of minimizer configs, If a minimization configuration fails
        will attempt to minimize with the next configuration. The first configuration to succeed
        will return the minimized coordinates.

    verbose: bool
        Print internal potential energy + gradient norm

    assert_energy_decreased: bool
        Throw an assertion if the energy does not decrease

    restraint_k: float
        Restraint k to wrap val_and_grad_fn in a positional harmonic restraint to the input positions of the local_idxs.
        Refer to `tmd.potentials.bonded.harmonic_positional_restraint` for implementation.

    restrained_idxs: np.ndarray, optional
        A subset of idxs to restrain, must be a subset of local_idxs. If restrained_idxs is None, all local_idxs are restrained.


    Returns
    -------
    Optimized set of coordinates (N,3)

    Raises
    ------
    MinimizationError

    """

    if not all(isinstance(config, (FireMinimizationConfig, ScipyMinimizationConfig)) for config in minimizer_config):
        raise ValueError(f"Invalid minimizer configs: {[type(config) for config in minimizer_config]}")
    assert restraint_k >= 0.0, "Restraint k must be greater than or equal to 0.0"
    if restrained_idxs is not None:
        assert restraint_k > 0.0, "Restraint k must be greater than 0.0 if restrained indices provided"
        assert set(restrained_idxs).issubset(set(local_idxs)), "Restrained indices must be a subset of local indices"

    assert len(local_idxs) == len(set(local_idxs))
    n_frozen = len(x0) - len(local_idxs)

    free_idxs = np.asarray(local_idxs)

    U_0, _ = val_and_grad_fn(x0)

    # Only use the restrained function when minimizing, don't otherwise use to compute energy/forces
    minimizer_val_and_grad = val_and_grad_fn
    if restraint_k > 0.0:
        if restrained_idxs is None:
            restrained_idxs = free_idxs
        minimizer_val_and_grad = wrap_val_and_grad_with_positional_restraint(
            minimizer_val_and_grad, x0, box0, restrained_idxs, restraint_k
        )

    x_local_0 = x0[free_idxs]

    def val_and_grad_fn_local(x_local):
        x_prime = x0.copy()
        x_prime[free_idxs] = x_local
        U_full, grad_full = minimizer_val_and_grad(x_prime)
        # The GPU Potentials can return NaN if value would have overflowed in uint64
        # FIRE only looks at the gradients and the gradients may be accurate when the energy is NaN
        if method != "FIRE" and np.isnan(U_full):
            U_full = np.inf
            grad_full = np.nan * grad_full
        return U_full, grad_full[free_idxs]

    last_err = None
    for config in minimizer_config:
        method = "FIRE"
        if isinstance(config, ScipyMinimizationConfig):
            method = config.method

        if verbose:
            print("-" * 70)
            print(
                f"performing {method} minimization on {len(free_idxs)} atoms\n(holding the other {n_frozen} atoms frozen)"
            )
            print(f"U(x_0) = {U_0:.3f}")

        if isinstance(config, ScipyMinimizationConfig):
            x_local_final = scipy_minimize(x_local_0, val_and_grad_fn_local, config)
        else:
            x_local_final = fire_minimize(x_local_0, lambda x: val_and_grad_fn_local(x)[1], config)

        x_final = x0.copy()
        x_final[free_idxs] = x_local_final

        U_final, grad_final = val_and_grad_fn(x_final)
        forces = -grad_final

        if verbose:
            per_atom_force_norms = np.linalg.norm(forces[free_idxs], axis=1)
            print(f"U(x_final) = {U_final:.3f}")
            # identify worst atom
            argmax_local = np.argmax(per_atom_force_norms)
            worst_atom_idx = free_idxs[argmax_local]
            print(f"atom with highest force norm after minimization: {worst_atom_idx}")
            print(f"force(x_final)[{worst_atom_idx}] = {forces[worst_atom_idx]}")
            print("-" * 70)

        try:
            check_force_norm(forces)

            if assert_energy_decreased:
                if not np.isnan(U_0) and U_final >= U_0:
                    raise MinimizationError(f"Final energy didn't decrease. U_0: {U_0:.3f}, U_f: {U_final:.3f}")
                elif not np.isfinite(U_final):
                    raise MinimizationError(f"Final energy is non-finite. U_0: {U_0:.3f}, U_f: {U_final:.3f}")
            elif U_final >= U_0:
                warnings.warn(
                    f"WARNING: Energy did not decrease: U_0: {U_0:.3f}, u_f: {U_final:.3f}", MinimizationWarning
                )
            # Reset the error, if minimization was a success
            last_err = None
            break
        except MinimizationError as e:
            warnings.warn(f"WARNING: Minimization failed for {type(config)}, falling back to next minimizer")
            last_err = e

    if last_err is not None:
        raise last_err
    return x_final


def replace_conformer_with_minimized(
    mol: Chem.rdchem.Mol,
    ff: Forcefield,
    minimizer_configs: Optional[Sequence[MinimizationConfig]] = None,
    conf_id: int = 0,
):
    """Replace the first conformer of the given mol with a conformer minimized with respect to the given forcefield.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Input mol. Must have at least one conformer.

    ff : Forcefield
        Forcefield to use in energy minimization

    minimizer_configs: Sequence[FireMinimizationConfig or ScipyMinimizationConfig], optional
        Defaults to BFGS minimization if not provided

    conf_id : int
        ID of the conformer to replace
    """
    top = topology.BaseTopology(mol, ff)
    system = top.setup_end_state()
    val_and_grad_fn = jax.value_and_grad(system.get_U_fn())
    xs = get_romol_conf(mol, conf_id)
    box = np.eye(3) * 100.0
    all_idxs = np.arange(mol.GetNumAtoms())

    if minimizer_configs is None:
        minimizer_configs = [ScipyMinimizationConfig(method="BFGS")]

    xs_opt = local_minimize(xs, box, val_and_grad_fn, all_idxs, minimizer_configs, verbose=False)
    set_romol_conf(mol, xs_opt, conf_id)
