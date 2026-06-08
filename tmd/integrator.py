# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2026, Forrest York
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
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom

from tmd.constants import BOLTZ
from tmd.lib.fixed_point import fixed_to_float, float_to_fixed


def langevin_coefficients(temperature, dt, friction, masses):
    """
    Compute coefficients for langevin dynamics

    Parameters
    ----------
    temperature: float
        units of Kelvin

    dt: float
        units of picoseconds

    friction: float
        collision rate in 1 / picoseconds

    masses: array
        mass of each atom in standard mass units. np.inf masses will
        effectively freeze the particles.

    Returns
    -------
    tuple (ca, cb, cc)
        ca is scalar, and cb and cc are n length arrays
        that are used during langevin dynamics as follows:

        during heat-bath update
        v -> ca * v + cc * gaussian

        during force update
        v -> v + cb * force
    """
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT / masses)

    ca = np.exp(-friction * dt)
    cb = dt / masses
    cc = np.sqrt(1 - np.exp(-2 * friction * dt)) * nscale

    return ca, cb, cc


class Integrator(ABC):
    @abstractmethod
    def step(self, x, v) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep"""
        pass

    def multiple_steps(self, x, v, n_steps: int = 1000):
        """Return trajectories of x and v, advanced by n_steps"""
        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)


class StochasticIntegrator(ABC):
    @abstractmethod
    def step(self, x, v, rng: np.random.Generator) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep. Accepts a numpy Generator instance for determinism."""
        pass

    @abstractmethod
    def step_lax(self, key, x, v) -> tuple[Any, Any]:
        """Return copies x and v, updated by a single timestep. Accepts a jax PRNG key for determinism."""
        pass

    def multiple_steps(self, x, v, n_steps: int = 1000, rng: Optional[np.random.Generator] = None):
        """Return trajectories of x and v, advanced by n_steps"""

        rng = rng or np.random.default_rng()

        xs, vs = [x], [v]

        for _ in range(n_steps):
            new_x, new_v = self.step(xs[-1], vs[-1], rng)

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)

    @partial(jax.jit, static_argnums=(0, 4))
    def multiple_steps_lax(self, key, x, v, n_steps: int = 1000):
        """
        Return trajectories of x and v, advanced by n_steps. Implemented using jax.lax.scan to allow jax.jit to produce
        efficient code.

        Note: requires that force_fxn be jax-transformable
        """

        def f(xv, key):
            x, v = xv
            xv_ = self.step_lax(key, x, v)
            return xv_, xv_

        keys = jax.random.split(key, n_steps)
        _, (xs, vs) = jax.lax.scan(f, (x, v), keys)

        return (
            jnp.concatenate((x[jnp.newaxis, :], xs)),
            jnp.concatenate((v[jnp.newaxis, :], vs)),
        )


class LangevinIntegrator(StochasticIntegrator):
    def __init__(self, force_fxn, masses, temperature, dt, friction):
        """BAOAB (https://arxiv.org/abs/1203.5428), rotated by half a timestep"""
        self.dt = dt
        self.masses = masses
        self.temperature = temperature
        ca, cb, cc = langevin_coefficients(temperature, dt, friction, masses)
        self.force_fxn = force_fxn

        # make masses, frictions, etc. (scalar or (N,)) shape-compatible with coordinates (vector or (N,3))
        # note: per-atom frictions allowed
        self.ca, self.cb, self.cc = np.expand_dims(ca, -1), np.expand_dims(cb, -1), np.expand_dims(cc, -1)

    def _step(self, x, v, noise):
        """Intended to match https://github.com/proteneer/timemachine/blob/37e60205b3ae3358d9bb0967d03278ed184b8976/timemachine/cpp/src/integrator.cu#L71-L74"""
        v_mid = v + self.cb * self.force_fxn(x)

        new_v = (self.ca * v_mid) + (self.cc * noise)
        new_x = x + 0.5 * self.dt * (v_mid + new_v)

        return new_x, new_v

    def step(self, x, v, rng):
        return self._step(x, v, rng.normal(size=x.shape))

    def step_lax(self, key, x, v):
        return self._step(x, v, jax.random.normal(key, x.shape))


class ConstraintSolver:
    def __init__(self, masses, constraint_groups, constraint_distances, max_iter: int = 15, tolerance: float = 1e-8):
        """Solve holonomic constraints using SHAKE (positions) and RATTLE (velocities).

        Parameters
        ----------
        masses : array-like
            Mass of each atom.
        constraint_groups : list of list of int
            Atom groups where the first atom is bonded to all subsequent atoms.
        constraint_distances : list of list of float
            Target bond distances for each group.
        max_iter : int
            Maximum solver iterations.
        tolerance : float
            Convergence tolerance.
        """
        assert len(constraint_distances) == len(constraint_groups), "must provide equal number of groups and distances"
        if any(len(group) > 7 for group in constraint_groups):
            raise ValueError("limited to 7 atoms per group")
        self.constraint_groups = constraint_groups
        self.constraint_distances = constraint_distances

        self._tol = tolerance
        self._max_iter = max_iter
        self._inv_masses = 1.0 / np.array(masses, dtype=np.float64)

    def apply_shake(self, x):
        """Apply SHAKE position constraint correction.

        Iteratively adjusts positions so that all bond distance constraints
        are satisfied to within tolerance. Uses the standard SHAKE algorithm
        with mass-weighted corrections based on squared-distance constraints.
        Does not mutate ``x``.

        Parameters
        ----------
        x : np.ndarray
            Drifted position array of shape (N, 3).

        Returns
        -------
        np.ndarray
            The corrected position array (new array).
        """
        if len(self.constraint_groups) == 0:
            return x.copy()
        x_ref = x.copy()
        for group, dists in zip(self.constraint_groups, self.constraint_distances):
            anchor = group[0]
            for i in range(self._max_iter):
                converged = True
                for atom, target_dist in zip(group[1:], dists):
                    delta_ref = x_ref[anchor] - x_ref[atom]
                    target_dist_2 = target_dist**2
                    delta = x[anchor] - x[atom]
                    dist2 = np.dot(delta, delta)
                    diff = dist2 - target_dist_2
                    if float(np.abs(diff)) <= self._tol * target_dist_2:
                        continue
                    inv_mi = self._inv_masses[anchor]
                    inv_mj = self._inv_masses[atom]
                    converged = False
                    denom = 2.0 * (inv_mi + inv_mj) * np.dot(delta, delta_ref)
                    if np.allclose(denom, 0.0):
                        continue
                    grad = diff / denom
                    x[anchor] -= grad * inv_mi * delta_ref
                    x[atom] += grad * inv_mj * delta_ref
                if converged:
                    break
        return x

    def apply_rattle(self, last_x, v):
        """Apply RATTLE velocity constraint correction.

        Adjusts velocities so that the time derivative of all position
        constraints is zero, i.e., ``(x[i] - x[j]) . (v[i] - v[j]) = 0``.
        This ensures constrained atoms move consistently. Their relative
        velocity along the bond direction is zero. Uses a mass-weighted
        correction along the position difference vector. Does not mutate ``v``.

        Parameters
        ----------
        last_x : np.ndarray
            Positions array of shape (N, 3) (SHAKE-corrected positions).
        v : np.ndarray
            Velocity array of shape (N, 3).

        Returns
        -------
        np.ndarray
            The corrected velocity array (new array).
        """
        if len(self.constraint_groups) == 0:
            return v.copy()

        v = v.copy()
        for group in self.constraint_groups:
            anchor = group[0]
            for _ in range(self._max_iter):
                converged = True
                for atom in group[1:]:
                    delta_x = last_x[anchor] - last_x[atom]
                    dist2 = np.dot(delta_x, delta_x)
                    inv_mi = self._inv_masses[anchor]
                    inv_mj = self._inv_masses[atom]

                    denom = (inv_mi + inv_mj) * dist2
                    if np.allclose(denom, 0.0):
                        continue
                    delta_v = v[anchor] - v[atom]
                    rv = np.dot(delta_x, delta_v)
                    if np.abs(rv) <= self._tol:
                        continue
                    converged = False
                    lam = rv / denom
                    v[anchor] -= lam * inv_mi * delta_x
                    v[atom] += lam * inv_mj * delta_x
                if converged:
                    break
        return v

    def apply_velocity_constraints(self, x, v):
        return self.apply_rattle(x, v)

    def apply_positional_constraints(self, x):
        return self.apply_shake(x)

    def solve(self, x, v):
        """Apply SHAKE to positions then RATTLE to velocities.

        Returns copies of the corrected position and velocity arrays.
        """
        if len(self.constraint_groups) == 0:
            return x.copy(), v.copy()
        v = self.apply_velocity_constraints(x, v)
        x = self.apply_positional_constraints(x)
        return x, v


class ConstrainedLangevinIntegrator(LangevinIntegrator):
    """Langevin integrator with SHAKE/RATTLE (https://doi.org/10.1016/0021-9991(83)90014-1) holonomic constraints.

    Constraint groups can contain up to 7 atoms. Each group specifies
    a list of bond distances (pairs of atom indices + target distance).
    """

    def __init__(
        self,
        force_fxn,
        masses,
        temperature,
        dt,
        friction,
        constraint_groups,
        constraint_distances,
        max_iter: int = 15,
        tolerance: float = 1e-5,
    ):
        """Langevin integrator with SHAKE/RATTLE holonomic constraints.

        Parameters
        ----------
        force_fxn : callable
            Force function returning (N, 3) array of forces.
        masses : array-like
            Masses for each atom. np.inf freezes the atom.
        temperature : float
            Temperature in Kelvin.
        dt : float
            Timestep in picoseconds.
        friction : float
            Collision rate in 1/ps.
        constraint_groups : list of list of int
            Groups of atoms where the first atom is bonded to all subsequent atoms.
        constraint_distances : list of list of float
            The lengths of the bonds implied by each constraint group.
        max_iter : int
            Maximum number of iterations for the constraint solver.
        tolerance : float
            The tolerance of the constraint solver.
        """

        super().__init__(force_fxn, masses, temperature, dt, friction)

        self._solver = ConstraintSolver(masses, constraint_groups, constraint_distances, max_iter, tolerance)

    def _step(self, x, v, noise):
        """BAOAB step with RATTLE constraint enforcement."""
        v_mid = v + self.cb * self.force_fxn(x)
        v_mid = self._solver.apply_velocity_constraints(x, v_mid)
        new_v = (self.ca * v_mid) + (self.cc * noise)
        preconstrained = x + 0.5 * self.dt * (v_mid + new_v)
        new_x = self._solver.apply_positional_constraints(preconstrained)
        # Adjust the velocities by the change in the positions
        # Ref: https://github.com/choderalab/integrator-benchmark/blob/bb307e6ebf476b652e62e41ae49730f530732da3/benchmark/integrators/langevin.py#L130-L133
        new_v += (new_x - preconstrained) / self.dt

        # Unclear if this is required. OpenMM excludes this step:
        # https://github.com/openmm/openmm/blob/6e864310520172a4d108d195bd6b15ca46238223/platforms/common/src/CommonKernels.cpp#L3189-L3194
        # new_v = self._solver.apply_velocity_constraints(new_x, new_v)

        return new_x, new_v

    def step_lax(self, key, x, v):
        """Not implemented. SHAKE/RATTLE constraints use numpy operations not compatible with jax.jit.

        This could be supported, but currently it's a low priority."""
        raise NotImplementedError(
            "ConstrainedLangevinIntegrator does not support step_lax/multiple_steps_lax. "
            "The SHAKE/RATTLE constraint solver uses numpy operations that are not JAX-transformable."
        )


class VelocityVerletIntegrator(Integrator):
    def __init__(self, force_fxn, masses, dt):
        """WARNING: `.step` makes 2x more calls to force_fxn per timestep than `.multiple_steps`"""
        self.dt = dt
        self.masses = masses[:, np.newaxis]  # TODO: cleaner way to handle (n_atoms,) vs. (n_atoms, 3) mismatch?
        self.force_fxn = force_fxn
        self.cb = self.dt / self.masses

    def step(self, x, v):
        """WARNING: makes 2 calls to force_fxn per timestep -- prefer `.multiple_steps` in most cases"""
        v_mid = float_to_fixed(v) + float_to_fixed((0.5 * self.cb) * self.force_fxn(x))
        fixed_x = float_to_fixed(x) + float_to_fixed(self.dt * fixed_to_float(v_mid))
        fixed_v = v_mid + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(fixed_x)))

        return fixed_to_float(fixed_x), fixed_to_float(fixed_v)

    def multiple_steps(self, x, v, n_steps=1000):
        # note: intermediate timesteps are staggered
        #    xs[0], vs[0] = x_0, v_0
        #    xs[1], vs[1] = x_2, v_{1.5}
        #    xs[2], vs[2] = x_3, v_{2.5}
        #    ...
        #    xs[T-1], vs[T-1] = x_T, v_{T-0.5}
        #    xs[T],   vs[T]   = x_T, v_T

        # note: reorders loop slightly to avoid ~n_steps extraneous calls to force_fxn
        x_fixed = float_to_fixed(x)
        v_fixed = float_to_fixed(v)

        zs = [(x_fixed, v_fixed)]
        # initialize traj
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))
        x_fixed = x_fixed + float_to_fixed(self.dt * fixed_to_float(v_fixed))

        # run n_steps-1 steps
        for t in range(n_steps - 1):
            v_fixed = v_fixed + float_to_fixed(self.cb * self.force_fxn(fixed_to_float(x_fixed)))
            x_fixed = x_fixed + float_to_fixed(self.dt * fixed_to_float(v_fixed))

            zs.append((x_fixed, v_fixed))

        # finalize traj
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))

        zs.append((x_fixed, v_fixed))

        xs = np.array([x for (x, _) in zs])
        vs = np.array([v for (_, v) in zs])
        return fixed_to_float(xs), fixed_to_float(vs)

    def _update_via_fori_loop(self, x, v, n_steps=1000):
        # initialize

        v_fixed = float_to_fixed(v) + float_to_fixed((0.5 * self.cb) * self.force_fxn(x))
        x_fixed = float_to_fixed(x) + float_to_fixed(self.dt * fixed_to_float(v_fixed))

        def velocity_verlet_loop_body(_, val):
            x_prev, v_prev = val

            v_fixed = v_prev + float_to_fixed(self.cb * self.force_fxn(fixed_to_float(x_prev)))
            x_fixed = x_prev + float_to_fixed(self.dt * fixed_to_float(v_fixed))
            return x_fixed, v_fixed

        # run n_steps - 1 steps
        x_fixed, v_fixed = jax.lax.fori_loop(0, n_steps - 1, velocity_verlet_loop_body, (x_fixed, v_fixed))

        # finalize
        v_fixed = v_fixed + float_to_fixed((0.5 * self.cb) * self.force_fxn(fixed_to_float(x_fixed)))

        return fixed_to_float(x_fixed), fixed_to_float(v_fixed)


def _fori_steps(x0, v0, key0, grad_fn, num_steps, dt, ca, cbs, ccs):
    def body_fn(_, val):
        # BAOAB integrator
        x_t, v_t, key = val
        du_dx = grad_fn(x_t)[0]
        v_mid = v_t + cbs * du_dx
        noise = jrandom.normal(key, v_t.shape)
        _, sub_key = jrandom.split(key)
        v_t = ca * v_mid + ccs * noise
        x_t += 0.5 * dt * (v_mid + v_t)
        return x_t, v_t, sub_key

    return jax.lax.fori_loop(0, num_steps, body_fn, (x0, v0, key0))


def simulate(x0, U_fn, temperature, masses, steps_per_batch, num_batches, num_workers, seed=None):
    """
    Simulate a gas-phase system using a reference jax implementation.

    Parameters
    ----------

    x0: (N,3) np.ndarray
        initial coordinates

    U_fn: function
        Potential energy function

    temperature: float
        Temperature in Kelvin

    steps_per_batch: int
        number of steps we run for each batch

    num_batches: int
        number of batches we run

    num_workers: int
        How many jobs to run in parallel

    seed: int
        used for the random number generated

    Returns
    -------

    """
    dt = 1.5e-3
    friction = 1.0
    ca, cbs, ccs = langevin_coefficients(temperature, dt, friction, masses)
    cbs = np.expand_dims(cbs * -1, axis=-1)
    ccs = np.expand_dims(ccs, axis=-1)

    grad_fn = jax.jit(jax.grad(U_fn, argnums=(0,)))
    U_fn = jax.jit(U_fn)

    if seed is None:
        seed = int(time.time())

    @jax.jit
    def multiple_steps(x0, v0, key0):
        return _fori_steps(x0, v0, key0, grad_fn, steps_per_batch, dt, ca, cbs, ccs)

    v0 = np.zeros_like(x0)

    # jitting a pmap will result in a warning about inefficient data movement
    batched_multiple_steps_fn = jax.pmap(multiple_steps)

    xs_t = np.array([x0] * num_workers)
    vs_t = np.array([v0] * num_workers)
    keys_t = np.array([jrandom.PRNGKey(seed + idx) for idx in range(num_workers)])

    all_xs = []
    all_vs = []

    for batch_step in range(num_batches):
        #                                             [B,N,3][B,N,3][B,2]
        xs_t, vs_t, keys_t = batched_multiple_steps_fn(xs_t, vs_t, keys_t)
        all_xs.append(xs_t)
        all_vs.append(vs_t)

    # result has shape [num_workers, num_batches, num_atoms, num_dimensions]
    return np.transpose(np.array(all_xs), axes=[1, 0, 2, 3]), np.transpose(np.array(all_vs), axes=[1, 0, 2, 3])
