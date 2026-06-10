# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2026 Justin Gullingsrud
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


class ConstrainedVelocityVerletIntegrator(Integrator):
    """Reference RATTLE velocity-Verlet (NVE) integrator with SHAKE/RATTLE
    distance constraints.

    This is a pure-numpy reference for the GPU
    ``custom_ops.ConstrainedVelocityVerletIntegrator``: it freezes a set of
    interatomic distances while integrating the remaining degrees of freedom,
    and (absent a thermostat) conserves the total energy up to integration
    error. It is intended for cross-checking the GPU integrator and for
    measuring energy drift as a function of timestep.

    Parameters
    ----------
    force_fxn:
        Callable mapping coordinates (N, 3) to forces (N, 3), i.e. -dU/dx.
    masses:
        Per-atom masses (N,). np.inf freezes the particle.
    dt:
        Timestep.
    constraint_pairs:
        (M, 2) integer array of constrained atom pairs.
    constraint_lengths:
        (M,) target distances for each pair.
    tol:
        Convergence tolerance for the iterative SHAKE/RATTLE projections.
    max_iters:
        Maximum number of Gauss-Seidel sweeps per projection.
    """

    def __init__(self, force_fxn, masses, dt, constraint_pairs, constraint_lengths, tol=1e-10, max_iters=100):
        self.dt = dt
        self.masses = np.asarray(masses, dtype=np.float64)
        self.inv_mass = np.where(np.isinf(self.masses), 0.0, 1.0 / self.masses)
        self.force_fxn = force_fxn
        self.pairs = np.asarray(constraint_pairs, dtype=int).reshape(-1, 2)
        self.r0 = np.asarray(constraint_lengths, dtype=np.float64).reshape(-1)
        self.tol = tol
        self.max_iters = max_iters
        # Atoms touched by any constraint; their velocities are overwritten by
        # the constraint-consistent value after the SHAKE drift.
        self.constrained_atoms = np.unique(self.pairs) if self.pairs.size else np.array([], dtype=int)

    def _shake(self, x, x_ref):
        """Project drifted positions x back onto the constraint manifold, using
        the pre-drift positions x_ref to build the constraint directions."""
        x = np.array(x, dtype=np.float64)
        winv = self.inv_mass
        for _ in range(self.max_iters):
            done = True
            for k in range(len(self.pairs)):
                i, j = self.pairs[k]
                d = x[i] - x[j]
                r2 = float(d @ d)
                target = self.r0[k] ** 2
                diff = r2 - target
                if abs(diff) > self.tol * target:
                    done = False
                    dref = x_ref[i] - x_ref[j]
                    denom = 2.0 * (winv[i] + winv[j]) * float(d @ dref)
                    if denom == 0.0:
                        continue
                    g = diff / denom
                    x[i] = x[i] - g * winv[i] * dref
                    x[j] = x[j] + g * winv[j] * dref
            if done:
                break
        return x

    def _rattle(self, x, v):
        """Project velocities so that every constrained pair satisfies r . v = 0."""
        v = np.array(v, dtype=np.float64)
        winv = self.inv_mass
        for _ in range(self.max_iters):
            done = True
            for k in range(len(self.pairs)):
                i, j = self.pairs[k]
                d = x[i] - x[j]
                dv = v[i] - v[j]
                rv = float(d @ dv)
                r2 = float(d @ d)
                denom = (winv[i] + winv[j]) * r2
                if denom == 0.0:
                    continue
                if abs(rv) > self.tol:
                    done = False
                    lam = rv / denom
                    v[i] = v[i] - lam * winv[i] * d
                    v[j] = v[j] + lam * winv[j] * d
            if done:
                break
        return v

    def step(self, x, v):
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        half_cb = 0.5 * self.dt * self.inv_mass[:, None]

        # First half kick using the force at the current positions.
        v = v + half_cb * self.force_fxn(x)

        # Full drift, then SHAKE. The constraint-consistent velocity update for
        # the cluster atoms is (x_constrained - x_ref) / dt = v(t + dt/2).
        x_ref = x
        x_new = self._shake(x_ref + self.dt * v, x_ref)
        v = np.array(v)
        ca = self.constrained_atoms
        if len(ca):
            v[ca] = (x_new[ca] - x_ref[ca]) / self.dt

        # Second half kick using the force at the new positions, then RATTLE.
        v = v + half_cb * self.force_fxn(x_new)
        v = self._rattle(x_new, v)
        return x_new, v

    def multiple_steps(self, x, v, n_steps=1000):
        x = np.asarray(x, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        # Project the initial state onto the constraint manifold.
        x = self._shake(x, x)
        v = self._rattle(x, v)
        xs = [x]
        vs = [v]
        for _ in range(n_steps):
            x, v = self.step(x, v)
            xs.append(x)
            vs.append(v)
        return np.array(xs), np.array(vs)


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
