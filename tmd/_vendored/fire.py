# Refer to License below. Original Code taken from https://github.com/google/jax-md/blob/master/jax_md
#
# Modified to implement the FIRE 2.0 algorithm: Velocity Verlet integration,
# backtracking (half-step correction of uphill motion), and per-atom masses,
# as described in Guenole et al., "Assessment and optimization of the fast
# inertial relaxation engine (fire) for energy minimization in atomistic
# simulations and its implementation in lammps", arXiv:1908.02038.

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code to minimize the energy of a system.

Implements the FIRE 2.0 ("Fast Inertial Relaxation Engine") minimizer [2], an
upgrade of the original FIRE algorithm [1] that uses:
  * Velocity Verlet time integration (paper Algorithm 6) with the FIRE mixing
    step applied on every step,
  * backtracking: a half-step correction of uphill motion when P = F.v <= 0
    (paper Algorithm 2, line 28),
  * per-atom masses (the integrator uses F/m),
  * a dt_min floor, an initial delay, and a stuck-detection counter,
  * an optional per-atom displacement cap per step (LAMMPS `dmax`; cf. paper
    section 5.3, which uses dmax to stabilize systems starting far from a
    minimum and prevents wall tunneling in stiff repulsive regions).

`apply_fn` uses only numpy function-API calls (np.where / np.minimum / np.sum /
np.sqrt / np.dot / np.logical_and) that have jax.numpy counterparts, and no Python
if/else: replacing `import numpy as np` with `import jax.numpy as np` yields an
implementation that traces cleanly under jax.jit. The state is a stdlib NamedTuple
(a JAX pytree).

Structure follows JAX optimizers: returns (init_fn, apply_fn).
  init_fn: takes positions [n, dim], returns the initial state.
  apply_fn: takes a state, returns the state after one step.

[1] Bitzek et al., "Structural relaxation made simple.", PRL 97 (2006) 170201.
[2] Guenole et al., arXiv:1908.02038.
"""
from typing import NamedTuple

import numpy as np


class FireDescentState(NamedTuple):
    """State of the FIRE 2.0 minimizer.

    Attributes:
      position: particle positions, ndarray [n, spatial_dimension].
      velocity: particle velocities, ndarray [n, spatial_dimension].
      force: force on particles, ndarray [n, spatial_dimension].
      dt: current step size (float).
      alpha: current momentum/mixing coefficient (float).
      n_pos: number of consecutive steps with P > 0.
      n_neg: number of consecutive steps with P <= 0.
      n_steps: total number of steps taken so far.
    """

    position: np.ndarray
    velocity: np.ndarray
    force: np.ndarray
    dt: float
    alpha: float
    n_pos: int
    n_neg: int
    n_steps: int


def fire_descent(
    force,
    shift_fn,
    masses,
    dt_start: float = 1e-5,
    dt_max: float = 1e-4,
    dt_min: float = 2e-7,
    n_min: float = 20,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    initial_delay: bool = True,
    dmax: float = np.inf,
):
    """Defines the FIRE 2.0 minimization.

    Args:
      force: f(positions [n, dim]) -> forces [n, dim].
      shift_fn: transforms the updated positions (e.g. PBC wrapping) or is the
        identity; called once per step as shift_fn(R_new).
      masses: per-atom masses, ndarray [n]. The integrator uses F/m.
      dt_start: initial step size.
      dt_max: maximum step size.
      dt_min: minimum step size (floor for the shrink).
      n_min: number of consecutive P>0 steps required before dt grows / alpha
        decays (paper N_delay); also the length of the initial delay.
      f_inc: factor by which dt is increased.
      f_dec: factor by which dt is decreased.
      alpha_start: initial momentum/mixing coefficient.
      f_alpha: factor by which alpha is decreased.
      initial_delay: if True, skip the dt-shrink/alpha-reset for the first
        n_min steps (velocities start at 0 so P=0 initially).
      dmax: optional cap on the per-atom displacement per step (same length
        units as positions). Clamps |dR| of the position update only (the
        LAMMPS "dmax" convention); velocities are untouched. Defaults to inf
        (no cap). Prevents wall tunneling when dt*|v| would jump an atom into
        a stiff repulsive region.

    Returns:
      (init_fn, apply_fn) as described in the module docstring.
    """
    # Static (concrete) configuration; float() keeps the values jit-compat-friendly the
    # way the original static_cast did, while remaining jnp-compatible.
    (dt_start, dt_max, dt_min, n_min, f_inc, f_dec, alpha_start, f_alpha) = (
        float(x) for x in (dt_start, dt_max, dt_min, n_min, f_inc, f_dec, alpha_start, f_alpha)
    )
    dmax = float(dmax)
    inv_masses = 1 / np.asarray(masses)

    def init_fn(R: np.ndarray, **kwargs) -> FireDescentState:
        V = np.zeros_like(R)
        # Pin the scalar state to the array dtype so a float32 pipeline stays float32
        # (under numpy, np.where on python floats would otherwise promote to float64).
        dt = np.asarray(dt_start, dtype=R.dtype)
        alpha = np.asarray(alpha_start, dtype=R.dtype)
        n_pos = np.zeros((), np.int32)
        n_neg = np.zeros((), np.int32)
        n_steps = np.zeros((), np.int32)
        F = force(R, **kwargs)
        return FireDescentState(R, V, F, dt, alpha, n_pos, n_neg, n_steps)

    def apply_fn(state: FireDescentState, **kwargs) -> FireDescentState:
        R, V, F, dt, alpha, n_pos, n_neg, n_steps = state

        # Power at the START of the step (paper Algorithm 2, line 8): stored F, v.
        P = np.sum(F * V)
        pos = P > 0
        neg = P <= 0

        # --- Adaptive step size / momentum (paper Algorithm 2) ---
        grow = np.logical_and(pos, n_pos + 1 > n_min)
        dt_grow = np.where(grow, np.minimum(dt * f_inc, dt_max), dt)
        alpha_grow = np.where(grow, alpha * f_alpha, alpha)

        in_delay = np.logical_and(initial_delay, n_steps < n_min)
        shrink = np.logical_and(neg, np.logical_not(in_delay))
        dt_shrink = np.where(np.logical_and(shrink, dt * f_dec >= dt_min), dt * f_dec, dt)
        alpha_shrink = np.where(shrink, alpha_start, alpha)

        dt_new = np.where(pos, dt_grow, dt_shrink)
        alpha_new = np.where(pos, alpha_grow, alpha_shrink)

        # --- half-step correction of uphill motion (Alg. 2, line 28) ---
        # Uses the possibly-shrunk dt, per the algorithm's ordering.
        R = np.where(neg, R - 0.5 * dt_shrink * V, R)
        V = np.where(pos, V, 0.0)  # zero the velocity on uphill steps

        # --- Velocity Verlet integration + FIRE mixing (paper Algorithm 6), F/m ---
        V_half = V + 0.5 * dt_new * F * inv_masses[:, np.newaxis]
        V_norm = np.sqrt(np.sum(V_half**2))
        F_norm = np.sqrt(np.sum(F**2) + 1e-6)  # epsilon guard
        V_half = (1 - alpha_new) * V_half + alpha_new * F * (V_norm / F_norm)

        # --- Per-atom displacement cap per step (LAMMPS "dmax") -----------------
        # Clamps the position update only; v is left untouched (LAMMPS convention).
        # With dmax=inf the scale is exactly 1, so the default path is unchanged.
        dR = dt_new * V_half
        dR_norm = np.sqrt(np.sum(dR**2, axis=-1, keepdims=True))
        dR = dR * np.minimum(1.0, dmax / (dR_norm + 1e-12))

        R = shift_fn(R + dR, **kwargs)
        F = force(R, **kwargs)
        V = V_half + 0.5 * dt_new * F * inv_masses[:, np.newaxis]

        # --- Counters ---
        n_pos = np.where(pos, n_pos + 1, 0)
        n_neg = np.where(pos, 0, n_neg + 1)
        n_steps = n_steps + 1

        return FireDescentState(R, V, F, dt_new, alpha_new, n_pos, n_neg, n_steps)

    return init_fn, apply_fn
