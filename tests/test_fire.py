# Copyright 2026, Forrest York
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

import jax
import numpy as np
import pytest

from tmd._vendored.fire import FireDescentState, fire_descent
from tmd.md.minimizer import FireMinimizationConfig, MinimizationWarning, fire_minimize, local_minimize

pytestmark = [pytest.mark.nogpu]

IDENTITY = lambda x: x


def const_force(fvec):
    """Force field returning a constant force fvec regardless of positions."""
    fvec = np.asarray(fvec, dtype=np.float32)
    return lambda R: np.full_like(R, fvec, dtype=np.float32)


def state_at(R, V, F, dt, alpha, n_pos=0, n_neg=0, n_steps=0):
    """Hand-build a FireDescentState with numpy scalar/int fields."""
    return FireDescentState(
        np.asarray(R, dtype=np.float32),
        np.asarray(V, dtype=np.float32),
        np.asarray(F, dtype=np.float32),
        np.float32(dt),
        np.float32(alpha),
        np.int32(n_pos),
        np.int32(n_neg),
        np.int32(n_steps),
    )


def test_quadratic_bowl_converges():
    """F = -x: the minimizer drives the system toward the minimum at the origin."""
    rng = np.random.default_rng(0)
    R0 = (rng.normal(size=(6, 3)) * 0.5).astype(np.float32)
    init, apply = fire_descent(
        lambda R: -R, IDENTITY, np.ones(6, dtype=np.float32), dt_start=1e-3, dt_max=1e-2, n_min=10
    )
    s = init(R0)
    e0 = 0.5 * float(np.sum(R0**2))
    for _ in range(3000):
        s = apply(s)
    ef = 0.5 * float(np.sum(s.position**2))
    assert ef < e0
    assert ef < 1e-6


def test_velocity_verlet_constant_force():
    """Constant force, zero start, unit mass: R = R0 + 0.5*F*(n*dt)^2 (mixing is a no-op when v_half || F)."""
    dt = 1e-4
    n = 6
    F0 = np.array([2.0, 0.0, -1.0], dtype=np.float32)
    R0 = np.array([[0.3, -0.1, 0.2]], dtype=np.float32)
    init, apply = fire_descent(
        const_force(F0), IDENTITY, np.array([1.0], dtype=np.float32), dt_start=dt, dt_max=1e-3, n_min=20
    )
    s = init(R0)
    for _ in range(n):
        s = apply(s)
    pred = R0 + 0.5 * F0 * (n * dt) ** 2
    np.testing.assert_allclose(s.position, pred, rtol=1e-4, atol=1e-9)


def test_mass_scaling():
    """Mass x4 under the same constant force yields 1/4 the displacement (confirms the integrator uses F/m)."""
    dt = 1e-4
    n = 6
    F0 = np.array([2.0, 0.0, -1.0], dtype=np.float32)
    R0 = np.array([[0.3, -0.1, 0.2]], dtype=np.float32)
    _, apply1 = fire_descent(
        const_force(F0), IDENTITY, np.array([1.0], dtype=np.float32), dt_start=dt, dt_max=1e-3, n_min=20
    )
    _, apply4 = fire_descent(
        const_force(F0), IDENTITY, np.array([4.0], dtype=np.float32), dt_start=dt, dt_max=1e-3, n_min=20
    )
    s1 = state_at(R0, np.zeros_like(R0), np.full_like(R0, F0), dt, 0.1)
    s4 = state_at(R0, np.zeros_like(R0), np.full_like(R0, F0), dt, 0.1)
    for _ in range(n):
        s1 = apply1(s1)
        s4 = apply4(s4)
    d1 = s1.position - R0
    d4 = s4.position - R0
    np.testing.assert_allclose(d4, 0.25 * d1, rtol=1e-3, atol=1e-12)


def test_backtracking_uphill():
    """P = F.v <= 0: back off a half step, zero velocity, shrink dt, reset alpha, bump n_neg."""
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dt = 2e-4
    f_dec = 0.5
    alpha_start = 0.2
    init, apply = fire_descent(
        const_force(f0),
        IDENTITY,
        np.array([1.0], dtype=np.float32),
        dt_start=dt,
        dt_max=1e-3,
        dt_min=2e-7,
        f_dec=f_dec,
        alpha_start=alpha_start,
        n_min=20,
    )
    F = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    V = -F  # P = F.v < 0
    R0 = np.array([[0.3, -0.1, 0.2]], dtype=np.float32)
    s = state_at(R0, V, F, dt, 0.1, n_pos=0, n_neg=0, n_steps=100)  # past delay
    out = apply(s)
    dt_shrink = dt * f_dec  # >= dt_min, so it shrinks
    np.testing.assert_allclose(float(out.dt), dt_shrink, rtol=1e-3)
    np.testing.assert_allclose(float(out.alpha), alpha_start, rtol=1e-3)
    assert int(out.n_neg) == 1
    assert int(out.n_pos) == 0
    # Leading displacement is the back-off -0.5*dt_shrink*V (V=-F, so +F); reintegration drift is O(dt^2).
    np.testing.assert_allclose(out.position - R0, (0.5 * dt_shrink * f0)[None, :], rtol=1e-2, atol=1e-9)
    # Velocity was zeroed before the reintegration kick, so final |v| stays O(dt) and points along +F.
    assert float(np.linalg.norm(out.velocity)) < 1e-3
    assert out.velocity[0, 0] > 0


def test_dt_min_floor():
    """When dt*f_dec would fall below dt_min, dt is held at its current value (floor), but alpha still resets."""
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dt = 1e-7  # dt * f_dec = 5e-8 < dt_min = 2e-7
    init, apply = fire_descent(
        const_force(f0),
        IDENTITY,
        np.array([1.0], dtype=np.float32),
        dt_start=dt,
        dt_min=2e-7,
        f_dec=0.5,
        alpha_start=0.2,
        n_min=20,
    )
    F = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    s = state_at(np.array([[0.3, -0.1, 0.2]], dtype=np.float32), -F, F, dt, 0.1, n_steps=100)
    out = apply(s)
    np.testing.assert_allclose(float(out.dt), dt, rtol=1e-3)  # not shrunk below floor
    np.testing.assert_allclose(float(out.alpha), 0.2, rtol=1e-3)  # reset to alpha_start
    assert int(out.n_neg) == 1


def test_initial_delay():
    """During the initial delay (n_steps < n_min, P <= 0): dt and alpha are untouched, but backtracking still runs."""
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dt = 2e-4
    alpha_start = 0.3  # deliberately different from the state's alpha so a reset would be observable
    init, apply = fire_descent(
        const_force(f0),
        IDENTITY,
        np.array([1.0], dtype=np.float32),
        dt_start=dt,
        dt_min=2e-7,
        f_dec=0.5,
        alpha_start=alpha_start,
        n_min=20,
        initial_delay=True,
    )
    F = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    R0 = np.array([[0.3, -0.1, 0.2]], dtype=np.float32)
    s = state_at(R0, -F, F, dt, 0.1, n_steps=0)
    out = apply(s)
    np.testing.assert_allclose(float(out.dt), dt, rtol=1e-3)  # unchanged (delay)
    np.testing.assert_allclose(float(out.alpha), 0.1, rtol=1e-3)  # unchanged (NOT reset to alpha_start)
    assert int(out.n_neg) == 1
    # backtracking still applied: leading displacement 0.5*dt*F
    np.testing.assert_allclose(out.position - R0, (0.5 * dt * f0)[None, :], rtol=1e-2, atol=1e-9)


def test_growth_gating():
    """P > 0 with n_pos == n_min: dt grows toward min(dt*f_inc, dt_max) and alpha decays by f_alpha."""
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dt = 5e-5
    f_inc = 1.1
    dt_max = 1e-3
    f_alpha = 0.99
    init, apply = fire_descent(
        const_force(f0),
        IDENTITY,
        np.array([1.0], dtype=np.float32),
        dt_start=dt,
        dt_max=dt_max,
        f_inc=f_inc,
        f_alpha=f_alpha,
        n_min=20,
    )
    F = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    V = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)  # P > 0
    s = state_at(np.array([[0.3, -0.1, 0.2]], dtype=np.float32), V, F, dt, 0.1, n_pos=20, n_neg=0, n_steps=50)
    out = apply(s)
    np.testing.assert_allclose(float(out.dt), min(dt * f_inc, dt_max), rtol=1e-3)
    np.testing.assert_allclose(float(out.alpha), 0.1 * f_alpha, rtol=1e-3)
    assert int(out.n_pos) == 21
    assert int(out.n_neg) == 0


def test_growth_gated_until_n_min():
    """P > 0 but n_pos < n_min: dt/alpha are held (growth is gated until N_delay consecutive P>0 steps)."""
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dt = 5e-5
    init, apply = fire_descent(
        const_force(f0),
        IDENTITY,
        np.array([1.0], dtype=np.float32),
        dt_start=dt,
        dt_max=1e-3,
        f_inc=1.1,
        f_alpha=0.99,
        n_min=20,
    )
    F = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    V = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    s = state_at(np.array([[0.3, -0.1, 0.2]], dtype=np.float32), V, F, dt, 0.1, n_pos=5, n_neg=0, n_steps=50)
    out = apply(s)
    np.testing.assert_allclose(float(out.dt), dt, rtol=1e-3)  # not grown
    np.testing.assert_allclose(float(out.alpha), 0.1, rtol=1e-3)  # not decayed


def test_stuck_early_exit_warns():
    """fire_minimize breaks early (and warns) once n_neg exceeds n_neg_max on a flat potential (P <= 0 every step)."""
    x0 = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    calls = {"n": 0}

    def du_dx(x):
        calls["n"] += 1
        return np.zeros_like(x)

    cfg = FireMinimizationConfig(n_steps=100000, n_neg_max=5)
    with pytest.warns(MinimizationWarning, match="stopped early"):
        # pass masses so the only warning is the stuck early-exit
        out = fire_minimize(x0, du_dx, cfg, masses=np.array([1.0], dtype=np.float32))
    # It returned without running anywhere near n_steps force evaluations.
    assert calls["n"] < 100
    assert np.all(np.isfinite(out))


def test_dmax_caps_displacement():
    """dmax clamps |dR| of the position update per atom: a huge velocity (V=1000, dt=1e-3 =>
    ~1 nm/step uncapped) is limited to dmax per step."""
    dmax = 1e-4
    dt = 1e-3
    f0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, apply = fire_descent(
        const_force(f0), IDENTITY, np.array([1.0], dtype=np.float32), dt_start=dt, dt_max=dt, n_min=1_000_000, dmax=dmax
    )
    R0 = np.zeros((1, 3), dtype=np.float32)
    s = state_at(R0, np.array([[1000.0, 0.0, 0.0]], dtype=np.float32), np.full_like(R0, f0), dt, 0.1)
    prev = np.asarray(s.position).copy()
    for _ in range(10):
        s = apply(s)
        disp = np.linalg.norm(np.asarray(s.position) - prev, axis=-1)
        assert np.all(disp <= dmax * 1.01)  # 1% float slack
        prev = np.asarray(s.position)


def test_dmax_inactive_when_large():
    """A dmax above any real step size leaves the trajectory bit-identical to the default (inf)."""
    R0 = np.array([[0.3, -0.1, 0.2]], dtype=np.float32)
    F0 = np.array([[2.0, 0.0, -1.0]], dtype=np.float32)
    kwargs = dict(dt_start=1e-4, dt_max=1e-3, n_min=20)
    _, apply_default = fire_descent(const_force(F0), IDENTITY, np.array([1.0], dtype=np.float32), **kwargs)
    _, apply_big = fire_descent(const_force(F0), IDENTITY, np.array([1.0], dtype=np.float32), dmax=1e6, **kwargs)
    s1 = state_at(R0, np.array([[5.0, 0.0, 0.0]], dtype=np.float32), F0.copy(), 1e-4, 0.1)
    s2 = state_at(R0, np.array([[5.0, 0.0, 0.0]], dtype=np.float32), F0.copy(), 1e-4, 0.1)
    for _ in range(6):
        s1 = apply_default(s1)
        s2 = apply_big(s2)
    np.testing.assert_array_equal(np.asarray(s2.position), np.asarray(s1.position))


def test_stall_detector_fires_on_no_progress():
    """A constant force keeps max|F| flat even though P > 0 every step (so n_neg never fires);
    the max|force|-improvement detector must still exit early with a warning."""
    x0 = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    calls = {"n": 0}

    def du_dx(x):
        calls["n"] += 1
        return -np.ones_like(x)

    cfg = FireMinimizationConfig(n_steps=100000, n_neg_max=10_000_000, stall_patience=50)
    with pytest.warns(MinimizationWarning, match="did not improve"):
        out = fire_minimize(x0, du_dx, cfg, masses=np.array([1.0], dtype=np.float32))
    assert calls["n"] < 200
    assert np.all(np.isfinite(out))

    # With the detector disabled (stall_patience=0) the same run completes silently.
    cfg = FireMinimizationConfig(n_steps=120, n_neg_max=10_000_000, stall_patience=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = fire_minimize(x0, du_dx, cfg, masses=np.array([1.0], dtype=np.float32))
    assert calls["n"] >= 120
    assert np.all(np.isfinite(out))


def test_jittable_via_jnp_swap(monkeypatch):
    """fire.py uses only numpy function-API calls that jax.numpy also provides: swapping the
    module's `numpy as np` for `jax.numpy` makes init_fn/apply_fn trace cleanly under jax.jit,
    and the jitted run matches the eager numpy run."""
    import jax.numpy as jnp

    import tmd._vendored.fire as fire_module

    R0 = np.array([[0.3, -0.1, 0.2], [1.0, 0.0, 0.5]], dtype=np.float32)
    masses = np.array([1.0, 2.0], dtype=np.float32)
    kwargs = dict(dt_start=1e-3, dt_max=1e-2, n_min=20)

    n_steps = 4
    init, apply = fire_descent(lambda R: -0.7 * R, IDENTITY, masses, **kwargs)
    s = init(R0)
    for _ in range(n_steps):
        s = apply(s)

    # np -> jnp swap: init_fn/apply_fn resolve `np` from the module globals at call time.
    monkeypatch.setattr(fire_module, "np", jnp)
    jinit, japply = fire_module.fire_descent(lambda R: -0.7 * R, IDENTITY, masses, **kwargs)
    js = jinit(jnp.asarray(R0))
    jstep = jax.jit(japply)
    for _ in range(n_steps):
        js = jstep(js)

    for name in ("position", "velocity", "force"):
        np.testing.assert_allclose(np.asarray(getattr(js, name)), getattr(s, name), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(float(np.asarray(js.dt)), float(s.dt), rtol=1e-5)
    np.testing.assert_allclose(float(np.asarray(js.alpha)), float(s.alpha), rtol=1e-5)
    assert int(np.asarray(js.n_steps)) == int(s.n_steps)


def test_energy_monotonicity_convex():
    """On a smooth convex (harmonic) potential no accepted step raises the energy above the initial energy."""
    rng = np.random.default_rng(1)
    R0 = (rng.normal(size=(8, 3)) * 0.4).astype(np.float32)
    k = 0.9
    init, apply = fire_descent(
        lambda R: -k * R, IDENTITY, np.ones(8, dtype=np.float32), dt_start=1e-3, dt_max=1e-2, n_min=10
    )

    def energy(pos):
        return 0.5 * k * float(np.sum(pos**2))

    s = init(R0)
    e0 = energy(R0)
    energies = [e0]
    for _ in range(1500):
        s = apply(s)
        energies.append(energy(s.position))
    energies = np.array(energies)
    assert energies[-1] < energies[0]
    # Allow only float noise above the initial energy: FIRE never climbs past its start on a convex bowl.
    assert np.max(energies) <= energies[0] * (1.0 + 1e-4) + 1e-9


def test_fire_minimize_warns_without_masses():
    """fire_minimize emits MinimizationWarning when masses are omitted."""
    x0 = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    with pytest.warns(MinimizationWarning, match="unit masses"):
        fire_minimize(x0, lambda x: x, FireMinimizationConfig(50))


def test_fire_minimize_no_warn_with_masses():
    """fire_minimize stays quiet (no MinimizationWarning) when per-atom masses are supplied."""
    x0 = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    masses = np.array([1.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (incl. a spurious stuck-exit) raises
        out = fire_minimize(x0, lambda x: x, FireMinimizationConfig(300), masses=masses)
    assert np.all(np.isfinite(out))


def test_local_minimize_slices_full_system_masses():
    """local_minimize accepts full-system masses (one per atom) and slices them to the free atoms."""
    n_atoms = 5
    free_idxs = [0, 2, 4]
    x0 = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32
    )
    box = np.eye(3, dtype=np.float32) * 10.0
    # pure-numpy harmonic energy/grad -> keeps this test CPU-only (local_minimize itself needs no GPU)
    val_and_grad_fn = lambda x: (0.5 * float(np.sum(x**2)), x.copy())
    masses = np.full(n_atoms, 2.0, dtype=np.float32)
    cfg = FireMinimizationConfig(n_steps=800, dt_start=1e-2, dt_max=1e-1, n_min=10)

    x_opt = local_minimize(x0, box, val_and_grad_fn, free_idxs, [cfg], verbose=False, masses=masses)

    frozen = [i for i in range(n_atoms) if i not in free_idxs]
    # frozen atoms untouched, free atoms moved toward the minimum
    np.testing.assert_array_equal(x_opt[frozen], x0[frozen])
    assert np.linalg.norm(x_opt[free_idxs] - x0[free_idxs]) > 1e-4
    assert 0.5 * float(np.sum(x_opt[free_idxs] ** 2)) < 0.5 * float(np.sum(x0[free_idxs] ** 2))

    # A wrong-length masses array is rejected up front.
    with pytest.raises(AssertionError, match="one entry per atom"):
        local_minimize(x0, box, val_and_grad_fn, free_idxs, [cfg], verbose=False, masses=np.full(n_atoms + 1, 2.0))
