# Copyright 2026 Justin Gullingsrud
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

"""Energy-conservation diagnostic for the constrained velocity-Verlet integrator.

Builds a small rigid-water + methane system in which all X-H bonds are held by
SHAKE/RATTLE constraints while the methane H-C-H angles remain flexible, so that
there is a genuine kinetic <-> potential energy exchange to conserve. The system
is integrated as an isolated NVE ensemble (no thermostat) with both the GPU
``custom_ops.ConstrainedVelocityVerletIntegrator`` and the pure-numpy reference
``tmd.integrator.ConstrainedVelocityVerletIntegrator`` over a sweep of
timesteps, reporting the total-energy drift for each. The GPU and reference
trajectories are also cross-checked against each other at a representative dt.

For contrast, the same system is also integrated with the ordinary
(unconstrained) ``custom_ops.VelocityVerletIntegrator``, replacing the rigid
X-H bonds and the rigid-water bend with stiff harmonic terms. The high-frequency
X-H stretches that the constraints remove are precisely what limit the
unconstrained timestep, so this shows directly how much the constraints buy in
energy conservation.

This is groundwork for deciding whether a 4 fs timestep is acceptable.
"""

import csv
from pathlib import Path

import numpy as np

from tmd import lib
from tmd.constants import BOLTZ
from tmd.fe import constraints as cst
from tmd.integrator import ConstrainedVelocityVerletIntegrator as RefIntegrator
from tmd.potentials import HarmonicAngle, HarmonicBond

PRECISION = np.float64
TEMPERATURE = 300.0  # K, only used to seed initial velocities
BOX = np.eye(3) * 5.0
TOTAL_TIME_PS = 5.0  # total simulated time per run, picoseconds
SAMPLE_INTERVAL_PS = 0.02  # energy sampling cadence, picoseconds
DT_SWEEP_FS = [1.0, 2.0, 3.0, 4.0]
RESULTS_CSV = Path(__file__).with_name("constrained_verlet_energy_check_results.csv")

# Realistic stiff bonded force constants for the unconstrained reference, so
# that the X-H stretches vibrate at their true (timestep-limiting) frequency.
K_OH = 462750.0  # kJ/mol/nm^2
K_CH = 376560.0  # kJ/mol/nm^2
K_HOH = 836.8  # kJ/mol/rad^2 (water bend)


def build_system():
    """Rigid water (O + 2H) and methane (C + 4H) with flexible methane angles.

    Returns the masses, coordinates, the bonded terms needed to derive the
    constraints, and the flexible H-C-H angle terms used as the only potential.
    """
    masses = np.array(
        [15.9994, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008, 1.008], dtype=np.float64
    )

    r_oh = 0.09572
    theta_w = np.deg2rad(104.52)
    half = theta_w / 2.0
    O = np.array([0.0, 0.0, 0.0])
    H1 = r_oh * np.array([np.sin(half), np.cos(half), 0.0])
    H2 = r_oh * np.array([-np.sin(half), np.cos(half), 0.0])

    r_ch = 0.109
    C = np.array([2.0, 2.0, 2.0])
    dirs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Hs = C + r_ch * dirs

    x0 = np.vstack([O, H1, H2, C, Hs])

    bond_idxs = np.array([[0, 1], [0, 2], [3, 4], [3, 5], [3, 6], [3, 7]], dtype=np.int32)
    bond_params = np.array([[1000.0, r_oh]] * 2 + [[1000.0, r_ch]] * 4)

    # Methane H-C-H angle triples (center atom in the middle), all six pairs.
    methane_h = [4, 5, 6, 7]
    methane_angle_idxs = np.array(
        [[a, 3, b] for i, a in enumerate(methane_h) for b in methane_h[i + 1 :]],
        dtype=np.int32,
    )
    theta_t = np.arccos(-1.0 / 3.0)  # ideal tetrahedral angle
    # HarmonicAngle params are [k, a0, eps].
    methane_angle_params = np.array([[100.0, theta_t, 0.0]] * len(methane_angle_idxs))

    # The water H-O-H angle is only needed so build_constraints can derive the
    # rigid H-H distance; it is not part of the flexible potential.
    water_angle_idxs = np.array([[1, 0, 2]], dtype=np.int32)
    water_angle_params = np.array([[100.0, theta_w]])

    # Angles handed to build_constraints (water angle drives rigid-water H-H).
    all_angle_idxs = np.vstack([water_angle_idxs, methane_angle_idxs[:, :3]])
    all_angle_params = np.vstack(
        [water_angle_params, methane_angle_params[:, :2]]
    )

    return (
        masses,
        x0,
        bond_idxs,
        bond_params,
        all_angle_idxs,
        all_angle_params,
        methane_angle_idxs,
        methane_angle_params,
    )


def global_constraint_pairs(clusters):
    """Return (pairs, r0) in global atom indices."""
    pairs, r0s = [], []
    cao = clusters.cluster_atom_offsets
    ca = clusters.cluster_atoms
    cco = clusters.cluster_constraint_offsets
    li = clusters.constraint_local_i
    lj = clusters.constraint_local_j
    r0 = clusters.constraint_r0
    for c in range(clusters.num_clusters):
        astart = cao[c]
        for k in range(cco[c], cco[c + 1]):
            pairs.append((ca[astart + li[k]], ca[astart + lj[k]]))
            r0s.append(r0[k])
    return np.array(pairs, dtype=int), np.array(r0s, dtype=float)


def maxwell_boltzmann(masses, temperature, seed):
    """Initial velocities (nm/ps); infinite-mass atoms are frozen."""
    rng = np.random.default_rng(seed)
    inv_m = np.where(np.isinf(masses), 0.0, 1.0 / masses)
    sigma = np.sqrt(BOLTZ * temperature * inv_m)[:, None]
    return sigma * rng.standard_normal((len(masses), 3))


def kinetic_energy(masses, v):
    finite = np.isfinite(masses)
    return 0.5 * np.sum(masses[finite, None] * v[finite] ** 2)


def drift_report(times, energies):
    """Summarize total-energy conservation for one trajectory."""
    energies = np.asarray(energies)
    e0 = energies[0]
    span = energies.max() - energies.min()
    # Linear drift slope (kJ/mol per ps), normalized per unit energy.
    slope = np.polyfit(times, energies, 1)[0]
    return {
        "E0": e0,
        "mean": energies.mean(),
        "std": energies.std(),
        "max_abs_dev": np.max(np.abs(energies - e0)),
        "peak_to_peak": span,
        "slope_per_ps": slope,
    }


def run_gpu(masses, x0, v0, clusters, angle_bp_impl, dt_ps, n_samples, steps_per_sample):
    intg = lib.ConstrainedVelocityVerletIntegrator(
        dt=dt_ps, masses=masses, constraints=clusters
    ).impl(precision=PRECISION)
    ctxt = lib.Context(
        x0, v0, BOX, intg, [angle_bp_impl], precision=PRECISION
    )

    times, energies, frames = [], [], []
    for s in range(n_samples + 1):
        if s > 0:
            ctxt.multiple_steps(steps_per_sample, steps_per_sample)
        x = ctxt.get_x_t()
        v = ctxt.get_v_t()
        du_dx, u = angle_bp_impl.execute(x, BOX.astype(PRECISION))
        ke = kinetic_energy(masses, v)
        times.append(s * steps_per_sample * dt_ps)
        energies.append(ke + float(np.sum(u)))
        frames.append((np.array(x), np.array(v)))
    return np.array(times), np.array(energies), frames


def run_reference(masses, x0, v0, pairs, r0s, force_fxn, energy_fxn, dt_ps, n_samples, steps_per_sample):
    intg = RefIntegrator(force_fxn, masses, dt_ps, pairs, r0s)
    n_steps = n_samples * steps_per_sample
    xs, vs = intg.multiple_steps(x0, v0, n_steps=n_steps)

    times, energies, frames = [], [], []
    for s in range(n_samples + 1):
        idx = s * steps_per_sample
        x, v = xs[idx], vs[idx]
        u = energy_fxn(x)
        ke = kinetic_energy(masses, v)
        times.append(idx * dt_ps)
        energies.append(ke + u)
        frames.append((x, v))
    return np.array(times), np.array(energies), frames


def run_unconstrained(masses, x0, v0, bp_impls, dt_ps, n_samples, steps_per_sample):
    """Plain (unconstrained) velocity-Verlet over the stiff-bond potential."""
    intg = lib.VelocityVerletIntegrator(dt=dt_ps, masses=masses).impl(precision=PRECISION)
    ctxt = lib.Context(x0, v0, BOX, intg, bp_impls, precision=PRECISION)
    box_p = BOX.astype(PRECISION)

    times, energies = [], []
    for s in range(n_samples + 1):
        if s > 0:
            ctxt.multiple_steps(steps_per_sample, steps_per_sample)
        x = ctxt.get_x_t()
        v = ctxt.get_v_t()
        u = sum(float(np.sum(bp.execute(x, box_p)[1])) for bp in bp_impls)
        ke = kinetic_energy(masses, v)
        times.append(s * steps_per_sample * dt_ps)
        energies.append(ke + u)
    return np.array(times), np.array(energies)


def main():
    (
        masses,
        x0,
        bond_idxs,
        bond_params,
        all_angle_idxs,
        all_angle_params,
        methane_angle_idxs,
        methane_angle_params,
    ) = build_system()

    clusters = cst.build_constraints(
        bond_idxs, bond_params, masses, all_angle_idxs, all_angle_params, rigid_water=True
    )
    pairs, r0s = global_constraint_pairs(clusters)
    print(f"system: {len(masses)} atoms, {clusters.num_clusters} clusters, "
          f"{clusters.num_constraints} constraints, {len(methane_angle_idxs)} flexible angles")

    angle_pot = HarmonicAngle(len(masses), methane_angle_idxs)
    angle_bp = angle_pot.bind(methane_angle_params).to_gpu(precision=PRECISION)
    angle_bp_impl = angle_bp.bound_impl

    box_p = BOX.astype(PRECISION)

    def force_fxn(x):
        du_dx, _ = angle_bp_impl.execute(np.asarray(x, dtype=PRECISION), box_p)
        return -np.asarray(du_dx, dtype=np.float64)

    def energy_fxn(x):
        _, u = angle_bp_impl.execute(np.asarray(x, dtype=PRECISION), box_p)
        return float(np.sum(u))

    # Unconstrained potential: the same flexible methane angles, plus stiff
    # harmonic X-H bonds and a stiff water bend in place of the rigid
    # constraints. The X-H stretches here are what cap the unconstrained dt.
    r_oh = float(bond_params[0, 1])
    r_ch = float(bond_params[2, 1])
    stiff_bond_params = np.array([[K_OH, r_oh]] * 2 + [[K_CH, r_ch]] * 4)
    bond_pot = HarmonicBond(len(masses), bond_idxs)
    bond_bp_impl = bond_pot.bind(stiff_bond_params).to_gpu(precision=PRECISION).bound_impl

    theta_w = float(all_angle_params[0, 1])
    water_angle_idxs = np.array([[1, 0, 2]], dtype=np.int32)
    water_angle_params = np.array([[K_HOH, theta_w, 0.0]])
    water_angle_pot = HarmonicAngle(len(masses), water_angle_idxs)
    water_angle_bp_impl = water_angle_pot.bind(water_angle_params).to_gpu(precision=PRECISION).bound_impl

    unconstrained_bps = [bond_bp_impl, water_angle_bp_impl, angle_bp_impl]

    v0_raw = maxwell_boltzmann(masses, TEMPERATURE, seed=2026).astype(np.float64)
    x0_raw = np.array(x0, dtype=np.float64)

    # Project the initial state onto the constraint manifold so that both
    # constrained integrators start from the same on-manifold (x, v); otherwise
    # the raw v0 carries velocity along the rigid constraints that RATTLE
    # removes on the first step, showing up as a spurious one-time energy drop.
    projector = RefIntegrator(force_fxn, masses, DT_SWEEP_FS[0] * 1e-3, pairs, r0s)
    x0 = projector._shake(x0_raw, x0_raw)
    v0 = projector._rattle(x0, v0_raw)

    gpu_frames_by_dt = {}
    ref_frames_by_dt = {}
    rows = []

    for dt_fs in DT_SWEEP_FS:
        dt_ps = dt_fs * 1e-3
        steps_per_sample = max(1, int(round(SAMPLE_INTERVAL_PS / dt_ps)))
        n_samples = max(1, int(round(TOTAL_TIME_PS / (steps_per_sample * dt_ps))))

        gpu_t, gpu_e, gpu_frames = run_gpu(
            masses, x0, v0, clusters, angle_bp_impl, dt_ps, n_samples, steps_per_sample
        )
        ref_t, ref_e, ref_frames = run_reference(
            masses, x0, v0, pairs, r0s, force_fxn, energy_fxn, dt_ps, n_samples, steps_per_sample
        )
        unc_t, unc_e = run_unconstrained(
            masses, x0_raw, v0_raw, unconstrained_bps, dt_ps, n_samples, steps_per_sample
        )
        gpu_frames_by_dt[dt_fs] = (gpu_t, gpu_frames)
        ref_frames_by_dt[dt_fs] = (ref_t, ref_frames)

        gpu = drift_report(gpu_t, gpu_e)
        ref = drift_report(ref_t, ref_e)
        unc = drift_report(unc_t, unc_e)

        print(f"\ndt = {dt_fs:.1f} fs  ({n_samples} samples x {steps_per_sample} steps)")
        for label, key, rep in (
            ("constrained GPU", "constrained_gpu", gpu),
            ("constrained ref", "constrained_ref", ref),
            ("unconstrained ", "unconstrained", unc),
        ):
            print(
                f"  {label}: E0={rep['E0']:10.5f}  "
                f"std={rep['std']:.3e}  max|dE|={rep['max_abs_dev']:.3e}  "
                f"ptp={rep['peak_to_peak']:.3e}  slope={rep['slope_per_ps']:+.3e} kJ/mol/ps"
            )
            rows.append(
                {
                    "dt_fs": dt_fs,
                    "integrator": key,
                    "n_samples": n_samples,
                    "steps_per_sample": steps_per_sample,
                    "E0": rep["E0"],
                    "mean": rep["mean"],
                    "std": rep["std"],
                    "max_abs_dev": rep["max_abs_dev"],
                    "peak_to_peak": rep["peak_to_peak"],
                    "slope_per_ps": rep["slope_per_ps"],
                }
            )

    # Cross-check GPU vs reference at the smallest dt where they should track
    # closely over the trajectory.
    dt_fs = DT_SWEEP_FS[0]
    gpu_t, gpu_frames = gpu_frames_by_dt[dt_fs]
    ref_t, ref_frames = ref_frames_by_dt[dt_fs]
    pos_devs = [np.max(np.abs(gf[0] - rf[0])) for gf, rf in zip(gpu_frames, ref_frames)]
    vel_devs = [np.max(np.abs(gf[1] - rf[1])) for gf, rf in zip(gpu_frames, ref_frames)]
    print(
        f"\nGPU vs reference at dt={dt_fs:.1f} fs: "
        f"max|dx|={max(pos_devs):.3e} nm  max|dv|={max(vel_devs):.3e} nm/ps "
        f"(first-sample dx={pos_devs[1] if len(pos_devs) > 1 else pos_devs[0]:.3e})"
    )

    # Constraint satisfaction check on the final GPU frame.
    xf, vf = gpu_frames[-1]
    d = np.linalg.norm(xf[pairs[:, 0]] - xf[pairs[:, 1]], axis=1)
    print(f"final GPU constraint error: max|d-r0|={np.max(np.abs(d - r0s)):.3e} nm")

    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote energy-drift results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
