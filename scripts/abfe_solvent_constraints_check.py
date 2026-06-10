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

"""Sanity check: solvent ABFE leg with and without hydrogen constraints.

Runs the solvent decoupling leg of an absolute binding free energy calculation
using bisection to determine the lambda schedule (which adaptively inserts
windows where BAR overlap is poor, avoiding the fixed-schedule endpoint-overlap
problem). The leg is run both with the standard unconstrained Langevin
integrator and with the constrained (SHAKE/RATTLE) integrator, and the total
dG and per-pair overlaps are printed for comparison.
"""

import argparse
import time

import numpy as np

from tmd import testsystems
from tmd.constants import DEFAULT_TEMP
from tmd.fe.absolute import abfe
from tmd.fe.utils import read_sdf
from tmd.fe.free_energy import AbsoluteFreeEnergy, MDParams, get_context, run_sims_bisection
from tmd.fe.topology import BaseTopology
from tmd.ff import Forcefield
from tmd.md import builders, minimizer


def run_leg(afe, ff, host_config, host_conf, md_params, temperature, seed, n_bisections, min_overlap, constrain):
    def make_initial_state(lamb):
        return abfe.get_initial_state(
            afe, ff, host_config, host_conf, temperature, seed, lamb,
            constrain_hydrogens=constrain, dt=md_params.dt,
        )

    t0 = time.time()
    results, _ = run_sims_bisection(
        [0.0, 1.0],
        make_initial_state,
        md_params,
        n_bisections=n_bisections,
        temperature=temperature,
        min_overlap=min_overlap,
        verbose=False,
        batch_size=8,
    )
    wall = time.time() - t0

    final = results[-1]
    lambdas = np.array([s.lamb for s in final.initial_states])
    dGs = np.array(final.dGs)
    dG_errs = np.array(final.dG_errs)
    overlaps = np.array([r.overlap for r in final.bar_results])
    total_dG = float(np.sum(dGs))
    total_err = float(np.sqrt(np.sum(dG_errs**2)))

    print(f"  final schedule has {len(lambdas)} windows", flush=True)
    print("  pair  lambda_i -> lambda_j      dG       dG_err   overlap", flush=True)
    for i in range(len(dGs)):
        print(
            f"  {i:>3}   {lambdas[i]:8.4f} -> {lambdas[i + 1]:8.4f}   "
            f"{dGs[i]:8.3f}  {dG_errs[i]:8.3f}  {overlaps[i]:8.3f}",
            flush=True,
        )
    return total_dG, total_err, wall


def _global_constraint_pairs(clusters):
    """Return (pairs, r0) in global atom indices for the constraint clusters."""
    pairs = []
    r0s = []
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


def check_local_md_stability(
    afe, ff, host_config, host_conf, md_params, temperature, seed, local_steps, dist_atol=5e-3
):
    """Stability test: run local MD on a constrained, minimized solvent state and verify it stays
    stable. Local MD freezes a per-step subset of atoms (treated as infinite-mass anchors by the
    constrained integrator), which is the path most likely to destabilize SHAKE/RATTLE. The test
    passes when coordinates stay finite and every constraint distance is held to ``dist_atol``.
    """
    state = abfe.get_initial_state(
        afe, ff, host_config, host_conf, temperature, seed, lamb=0.0,
        constrain_hydrogens=True, dt=md_params.dt,
    )
    state = abfe.optimize_abfe_initial_state(state)

    clusters = state.integrator.constraints
    pairs, targets = _global_constraint_pairs(clusters)
    print(f"  {clusters.num_constraints} constraints over {clusters.num_clusters} clusters", flush=True)

    ctxt = get_context(state, md_params)
    ctxt.setup_local_md(temperature, freeze_reference=True)

    # Equilibrate globally, then run a long stretch of local MD centered on the ligand.
    ctxt.multiple_steps(500, 0)
    assert np.all(np.isfinite(ctxt.get_x_t())), "global equilibration produced a nan"

    store_interval = max(local_steps // 10, 1)
    xs, _ = ctxt.multiple_steps_local(
        local_steps,
        state.ligand_idxs.astype(np.int32),
        k=1.0,
        radius=1.2,
        store_x_interval=store_interval,
        seed=seed,
    )

    finite = bool(np.all(np.isfinite(xs)))
    max_dev = 0.0
    for frame in xs:
        d = np.linalg.norm(frame[pairs[:, 0]] - frame[pairs[:, 1]], axis=1)
        max_dev = max(max_dev, float(np.max(np.abs(d - targets))))

    print(f"  finite coordinates: {finite}", flush=True)
    print(f"  max constraint deviation: {max_dev:.2e} nm (tol {dist_atol:.0e})", flush=True)
    passed = finite and max_dev <= dist_atol
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}", flush=True)
    return passed



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--n-eq-steps", type=int, default=5000)
    parser.add_argument("--steps-per-frame", type=int, default=400)
    parser.add_argument(
        "--dt", type=float, default=2.5e-3, help="Integrator timestep in picoseconds (default 2.5e-3 = 2.5 fs)"
    )
    parser.add_argument("--n-bisections", type=int, default=12)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--box-width", type=float, default=4.0)
    parser.add_argument("--forcefield", type=str, default="smirnoff_2_0_0_sc.py")
    parser.add_argument("--sdf")
    parser.add_argument(
        "--local-md-steps",
        type=int,
        default=2000,
        help="Number of local MD steps to run in the constrained local-MD stability check",
    )
    parser.add_argument(
        "--skip-bisection",
        action="store_true",
        help="Only run the constrained local-MD stability check, skipping the dG bisection legs",
    )
    args = parser.parse_args()

    if args.sdf:
        mol, *_ = read_sdf(args.sdf)

    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_from_file(args.forcefield)
    temperature = DEFAULT_TEMP

    host_config = builders.build_water_system(args.box_width, ff.water_ff, mols=[mol], box_margin=0.1)
    afe = AbsoluteFreeEnergy(mol, BaseTopology(mol, ff))

    # Minimize the host once; reuse for both legs so they start from identical coords.
    host_conf = minimizer.fire_minimize_host([mol], host_config, ff)

    md_params = MDParams(
        seed=args.seed,
        n_eq_steps=args.n_eq_steps,
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
        dt=args.dt,
    )

    print(
        f"settings: n_bisections={args.n_bisections} min_overlap={args.min_overlap} "
        f"n_frames={args.n_frames} n_eq_steps={args.n_eq_steps} "
        f"steps_per_frame={args.steps_per_frame} dt={args.dt} seed={args.seed} ff={args.forcefield}",
        flush=True,
    )

    print("\n=== constrained local MD stability ===", flush=True)
    local_md_ok = check_local_md_stability(
        afe, ff, host_config, host_conf, md_params, temperature, args.seed, args.local_md_steps,
    )
    if not local_md_ok:
        raise SystemExit("constrained local MD stability check FAILED")

    if args.skip_bisection:
        return

    print("\n=== unconstrained ===", flush=True)
    dG_off, err_off, wall_off = run_leg(
        afe, ff, host_config, host_conf, md_params, temperature, args.seed,
        args.n_bisections, args.min_overlap, constrain=False,
    )
    print(f"dG = {dG_off:.3f} +/- {err_off:.3f} kJ/mol  ({wall_off:.1f}s)", flush=True)

    print("\n=== constrained (H-bonds + rigid water) ===", flush=True)
    dG_on, err_on, wall_on = run_leg(
        afe, ff, host_config, host_conf, md_params, temperature, args.seed,
        args.n_bisections, args.min_overlap, constrain=True,
    )
    print(f"dG = {dG_on:.3f} +/- {err_on:.3f} kJ/mol  ({wall_on:.1f}s)", flush=True)

    diff = dG_on - dG_off
    combined_err = float(np.sqrt(err_off**2 + err_on**2))
    print("\n=== comparison ===", flush=True)
    print(f"delta(constrained - unconstrained) = {diff:.3f} kJ/mol", flush=True)
    print(f"combined stat error                = {combined_err:.3f} kJ/mol", flush=True)
    print(f"|delta| / combined_err             = {abs(diff) / combined_err:.2f}", flush=True)


if __name__ == "__main__":
    main()
