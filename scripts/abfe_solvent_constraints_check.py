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
from tmd.fe.free_energy import AbsoluteFreeEnergy, MDParams, run_sims_bisection
from tmd.fe.topology import BaseTopology
from tmd.ff import Forcefield
from tmd.md import builders, minimizer


def run_leg(afe, ff, host_config, host_conf, md_params, temperature, seed, n_bisections, min_overlap, constrain):
    def make_initial_state(lamb):
        return abfe.get_initial_state(
            afe, ff, host_config, host_conf, temperature, seed, lamb, constrain_hydrogens=constrain
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--n-eq-steps", type=int, default=5000)
    parser.add_argument("--steps-per-frame", type=int, default=400)
    parser.add_argument("--n-bisections", type=int, default=12)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--box-width", type=float, default=4.0)
    parser.add_argument("--forcefield", type=str, default="smirnoff_2_0_0_sc.py")
    args = parser.parse_args()

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
    )

    print(
        f"settings: n_bisections={args.n_bisections} min_overlap={args.min_overlap} "
        f"n_frames={args.n_frames} n_eq_steps={args.n_eq_steps} "
        f"steps_per_frame={args.steps_per_frame} seed={args.seed} ff={args.forcefield}",
        flush=True,
    )

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
