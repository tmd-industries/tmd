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

"""Sanity check: run solvent hydration with and without hydrogen constraints.

Runs an absolute hydration free energy calculation for a small molecule both
with the standard unconstrained Langevin integrator and with the constrained
(SHAKE/RATTLE) integrator, and prints the total dG and per-window errors so the
two can be compared.
"""

import argparse
import time

import numpy as np

from tmd import testsystems
from tmd.fe.absolute import hydration as absolute_hydration
from tmd.fe.free_energy import MDParams
from tmd.ff import Forcefield


def run_once(mol, ff, md_params, n_windows, constrain_hydrogens):
    t0 = time.time()
    res, _ = absolute_hydration.run_solvent(
        mol,
        ff,
        None,
        md_params=md_params,
        n_windows=n_windows,
        constrain_hydrogens=constrain_hydrogens,
    )
    wall = time.time() - t0
    states = res.final_result.initial_states
    lambdas = np.array([s.lamb for s in states])
    dGs = np.array(res.final_result.dGs)
    dG_errs = np.array(res.final_result.dG_errs)

    # A pair is "reliable" if BAR returned a finite, small uncertainty. The
    # decoupling endpoint pair (lambda ~1.0 -> ~0.3) has near-zero phase-space
    # overlap and returns nan/huge error regardless of sampling; exclude it so
    # the constraint comparison is apples-to-apples over the well-sampled region.
    reliable = np.isfinite(dG_errs) & (dG_errs < 1.0)

    # Per-window diagnostics so a single low-overlap pair is obvious.
    print("  pair  lambda_i -> lambda_j      dG        dG_err", flush=True)
    for i in range(len(dGs)):
        flag = "" if reliable[i] else "  <-- low overlap (excluded)"
        print(
            f"  {i:>3}   {lambdas[i]:8.4f} -> {lambdas[i + 1]:8.4f}   "
            f"{dGs[i]:9.3f}  {dG_errs[i]:12.3f}{flag}",
            flush=True,
        )

    return {
        "lambdas": lambdas,
        "dGs": dGs,
        "dG_errs": dG_errs,
        "reliable": reliable,
        "wall": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--n-eq-steps", type=int, default=2000)
    parser.add_argument("--steps-per-frame", type=int, default=400)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--forcefield", type=str, default="smirnoff_2_0_0_sc.py")
    args = parser.parse_args()

    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_from_file(args.forcefield)

    md_params = MDParams(
        seed=args.seed,
        n_eq_steps=args.n_eq_steps,
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
    )

    print(
        f"settings: n_windows={args.n_windows} n_frames={args.n_frames} "
        f"n_eq_steps={args.n_eq_steps} steps_per_frame={args.steps_per_frame} "
        f"seed={args.seed} ff={args.forcefield}",
        flush=True,
    )

    print("\n=== unconstrained ===", flush=True)
    off = run_once(mol, ff, md_params, args.n_windows, constrain_hydrogens=False)

    print("\n=== constrained (H-bonds + rigid water) ===", flush=True)
    on = run_once(mol, ff, md_params, args.n_windows, constrain_hydrogens=True)

    # Compare only the region where both legs have reliable (high-overlap) pairs.
    # The schedule is identical across legs, so reliable pairs line up by index.
    both_reliable = off["reliable"] & on["reliable"]
    dG_off = float(np.sum(off["dGs"][both_reliable]))
    dG_on = float(np.sum(on["dGs"][both_reliable]))
    err_off = float(np.sqrt(np.sum(off["dG_errs"][both_reliable] ** 2)))
    err_on = float(np.sqrt(np.sum(on["dG_errs"][both_reliable] ** 2)))
    diff = dG_on - dG_off
    combined_err = float(np.sqrt(err_off**2 + err_on**2))

    print("\n=== comparison (over high-overlap windows only) ===", flush=True)
    n_excluded = int(np.sum(~both_reliable))
    print(f"windows compared: {int(np.sum(both_reliable))} of {len(both_reliable)} "
          f"({n_excluded} low-overlap pair(s) excluded)", flush=True)
    print(f"partial dG  unconstrained = {dG_off:.3f} +/- {err_off:.3f} kJ/mol  ({off['wall']:.1f}s)", flush=True)
    print(f"partial dG  constrained   = {dG_on:.3f} +/- {err_on:.3f} kJ/mol  ({on['wall']:.1f}s)", flush=True)
    print(f"delta(constrained - unconstrained) = {diff:.3f} kJ/mol", flush=True)
    print(f"combined stat error                = {combined_err:.3f} kJ/mol", flush=True)
    print(f"|delta| / combined_err             = {abs(diff) / combined_err:.2f}", flush=True)
    print(f"delta(constrained - unconstrained) = {diff:.3f} kJ/mol", flush=True)
    print(f"combined median-window error       = {combined_err:.3f} kJ/mol", flush=True)
    print(f"|delta| / combined_err             = {abs(diff) / combined_err:.2f}", flush=True)


if __name__ == "__main__":
    main()
