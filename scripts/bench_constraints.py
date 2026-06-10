"""Benchmark constrained vs unconstrained Langevin MD throughput.

Times steady-state ``multiple_steps`` throughput for a solvated water box of a
given size, for both the unconstrained and constrained (SHAKE/RATTLE)
integrators, and reports per-step wall time and the constrained overhead.

The key diagnostic is how the *delta* (constrained - unconstrained) per-step
time scales with system size: a roughly constant delta implies the overhead is
dominated by per-step kernel-launch latency (the constrained step issues ~10
extra kernel launches), while a delta that grows with N implies the constraint
math itself is the bottleneck.
"""

import argparse
import time

import numpy as np

from tmd.fe.model_utils import apply_hmr
from tmd.ff import Forcefield
from tmd.lib import ConstrainedLangevinIntegrator, LangevinIntegrator
from tmd.md.builders import build_water_system
from tmd.potentials import HarmonicAngle
from tmd.potentials import HarmonicBond as HarmonicBondPot
from tmd.potentials.potential import get_bound_potential_by_type
import tmd.fe.constraints as cst
from tmd import lib


def make_contexts(box_width, precision, dt, seed):
    ff = Forcefield.load_default()
    hc = build_water_system(box_width, ff.water_ff, box_margin=0.1)
    bp_list = hc.host_system.get_U_fns()
    masses = np.array(hc.masses)
    bbp = get_bound_potential_by_type(bp_list, HarmonicBondPot)
    abp = get_bound_potential_by_type(bp_list, HarmonicAngle)

    clusters = cst.build_constraints(
        bbp.potential.idxs, bbp.params, masses, abp.potential.idxs, abp.params, rigid_water=True
    )
    hmr = apply_hmr(masses, bbp.potential.idxs)

    conf = hc.conf.astype(precision)
    box = hc.box.astype(precision)
    v0 = np.zeros_like(conf)

    def unconstrained():
        intg = LangevinIntegrator(300.0, dt, 1.0, masses, seed).impl(precision)
        bps = [b.to_gpu(precision=precision).bound_impl for b in bp_list]
        return lib.Context(conf.copy(), v0.copy(), box.copy(), intg, bps, precision=precision)

    def constrained():
        intg = ConstrainedLangevinIntegrator(300.0, dt, 1.0, hmr, seed, clusters).impl(precision)
        bps = [b.to_gpu(precision=precision).bound_impl for b in bp_list]
        return lib.Context(conf.copy(), v0.copy(), box.copy(), intg, bps, precision=precision)

    return unconstrained, constrained, len(conf), clusters


def time_steps(ctxt, n_warmup, n_time):
    ctxt.multiple_steps(n_warmup)  # warm up (also syncs at the end)
    t0 = time.perf_counter()
    ctxt.multiple_steps(n_time)  # syncs at the end
    return (time.perf_counter() - t0) / n_time * 1e3  # ms/step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-widths", type=float, nargs="+", default=[2.5, 3.0, 4.0])
    parser.add_argument("--dt", type=float, default=2.5e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-time", type=int, default=1000)
    parser.add_argument("--precision", choices=["f32", "f64"], default="f32")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    precision = np.float32 if args.precision == "f32" else np.float64

    print(
        f"precision={args.precision} dt={args.dt} n_warmup={args.n_warmup} "
        f"n_time={args.n_time} repeats={args.repeats}",
        flush=True,
    )
    print(
        f"{'box':>5} {'N':>7} {'clusters':>8} {'uncon(ms)':>10} {'con(ms)':>9} "
        f"{'delta(ms)':>10} {'ratio':>6} {'delta/atom(us)':>14}",
        flush=True,
    )

    for bw in args.box_widths:
        unconstrained, constrained, n_atoms, clusters = make_contexts(bw, precision, args.dt, args.seed)

        def best(factory):
            return min(time_steps(factory(), args.n_warmup, args.n_time) for _ in range(args.repeats))

        t_unc = best(unconstrained)
        t_con = best(constrained)
        delta = t_con - t_unc
        print(
            f"{bw:>5.1f} {n_atoms:>7} {clusters.num_clusters:>8} {t_unc:>10.4f} {t_con:>9.4f} "
            f"{delta:>10.4f} {t_con / t_unc:>6.2f} {delta / n_atoms * 1e3:>14.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
