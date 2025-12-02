# Copyright 2025 Justin Gullingsrud
# Modifications Copyright 2025, Forrest York
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

import pickle
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from dataclasses import dataclass, replace
from typing import Any, Self

import mdtraj
from numpy.typing import NDArray
from rbfe_common import COMPLEX_LEG, SOLVENT_LEG, compute_total_ns

from tmd import potentials
from tmd.constants import BOLTZ, DEFAULT_FF, DEFAULT_PRESSURE, DEFAULT_TEMP
from tmd.fe import model_utils
from tmd.fe.absolute.restraints import (
    select_ligand_atoms_baumann,
    select_receptor_atoms_baumann,
)
from tmd.fe.cif_writer import build_openmm_topology
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
    HREXParams,
    HREXSimulationResult,
    InitialState,
    LocalMDParams,
    MDParams,
    RESTParams,
    SimulationResult,
    WaterSamplingParams,
    make_pair_bar_plots,
    run_sims_bisection,
    sample,
)
from tmd.fe.plots import (
    plot_as_png_fxn,
    plot_forward_and_reverse_dg,
    plot_water_proposals_by_state,
)
from tmd.fe.rbfe import DEFAULT_NUM_WINDOWS, HostConfig, estimate_relative_free_energy_bisection_hrex_impl
from tmd.fe.topology import BaseTopology
from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name, set_romol_conf
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, MonteCarloBarostat
from tmd.md import minimizer
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.md.builders import build_protein_system, build_water_system
from tmd.md.exchange.utils import get_radius_of_mol_pair
from tmd.parallel.client import AbstractFileClient, CUDAMPSPoolClient, FileClient, iterate_completed_futures
from tmd.parallel.utils import get_gpu_count
from tmd.potentials import (
    HarmonicAngle,
    HarmonicBond,
    PeriodicTorsion,
)
from tmd.potentials.bonded import kahan_angle, signed_torsion_angle
from tmd.potentials.potential import get_potential_by_type

Snapshot = tuple[NDArray, NDArray]


def get_abfe_endstate_trajectory(topology, mol, snapshots: Snapshot):
    combined_top = build_openmm_topology([topology, mol])
    top = mdtraj.Topology.from_openmm(combined_top)
    frames, boxes = snapshots
    angles = [[90.0] * 3] * len(boxes)
    lengths = [np.diag(box) for box in boxes]
    traj = mdtraj.Trajectory(frames, top, unitcell_lengths=lengths, unitcell_angles=angles)
    return traj


@dataclass(frozen=True)
class RestraintParams:
    kb: float = 5000  # bond restraint strength
    ka: float = 200  # angle restraint
    kd: float = 10  # dihedral restraint
    on: float = 0.0625  # where in lambda schedule the restraints go to full strength


class AbsoluteBindingFreeEnergy(AbsoluteFreeEnergy):
    def __init__(self, rec_atoms, lig_atoms, cpx_coords, box, params, *args, **kwds):
        super().__init__(*args, **kwds)
        self.rec_atoms = rec_atoms
        self.lig_atoms = lig_atoms
        self.cpx_coords = cpx_coords
        self.box = box
        self.params = params

    @classmethod
    def create(cls, bt, host_config, tmtrj, rst_params) -> Self:
        mol = bt.mol

        # construct trajectory from snapshots aligned to final frame
        # TODO: toss out initial 20% of frames?
        snapshots = (tmtrj.frames, np.array(tmtrj.boxes))
        trj = get_abfe_endstate_trajectory(host_config.omm_topology, mol, snapshots)
        trj.image_molecules(inplace=True)
        ref = trj[-1]
        """
        In the original implementation, rmsf was computed after aligning the
        ligand-only trajectory to the initial ligand coordinates.  Here we
        instead align the protein backbone using the final frame of the trajectory.
        The motivation here is that we want ligand restraint points which are stable
        relative to the protein starting point.
        """
        trj.superpose(ref, atom_indices=trj.topology.select("backbone"))
        # use rmsf to filter restraint atoms
        rmsf = mdtraj.rmsf(trj, ref)

        # select ligand atoms
        ligand_offset = trj.n_atoms - mol.GetNumAtoms()
        lig_ids = select_ligand_atoms_baumann(mol, rmsf[ligand_offset:])
        lig_ids = [i + ligand_offset for i in lig_ids]

        # select receptor atoms
        # FIXME: reuse the rmsf calculation instead of redoing it in this routine
        rec_ids = select_receptor_atoms_baumann(trj, lig_ids)

        pos = tmtrj.frames[-1]
        box = tmtrj.boxes[-1]
        return cls(rec_ids, lig_ids, pos, box, rst_params, mol, bt)

    def get_bond_geometry(self):
        """Get atom1, atom2, distance."""
        i0 = [self.rec_atoms[0], self.lig_atoms[0]]
        a0, b0 = self.cpx_coords[i0]
        r0 = np.linalg.norm(a0 - b0)
        return (i0, r0)

    def get_bond_restraint(self, scale: float) -> tuple[HarmonicBond, NDArray]:
        i0, r0 = self.get_bond_geometry()
        fc = self.params.kb * scale

        idxs = np.array([i0], dtype=np.int32)
        params = np.array([[fc, r0]], dtype=np.float32)

        return HarmonicBond(idxs), params

    def get_angle_geometry(self):
        """Get atom1, atom2, atom3, angle."""
        rec = self.rec_atoms
        lig = self.lig_atoms
        i0 = [rec[1], rec[0], lig[0]]
        i1 = [rec[0], lig[0], lig[1]]

        pos = self.cpx_coords
        t0 = kahan_angle(*pos[i0], 0, self.box)
        t1 = kahan_angle(*pos[i1], 0, self.box)
        return [(i0, t0), (i1, t1)]

    def get_angle_restraint(self, scale: float) -> tuple[HarmonicAngle, NDArray]:
        ((i0, t0), (i1, t1)) = self.get_angle_geometry()
        fc = self.params.ka * scale

        idxs = np.array([i0, i1], dtype=np.int32)
        params = np.array([[fc, t0, 0], [fc, t1, 0]], dtype=np.float32)
        return HarmonicAngle(idxs), params

    def get_dihedral_geometry(self):
        """Get atom1, atom2, atom3, atom4, angle."""
        rec = self.rec_atoms
        lig = self.lig_atoms
        i0 = [rec[2], rec[1], rec[0], lig[0]]
        i1 = [rec[1], rec[0], lig[0], lig[1]]
        i2 = [rec[0], lig[0], lig[1], lig[2]]

        pos = self.cpx_coords
        p0 = signed_torsion_angle(*pos[i0], self.box)
        p1 = signed_torsion_angle(*pos[i1], self.box)
        p2 = signed_torsion_angle(*pos[i2], self.box)
        return [(i0, p0), (i1, p1), (i2, p2)]

    def get_dihedral_restraint(self, scale: float) -> tuple[PeriodicTorsion, NDArray]:
        idxs_ = []
        params_ = []
        k = self.params.kd * scale
        const = 0
        for ids, phi in self.get_dihedral_geometry():
            for period in range(1, 6 + 1):
                phase = period * phi
                fc = -k / period
                params_.append([fc, phase, period])
                idxs_.append(ids)
                const += fc
        # add a constant term to make restraint minimum have E=0
        params_.append([-const, 0, 0])
        idxs_.append(ids)

        idxs = np.array(idxs_, dtype=np.int32)
        params = np.array(params_, dtype=np.float32)
        return PeriodicTorsion(idxs), params

    def get_restraint_correction(self, temperature: float) -> float:
        """Compute correction to FE from restraint.

        https://pubs.acs.org/doi/10.1021/jp0217839
        Equation 32
        """
        _, r0 = self.get_bond_geometry()
        ((_, t0), (_, t1)) = self.get_angle_geometry()
        s0 = np.sin(t0)
        s1 = np.sin(t1)
        V0 = 1660 * (0.1**3)  # 1M standard state in nm^3
        kT = BOLTZ * temperature

        kb = self.params.kb * 0.5  # TMD applies a factor of 0.5
        ka = self.params.ka * 0.5  # TMD applies a factor of 0.5
        kd = self.params.kd  # no prefactor I think

        return -kT * np.log(8 * np.pi**2 * V0 * (kb * ka**2 * kd**3) ** 0.5 / (r0**2 * s0 * s1 * (2 * np.pi * kT) ** 3))

    def prepare_host_edge(self, ff: Forcefield, host_config: HostConfig, lamb: float):
        """Construct alchemical schedule.

        lambda=0 -> fully coupled, no restraint
        lambda=restraint_on: fully coupled, full restraint
        lambda=1: decoupled, full restraint.
        """
        restraint_on = self.params.on
        m = 1 / (1 - restraint_on)
        b = 1 - m
        hglamb = max(0, m * lamb + b)
        ubps, params, masses = super().prepare_host_edge(ff, host_config, hglamb)

        scale = min(1, lamb / restraint_on)
        # must use the same set of potentials in all windows, even if scale is zero
        hb, hbp = self.get_bond_restraint(scale)
        ha, hap = self.get_angle_restraint(scale)
        dr, drp = self.get_dihedral_restraint(scale)

        ubps = (*list(ubps), hb, ha, dr)
        params = (*list(params), hbp, hap, drp)

        return ubps, params, masses


def get_initial_state(afe, ff, host_config, host_conf, temperature, seed, lamb):
    """Get initial state at a particular lambda.

    Pulled out of tmd/fe/absolute/hydration.py so that we can do bisection
    instead of a predefined schedule.
    """
    ubps, params, masses = afe.prepare_host_edge(ff, host_config, lamb)
    x0 = afe.prepare_combined_coords(host_coords=host_conf)
    bps = []
    for ubp, param in zip(ubps, params):
        bp = ubp.bind(param)
        bps.append(bp)

    bond_potential = get_potential_by_type(ubps, potentials.HarmonicBond)

    hmr_masses = model_utils.apply_hmr(masses, bond_potential.idxs)
    group_idxs = get_group_indices(get_bond_list(bond_potential), len(masses))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 25, seed)
    box0 = host_config.box

    v0 = np.zeros_like(x0)  # tbd resample from Maxwell-boltzman?
    num_ligand_atoms = afe.mol.GetNumAtoms()
    num_total_atoms = len(x0)
    ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms)

    dt = 2.5e-3
    friction = 1.0
    intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)

    return InitialState(bps, intg, baro, x0, v0, box0, lamb, ligand_idxs, np.array([], dtype=np.int32))


def estimate_abfe_leg(
    mol,
    ff: Forcefield,
    leg: str,
    host_config: HostConfig,
    prefix,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    rst_params: RestraintParams,
):
    host_conf = minimizer.fire_minimize_host(
        [mol],
        host_config,
        ff,
    )
    bt = BaseTopology(mol, ff)
    temperature = DEFAULT_TEMP

    afe = AbsoluteFreeEnergy(mol, bt)
    if leg == COMPLEX_LEG:
        # Run short equilibration to obtain trajectory used to pick restraint atoms
        initial_state = get_initial_state(afe, ff, host_config, host_conf, temperature, md_params.seed, 0.0)
        # TBD: How many frames do you want from here?
        sample_md_params = replace(md_params, n_eq_steps=200000, n_frames=100)
        trj = sample(initial_state, sample_md_params, 100)

        afe = AbsoluteBindingFreeEnergy.create(bt, host_config, trj, rst_params)

        # get equilibrated coordinates and box
        host_conf = afe.cpx_coords[: len(host_conf)]
        set_romol_conf(afe.mol, afe.cpx_coords[len(host_conf) :])
        host_config = replace(host_config, box=afe.box)
    else:
        # Disable water sampling
        md_params = replace(md_params, water_sampling_params=None)

    def create_abfe_initial_state(lamb):
        return get_initial_state(afe, ff, host_config, host_conf, temperature, md_params.seed, lamb)

    if md_params.hrex_params is None:
        bisection_params = md_params

        initial_lambdas = [0.0, 1.0]
        list_of_results, trjs = run_sims_bisection(
            initial_lambdas,
            create_abfe_initial_state,
            bisection_params,
            n_bisections=n_windows - len(initial_lambdas),
            temperature=temperature,
            min_overlap=min_overlap,
        )
        results = list_of_results[-1]

        plots = make_pair_bar_plots(results, temperature, prefix)
        sim_result = SimulationResult(results, plots, trjs, md_params, list_of_results)
    else:
        sim_result = estimate_relative_free_energy_bisection_hrex_impl(
            temperature,
            0.0,
            1.0,
            md_params,
            n_windows,
            create_abfe_initial_state,
            lambda x: x,  # No optimization done here
            combined_prefix=prefix,
            min_overlap=min_overlap,
        )
    if leg == COMPLEX_LEG:
        sim_result.correction = afe.get_restraint_correction(temperature)  # type: ignore
        sim_result.rec_atoms = afe.rec_atoms  # type: ignore
        sim_result.lig_atoms = afe.lig_atoms  # type: ignore
    return sim_result


def run_abfe(
    file_client: AbstractFileClient,
    mol_path: Path,
    mol: Chem.Mol,
    leg: str,
    ff: Forcefield,
    pdb_path: str,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    write_trajectories: bool,
    force_overwrite: bool,
) -> dict[str, Any]:
    """Run an ABFE calculation.

    Will store results using the file_client to a new directory that name of the ligand being run.

    Stores the following files:

    * results.npz - Predictions, overlaps and the number of windows
    * lambda*_traj.npz - Store the endstate trajectories (if write_trajectory is set to True)
    * final_pairbar_result.pkl - Pickled copy of the final PairBarResult object
    * host_config.pkl - Pickled HostConfig, if the leg is not vacuum
    * dg_errors.png - PNG of the dg errors
    * overlap_summary.png - PNG of the pair bar overlap between windows
    * forward_and_reverse_dg.png - PNG of forward and reverse dG for evaluating convergence
    * hrex_transition_matrix.png - PNG of the transition matrix plot
    * hrex_replica_state_distribution_heatmap.png - PNG of the HREX replica state distribution heatmap
    * water_sampling_acceptances.png - PNG of water sampling acceptances by window

    Parameters
    ----------
    file_client : FileClient
        File client for storing results of the simulation
    mol_path: Path
        Path to write out molecule results to
    mol : Chem.Mol
        Molecule in the system.
    leg: str
        Either complex or solvent
    ff : Forcefield
        Forcefield
    pdb_path : str
        Path to a PDB file
    md_params : MDParams
        Parameters for the RBFE simulation.
    n_windows : int
        Maximum number of windows to generate during bisection.
    min_overlap : float
        Minimum overlap used during bisection.
    write_trajectories: bool
        Whether or not to write trajectories
    force_overwrite: bool
        If results already exist, overwrite the results

    Returns
    -------
    Summary data
        Data contained in the results.npz. Will include pred_dg
    """
    assert leg in (COMPLEX_LEG, SOLVENT_LEG)
    # Ensure the output directories exists
    leg_path = mol_path / leg
    Path(file_client.full_path(leg_path)).mkdir(parents=True, exist_ok=True)
    results_path = Path(file_client.full_path(leg_path / "results.npz"))
    if not force_overwrite and results_path.is_file():
        print(f"Skipping abfe {leg} calculation: {get_mol_name(mol)}")
        return dict(np.load(results_path))

    with open(file_client.full_path(mol_path / "md_params.pkl"), "wb") as ofs:
        pickle.dump(md_params, ofs)
    with Chem.SDWriter(file_client.full_path(mol_path / "mol.sdf")) as writer:
        writer.write(mol)

    np.random.seed(md_params.seed)
    start = time.perf_counter()
    if leg == COMPLEX_LEG:
        host_config = build_protein_system(pdb_path, ff.protein_ff, ff.water_ff, mols=[mol], box_margin=0.1)
    else:
        host_config = build_water_system(4.0, ff.water_ff, mols=[mol], box_margin=0.1)
    # TBD: Expose restraint params?
    res = estimate_abfe_leg(
        mol, ff, leg, host_config, f"abfe_{leg}", md_params, n_windows, min_overlap, RestraintParams()
    )
    took = time.perf_counter() - start

    pred_dg = float(np.sum(res.final_result.dGs))
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))

    correction = 0.0 if leg == SOLVENT_LEG else res.correction

    print(
        f"{get_mol_name(mol)} {leg} (kJ/mol) {pred_dg:.2f} +- {pred_dg_err:.2f}, Correction {correction:.2f}, {took:.0f} Seconds"
    )

    summary_data = {
        "time": took,
        "total_ns": compute_total_ns(res, md_params),
        "pred_dg": pred_dg,
        "pred_dg_err": pred_dg_err,
        "n_windows": len(res.final_result.initial_states),
        "overlaps": res.final_result.overlaps,
    }
    if isinstance(res, HREXSimulationResult):
        summary_data["bisected_windows"] = len(res.intermediate_results[-1].initial_states)
        summary_data["normalized_kl_divergence"] = res.hrex_diagnostics.normalized_kl_divergence
    if leg == COMPLEX_LEG:
        summary_data["correction"] = res.correction
        summary_data["receptor_restraint_atoms"] = res.rec_atoms
        summary_data["ligand_restraint_atoms"] = res.lig_atoms

    np.savez_compressed(results_path, **summary_data)

    if write_trajectories:
        np.savez_compressed(
            file_client.full_path(leg_path / "lambda0_traj.npz"),
            coords=np.array(res.trajectories[0].frames),
            boxes=np.asarray(res.trajectories[0].boxes),
        )
        np.savez_compressed(
            file_client.full_path(leg_path / "lambda1_traj.npz"),
            coords=np.array(res.trajectories[-1].frames),
            boxes=np.asarray(res.trajectories[-1].boxes),
        )
    if host_config is not None:
        file_client.store(leg_path / "host_config.pkl", pickle.dumps(host_config))

    if isinstance(res, HREXSimulationResult):
        file_client.store(leg_path / "hrex_transition_matrix.png", res.hrex_plots.transition_matrix_png)
        file_client.store(
            leg_path / "hrex_replica_state_distribution_heatmap.png",
            res.hrex_plots.replica_state_distribution_heatmap_png,
        )
        if res.water_sampling_diagnostics is not None:
            file_client.store(
                leg_path / "water_sampling_acceptances.png",
                plot_as_png_fxn(
                    plot_water_proposals_by_state,
                    [state.lamb for state in res.final_result.initial_states],
                    res.water_sampling_diagnostics.cumulative_proposals_by_state,
                ),
            )
    file_client.store(leg_path / "dg_errors.png", res.plots.dG_errs_png)
    file_client.store(leg_path / "overlap_summary.png", res.plots.overlap_summary_png)
    u_kln_by_lambda = res.final_result.u_kln_by_component_by_lambda.sum(1)
    file_client.store(
        leg_path / "forward_and_reverse_dg.png",
        plot_forward_and_reverse_dg(u_kln_by_lambda, frames_per_step=min(100, u_kln_by_lambda.shape[-1])),
    )
    # Contains initial states and the complete u_kln
    file_client.store(leg_path / "final_pairbar_result.pkl", pickle.dumps(res.final_result))
    return summary_data


def main():
    parser = ArgumentParser(description="Run ABFE for a set of compounds")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--pdb_path", help="Path to pdb file containing structure")
    parser.add_argument("--mps_workers", type=int, default=1, help="Number of MPS processes per GPU")
    parser.add_argument("--n_eq_steps", default=200_000, type=int, help="Number of steps to perform equilibration")
    parser.add_argument("--n_frames", default=2000, type=int, help="Number of frames to generation")
    parser.add_argument("--steps_per_frame", default=400, type=int, help="Steps per frame")
    parser.add_argument(
        "--n_windows", default=DEFAULT_NUM_WINDOWS, type=int, help="Max number of windows from bisection"
    )
    parser.add_argument("--min_overlap", default=0.667, type=float, help="Overlap to target in bisection")
    parser.add_argument(
        "--target_overlap", default=0.667, type=float, help="Overlap to optimize final HREX schedule to"
    )
    parser.add_argument("--seed", default=2025, type=int, help="Seed")
    parser.add_argument("--forcefield", default=DEFAULT_FF)
    parser.add_argument(
        "--n_gpus", default=None, type=int, help="Number of GPUs to use, defaults to all GPUs if not provided"
    )
    parser.add_argument(
        "--water_sampling_padding",
        type=float,
        default=0.4,
        help="How much to expand the radius of the sphere used for water sampling (nm). Half of the largest intramolecular distance is used as the starting radius to which the padding is added: dist/2 + padding",
    )
    parser.add_argument(
        "--rest_max_temperature_scale",
        default=3.0,
        type=float,
        help="Maximum scale factor for the effective temperature of REST-softened interactions. Setting to 1.0 effectively disables REST.",
    )
    parser.add_argument(
        "--rest_temperature_scale_interpolation",
        default="exponential",
        type=str,
        help="Functional form to use for temperature scale interpolation in REST",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to output results, else generates a directory based on the time"
    )
    parser.add_argument("--local_md_k", default=10_000.0, type=float, help="Local MD k parameter")
    parser.add_argument("--local_md_radius", default=1.2, type=float, help="Local MD radius")
    parser.add_argument("--local_md_free_reference", action="store_true")
    parser.add_argument(
        "--local_md_steps",
        default=0,
        type=int,
        help="Number of steps to run with Local MD. Must be less than or equal to --steps_per_frame. If set to 0, no local MD is run",
    )
    parser.add_argument(
        "--store_trajectories",
        action="store_true",
        help="Store the trajectories of the simulations. Can take up a large amount of space",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing predictions, otherwise will skip the completed legs",
    )
    parser.add_argument(
        "--experimental_field", default="kcal/mol experimental dG", help="Field that contains the experimental label."
    )
    parser.add_argument(
        "--experimental_units",
        default="kcal/mol",
        choices=["kcal/mol", "kJ/mol", "uM", "nM"],
        help="Units of the experimental label.",
    )
    args = parser.parse_args()
    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"abfe_graph_{date_str}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with Chem.SDWriter(dest_dir / "mols.sdf") as writer:
        for mol in mols_by_name.values():
            writer.write(mol)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    with open(dest_dir / "ff.py", "w") as ofs:
        ofs.write(ff.serialize())

    num_gpus = args.n_gpus
    if num_gpus is None:
        num_gpus = get_gpu_count()

    # Set max_tasks_per_child=1 to reduce potential for accumulating memory
    pool = CUDAMPSPoolClient(num_gpus, workers_per_gpu=args.mps_workers, max_tasks_per_child=1)
    pool.verify()
    futures = []
    for name, mol in mols_by_name.items():
        # TBD: Fix this
        mol_radius = get_radius_of_mol_pair(mol, mol)
        md_params = MDParams(
            n_eq_steps=args.n_eq_steps,
            n_frames=args.n_frames,
            steps_per_frame=args.steps_per_frame,
            seed=args.seed,
            hrex_params=HREXParams(
                optimize_target_overlap=args.target_overlap,
                rest_params=RESTParams(args.rest_max_temperature_scale, args.rest_temperature_scale_interpolation),
            ),
            local_md_params=LocalMDParams(
                args.local_md_steps,
                k=args.local_md_k,
                min_radius=args.local_md_radius,
                max_radius=args.local_md_radius,
                freeze_reference=not args.local_md_free_reference,
            )
            if args.local_md_steps > 0
            else None,
            water_sampling_params=WaterSamplingParams(radius=mol_radius + args.water_sampling_padding),
        )
        for leg in (COMPLEX_LEG, SOLVENT_LEG):
            fut = pool.submit(
                run_abfe,
                file_client,
                Path(name),
                mol,
                leg,
                ff,
                args.pdb_path,
                md_params,
                args.n_windows,
                args.min_overlap,
                args.store_trajectories,
                args.force_overwrite,
            )
            futures.append(fut)
    for fut in iterate_completed_futures(futures):
        fut.result()


if __name__ == "__main__":
    main()
