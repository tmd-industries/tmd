# Copyright 2025 Justin Gullingsrud
# Modifications Copyright 2025-2026 Forrest York, Justin Gullingsrud
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


from dataclasses import replace

import numpy as np

from tmd import potentials
from tmd.constants import DEFAULT_PRESSURE
from tmd.fe import model_utils
from tmd.fe.constraints import build_constraints, remove_constrained_bonds
from tmd.fe.free_energy import (
    InitialState,
    Trajectory,
    get_batched_context,
    get_context,
    sample_with_context_iter,
)
from tmd.fe.rbfe import (
    optimize_coordinates,
)
from tmd.lib import ConstrainedLangevinIntegrator, LangevinIntegrator, MonteCarloBarostat
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.md.thermostat.utils import sample_velocities
from tmd.potentials.potential import get_potential_by_type


def get_initial_state(
    afe, ff, host_config, host_conf, temperature, seed, lamb, constrain_hydrogens: bool = False
) -> InitialState:
    """Get initial state at a particular lambda for use with ABFE"""
    ubps, params, masses = afe.prepare_host_edge(ff, host_config, lamb)
    x0 = afe.prepare_combined_coords(host_coords=host_conf)

    bond_potential = get_potential_by_type(ubps, potentials.HarmonicBond)

    hmr_masses = model_utils.apply_hmr(masses, bond_potential.idxs)
    group_idxs = get_group_indices(get_bond_list(bond_potential), len(masses))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 25, seed)
    box0 = host_config.box

    v0 = sample_velocities(hmr_masses, temperature, seed)
    num_ligand_atoms = afe.mol.GetNumAtoms()
    num_total_atoms = len(x0)
    ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms, dtype=np.int32)

    dt = 2.5e-3
    friction = 1.0

    constrained_bond_idxs = None
    constrained_bond_params = None
    if constrain_hydrogens:
        bond_params = next(p for u, p in zip(ubps, params) if u is bond_potential)
        angle_potential = get_potential_by_type(ubps, potentials.HarmonicAngle)
        angle_params = next(p for u, p in zip(ubps, params) if u is angle_potential)
        clusters = build_constraints(
            bond_potential.idxs,
            bond_params,
            masses,
            angle_potential.idxs,
            angle_params,
            rigid_water=True,
        )
        # Replace the constrained harmonic stretches with rigid SHAKE constraints.
        constrained_bond_idxs, constrained_bond_params = remove_constrained_bonds(
            bond_potential.idxs, bond_params, clusters.constrained_bond_rows
        )
        intg = ConstrainedLangevinIntegrator(temperature, dt, friction, hmr_masses, seed, clusters)
    else:
        intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)

    bps = []
    for ubp, param in zip(ubps, params):
        if constrain_hydrogens and ubp is bond_potential:
            bp = potentials.HarmonicBond(ubp.num_atoms, constrained_bond_idxs).bind(constrained_bond_params)
        else:
            bp = ubp.bind(param)
        bps.append(bp)

    protein_idxs = np.arange(
        len(host_config.conf) - host_config.num_water_atoms - host_config.num_membrane_atoms, dtype=np.int32
    )

    return InitialState(
        bps,
        intg,
        baro,
        x0,
        v0,
        box0,
        lamb,
        ligand_idxs,
        protein_idxs,
        interacting_atoms=ligand_idxs if lamb == 0.0 else None,
    )


def optimize_abfe_initial_state(state: InitialState) -> InitialState:
    # Disable min_cutoff check
    x_opted = optimize_coordinates([state], min_cutoff=None)
    assert len(x_opted) == 1
    return replace(state, x0=x_opted[0])


def sample_for_restraints(initial_state: InitialState, md_params, replicas: int = 1) -> Trajectory:
    if replicas > 1:
        ctxt = get_batched_context([initial_state] * replicas, md_params)
    else:
        ctxt = get_context(initial_state, md_params)
    traj = Trajectory.empty()
    for frame, box, _ in sample_with_context_iter(
        ctxt, md_params, initial_state.integrator.temperature, initial_state.ligand_idxs, 1
    ):
        if replicas == 1:
            traj.frames.extend([frame[-1]])
            traj.boxes.extend([box[-1]])
        else:
            for x, b in zip(frame[-1], box[-1]):
                traj.frames.extend([x])
                traj.boxes.extend([b])
    return traj
