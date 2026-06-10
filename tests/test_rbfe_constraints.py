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

"""Integration test for hydrogen constraints in the RBFE (single topology) pathway."""

import numpy as np

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_TEMP
from tmd.fe import atom_mapping
from tmd.fe.free_energy import get_context
from tmd.fe.rbfe import setup_initial_state
from tmd.fe.single_topology import SingleTopology
from tmd.ff import Forcefield
from tmd.lib import ConstrainedLangevinIntegrator
from tmd.potentials import HarmonicBond
from tmd.potentials.potential import get_bound_potential_by_type
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology


def _hydrogen_heavy_bonds(bond_idxs, bond_params, is_h):
    """Return (i, j, r0) for each bond connecting exactly one hydrogen to one heavy atom."""
    out = []
    for (i, j), (_, r0) in zip(bond_idxs, bond_params):
        i, j = int(i), int(j)
        if bool(is_h[i]) != bool(is_h[j]):
            out.append((i, j, float(r0)))
    return out


def test_rbfe_vacuum_constrain_hydrogens_holds_constraints():
    """A constrained single-topology vacuum state must hold every X-H bond rigid under MD, and
    the constrained X-H stretches must be removed from the harmonic bond potential."""
    # Simple-charge forcefield avoids the OpenEye AM1CCC dependency.
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()

    # Build a constraint-compatible core (stage 1) and prune it for constraints (stage 2).
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS, constrain_hydrogens=True)[0]
    st = SingleTopology(mol_a, mol_b, core, ff, constrain_hydrogens=True)

    lamb = 0.0

    masses = np.array(st.combine_masses(use_hmr=False))
    is_h = masses < 1.5

    # Reference unconstrained state (same topology) gives the full bond list and equilibrium lengths.
    state_unc = setup_initial_state(st, lamb, None, DEFAULT_TEMP, 2024, verify_constraints=False)
    bond_unc = get_bound_potential_by_type(state_unc.potentials, HarmonicBond)
    hx = _hydrogen_heavy_bonds(bond_unc.potential.idxs, bond_unc.params, is_h)
    assert len(hx) > 0

    # Constrained state.
    state = setup_initial_state(
        st, lamb, None, DEFAULT_TEMP, 2024, verify_constraints=False, constrain_hydrogens=True
    )
    assert isinstance(state.integrator, ConstrainedLangevinIntegrator)

    bond_con = get_bound_potential_by_type(state.potentials, HarmonicBond)
    # Every X-H stretch was replaced by a rigid constraint and dropped from the harmonic potential.
    assert len(bond_con.potential.idxs) == len(bond_unc.potential.idxs) - len(hx)

    ctxt = get_context(state)
    xs, _ = ctxt.multiple_steps(500, 50)
    assert len(xs) > 0

    pairs = np.array([(i, j) for i, j, _ in hx])
    targets = np.array([r0 for *_, r0 in hx])
    for frame in xs:
        d = np.linalg.norm(frame[pairs[:, 0]] - frame[pairs[:, 1]], axis=1)
        # float32 SHAKE/RATTLE tolerance
        np.testing.assert_allclose(d, targets, atol=5e-3)
