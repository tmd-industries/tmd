# Copyright 2019-2025, Relay Therapeutics
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

# test code for combining host guest systems and ensuring that parameters
# and lambda configurations are correct
import numpy as np
import pytest

from tmd import potentials
from tmd.constants import NBParamIdx
from tmd.fe.single_topology import SingleTopology
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

pytestmark = [pytest.mark.nocuda]


@pytest.fixture(scope="module")
def hif2a_ligand_pair_single_topology():
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    return SingleTopology(mol_a, mol_b, core, forcefield)


@pytest.fixture(scope="module")
def complex_host_system():
    # (YTZ): we need to clean this up later, since it uses a pre-solvated xml file.
    ff = Forcefield.load_default()
    with path_to_internal_file("tmd.testsystems.data", "hif2a_nowater_min.pdb") as path_to_pdb:
        host_config = builders.build_protein_system(str(path_to_pdb), ff.protein_ff, ff.water_ff)
    return host_config.host_system, host_config.masses, host_config.conf.shape[0], host_config.omm_topology


@pytest.fixture(scope="module")
def solvent_host_system():
    ff = Forcefield.load_default()
    host_config = builders.build_water_system(3.0, ff.water_ff)
    return host_config.host_system, host_config.masses, host_config.conf.shape[0], host_config.omm_topology


@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_bonded(host_system_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    host_sys, host_masses, num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    def check_bonded_idxs_consistency(bonded_idxs, num_host_idxs):
        for b_idx, atom_idxs in enumerate(bonded_idxs):
            if b_idx < num_host_idxs:
                assert np.all(atom_idxs < num_host_atoms)
            else:
                assert np.all(atom_idxs >= num_host_atoms)

    # generate host guest system
    host_guest_sys = st.combine_with_host(host_sys, lamb, num_water_atoms, omm_topology)

    # check bonds
    check_bonded_idxs_consistency(host_guest_sys.bond.potential.idxs, len(host_sys.bond.potential.idxs))
    check_bonded_idxs_consistency(host_guest_sys.angle.potential.idxs, len(host_sys.angle.potential.idxs))
    check_bonded_idxs_consistency(host_guest_sys.proper.potential.idxs, len(host_sys.proper.potential.idxs))
    check_bonded_idxs_consistency(host_guest_sys.improper.potential.idxs, len(host_sys.improper.potential.idxs))
    check_bonded_idxs_consistency(host_guest_sys.chiral_atom.potential.idxs, 0)
    check_bonded_idxs_consistency(host_guest_sys.chiral_bond.potential.idxs, 0)
    check_bonded_idxs_consistency(host_guest_sys.nonbonded_pair_list.potential.idxs, 0)


@pytest.mark.parametrize("lamb", [0.0, 1.0])
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded(host_system_fixture, lamb, hif2a_ligand_pair_single_topology, request):
    # test bonded and nonbonded parameters are correct at the end-states.
    # 1) we expected bonded idxs in the ligand to be shifted by num_host_atoms
    # 2) we expected nonbonded lambda_idxs to be shifted
    # 3) we expected nonbonded parameters on the core to be linearly interpolated

    st = hif2a_ligand_pair_single_topology
    host_sys, host_masses, num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    hgs = st.combine_with_host(host_sys, lamb, num_water_atoms, omm_topology)
    assert isinstance(hgs.nonbonded_all_pairs.potential, potentials.Nonbonded)
    # Should be using all of the atoms
    assert hgs.nonbonded_all_pairs.potential.atom_idxs is None

    # 2) decoupling parameters for host-guest interactions
    # 2a) w offsets
    potential = hgs.nonbonded_all_pairs.potential
    params = hgs.nonbonded_all_pairs.params

    # NBIxnGroup has the ligand interaction parameters
    assert isinstance(potential, potentials.Nonbonded)
    w_coords = params[:, NBParamIdx.W_IDX]

    for a_idx, w in enumerate(w_coords):
        if a_idx < num_host_atoms:
            # host atom
            assert w == 0.0
        else:
            # guest atom
            guest_atom_idx = a_idx - num_host_atoms
            indicator = st.c_flags[guest_atom_idx]
            if indicator == 0:
                # core
                assert w == 0.0
            elif indicator == 1:
                # mol_a dummy
                if lamb == 0.0:
                    assert w == 0.0
                elif lamb == 1.0:
                    assert w == potential.cutoff
            elif indicator == 2:
                # mol_b dummy
                if lamb == 0.0:
                    assert w == potential.cutoff
                elif lamb == 1.0:
                    assert w == 0.0
            else:
                assert 0

    # 2b) nonbonded parameter interpolation checks
    mol_a_charges = st.ff.q_handle.parameterize(st.mol_a)
    mol_a_sig_eps = st.ff.lj_handle.parameterize(st.mol_a)

    mol_b_charges = st.ff.q_handle.parameterize(st.mol_b)
    mol_b_sig_eps = st.ff.lj_handle.parameterize(st.mol_b)

    for a_idx, (test_q, test_sig, test_eps, _) in enumerate(params):
        if a_idx < num_host_atoms:
            continue

        guest_atom_idx = a_idx - num_host_atoms
        indicator = st.c_flags[guest_atom_idx]

        # dummy atom qlj parameters are arbitrary (since they will be decoupled via lambda parameters)
        if indicator != 0:
            continue

        if lamb == 0.0:
            # should resemble mol_a at lambda=0
            ref_q = mol_a_charges[st.c_to_a[guest_atom_idx]]
            ref_sig, ref_eps = mol_a_sig_eps[st.c_to_a[guest_atom_idx]]

            assert np.float32(ref_q) == np.float32(test_q)
            assert np.float32(test_sig) == np.float32(ref_sig)
            assert np.float32(test_eps) == np.float32(ref_eps)

        elif lamb == 1.0:
            # should resemble mol_b at lambda=1
            ref_q = mol_b_charges[st.c_to_b[guest_atom_idx]]
            ref_sig, ref_eps = mol_b_sig_eps[st.c_to_b[guest_atom_idx]]

            assert np.float32(ref_q) == np.float32(test_q)
            assert np.float32(test_sig) == np.float32(ref_sig)
            assert np.float32(test_eps) == np.float32(ref_eps)


@pytest.mark.parametrize("lamb", np.random.default_rng(2022).uniform(0.01, 0.99, (10,)))
@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_combined_parameters_nonbonded_intermediate(
    host_system_fixture, lamb, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    st = hif2a_ligand_pair_single_topology
    host_sys, host_masses, num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)
    num_host_atoms = len(host_masses)

    hgs = st.combine_with_host(host_sys, lamb, num_water_atoms, omm_topology)
    potential = hgs.nonbonded_all_pairs.potential
    params = hgs.nonbonded_all_pairs.params
    assert isinstance(potential, potentials.Nonbonded)

    guest_params = np.array(params[num_host_atoms:])
    ws_core = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 0]
    ws_a = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 1]
    ws_b = [w for flag, (_, _, _, w) in zip(st.c_flags, guest_params) if flag == 2]

    # core atoms fixed at w = 0
    assert all(w == 0.0 for w in ws_core)

    # dummy groups have consistent w coords
    assert len(set(ws_a)) == 1
    assert len(set(ws_b)) == 1
    (w_a,) = set(ws_a)
    (w_b,) = set(ws_b)

    # w in [0, cutoff]
    assert 0 <= w_a <= potential.cutoff
    assert 0 <= w_b <= potential.cutoff

    if lamb < 0.5:
        assert w_a <= w_b
    else:
        assert w_b <= w_a


@pytest.mark.parametrize("host_system_fixture", ["solvent_host_system", "complex_host_system"])
def test_nonbonded_host_params_independent_of_lambda(
    host_system_fixture, hif2a_ligand_pair_single_topology: SingleTopology, request
):
    st = hif2a_ligand_pair_single_topology
    host_sys, _, num_water_atoms, omm_topology = request.getfixturevalue(host_system_fixture)

    num_host_atoms = host_sys.nonbonded_all_pairs.potential.num_atoms

    def get_nonbonded_host_params(lamb):
        return st.combine_with_host(host_sys, lamb, num_water_atoms, omm_topology).nonbonded_all_pairs.params[
            :num_host_atoms
        ]

    params0 = get_nonbonded_host_params(0.0)
    for lamb in np.linspace(0.1, 1, 10):
        params = get_nonbonded_host_params(lamb)
        np.testing.assert_array_equal(params, params0)
