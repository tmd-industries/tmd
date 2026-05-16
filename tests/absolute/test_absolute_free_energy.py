from dataclasses import replace

import numpy as np

from tmd.constants import DEFAULT_TEMP
from tmd.fe import utils
from tmd.fe.absolute.abfe import get_initial_state, optimize_abfe_initial_state, sample_for_restraints
from tmd.fe.absolute.free_energy import AbsoluteBindingFreeEnergy, RestraintParams
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
    MDParams,
)
from tmd.fe.rbfe import setup_optimized_host
from tmd.fe.topology import BaseTopology
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.potentials import (
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file


def test_absolute_binding_free_energy():
    mol, _, _ = get_hif2a_ligand_pair_single_topology()
    # Use the simple charges because it is faster
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as protein_path:
        host_config = builders.build_protein_system(
            str(protein_path), ff.protein_ff, ff.water_ff, mols=[mol], box_margin=0.1
        )

    md_params = MDParams(
        n_frames=5,
        n_eq_steps=1000,
        steps_per_frame=10,
        seed=2026,
    )

    host_config = setup_optimized_host(host_config, [mol], ff)
    host_conf = host_config.conf
    bt = BaseTopology(mol, ff)
    temperature = DEFAULT_TEMP

    restraint_params = RestraintParams()

    afe = AbsoluteFreeEnergy(mol, bt)
    unbound_potentials, params, masses = afe.prepare_host_edge(ff, host_config, 0.0)
    assert len(unbound_potentials) == 6

    initial_state = get_initial_state(afe, ff, host_config, host_conf, temperature, md_params.seed, 0.0)
    minimized_state = optimize_abfe_initial_state(initial_state)
    # TBD: How many frames do you want from here?
    sample_md_params = replace(md_params, n_eq_steps=10_000)
    trj = sample_for_restraints(minimized_state, sample_md_params, replicas=1)

    afe = AbsoluteBindingFreeEnergy.create(bt, host_config, trj, restraint_params)
    assert len(afe.rec_atoms) == 3
    assert len(afe.lig_atoms) == 3

    unbound_potentials, params, masses = afe.prepare_host_edge(ff, host_config, 0.0)
    np.testing.assert_array_equal(masses, np.concatenate([host_config.masses, utils.get_mol_masses(mol)]))

    assert set(type(pot) for pot in unbound_potentials) == {
        HarmonicBond,
        HarmonicAngle,
        PeriodicTorsion,
        NonbondedPairListPrecomputed,
        Nonbonded,
    }

    # 6 potentials typically, add 3 for the boresch restraints
    assert len(unbound_potentials) == 6 + 3
