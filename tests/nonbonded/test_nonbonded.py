from dataclasses import replace
from typing import Optional

import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets, prepare_system_params, prepare_water_system
from numpy.typing import NDArray

from tmd import potentials
from tmd.ff import Forcefield
from tmd.md import builders

np.set_printoptions(linewidth=500)

pytestmark = [pytest.mark.memcheck]


def exclusion_atoms_to_exclusion_pairs(atoms):
    exclusion_idxs = []

    for idx, i in enumerate(atoms):
        for j in atoms[idx + 1 :]:
            exclusion_idxs.append((i, j))
    return exclusion_idxs


@pytest.mark.parametrize("precision", [np.float64, np.float32])
def test_nblist_hilbert(example_conf, example_box, example_nonbonded_potential, precision):
    """
    This test makes sure that hilbert ordering has no impact on numerical results. The
    computed forces, energies, etc. should be bitwise identical.
    """

    np.random.seed(2021)
    np.set_printoptions(precision=16)

    nonbonded_fn = example_nonbonded_potential.potential
    N = example_conf.shape[0]

    ref_nonbonded_impl = replace(nonbonded_fn, disable_hilbert_sort=True).to_gpu(precision).unbound_impl
    test_nonbonded_impl = nonbonded_fn.to_gpu(precision).unbound_impl

    padding = nonbonded_fn.nblist_padding
    deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
    divisor = 0.5 * (2 * np.sqrt(3)) / padding
    # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
    deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

    for d in deltas:
        assert np.linalg.norm(d) < padding / 2

    xs = [example_conf, example_conf + deltas]

    # under pure fixed point accumulation the results should be identical.
    for x in xs:
        x = x.astype(precision)
        params = example_nonbonded_potential.params.astype(precision)
        box = example_box.astype(precision)

        ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, params, box)
        test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, params, box)

        np.testing.assert_array_equal(ref_du_dx, test_du_dx)
        np.testing.assert_array_equal(ref_du_dp, test_du_dp)
        np.testing.assert_array_equal(ref_u, test_u)

        ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, params, box)
        test_du_dx = test_nonbonded_impl.execute_du_dx(x, params, box)

        for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
            if np.linalg.norm(a - b) != 0:
                print(idx, a, b)
                assert 0
            np.testing.assert_array_equal(a, b)

        np.testing.assert_array_equal(ref_du_dx, test_du_dx)


@pytest.mark.parametrize("precision", [np.float64])
def test_nblist_rebuild(example_conf, example_box, example_nonbonded_potential, precision):
    """
    This test makes sure that periodically rebuilding the neighborlist has no impact on numerical results. The
    computed forces, energies, etc. should be bitwise identical.

    Only passes under float64
    """

    nonbonded_fn = example_nonbonded_potential.potential
    N = example_conf.shape[0]

    np.random.seed(2021)

    ref_nonbonded_impl = replace(nonbonded_fn, nblist_padding=0.0).to_gpu(precision).unbound_impl

    padding = 0.1

    # rebuild only when deltas have moved more than padding/2 angstroms
    test_nonbonded_impl = replace(nonbonded_fn, nblist_padding=padding).to_gpu(precision).unbound_impl

    deltas = np.random.rand(N, 3) - 0.5  # [-0.5, +0.5]
    divisor = 0.5 * (2 * np.sqrt(3)) / padding
    # if deltas are kept under +- p/(2*sqrt(3)) then no rebuild gets triggered
    deltas = deltas / divisor  # exactly within bounds, and should not trigger a rebuild

    for d in deltas:
        assert np.linalg.norm(d) < padding / 2

    xs = [example_conf, example_conf + deltas]

    # under pure fixed point accumulation the results should be identical.
    for x in xs:
        ref_du_dx, ref_du_dp, ref_u = ref_nonbonded_impl.execute(x, example_nonbonded_potential.params, example_box)
        test_du_dx, test_du_dp, test_u = test_nonbonded_impl.execute(x, example_nonbonded_potential.params, example_box)

        np.testing.assert_array_equal(ref_du_dx, test_du_dx)
        np.testing.assert_array_equal(ref_du_dp, test_du_dp)
        np.testing.assert_array_equal(ref_u, test_u)

        ref_du_dx = ref_nonbonded_impl.execute_du_dx(x, example_nonbonded_potential.params, example_box)
        test_du_dx = test_nonbonded_impl.execute_du_dx(x, example_nonbonded_potential.params, example_box)

        for idx, (a, b) in enumerate(zip(ref_du_dx, test_du_dx)):
            if np.linalg.norm(a - b) != 0:
                print(idx, a, b)
                print(a)
                print(b)
                assert 0
            np.testing.assert_array_equal(a, b)

        np.testing.assert_array_equal(ref_du_dx, test_du_dx)


@pytest.mark.parametrize("select_atom_indices", [False, True])
# we can't go bigger than this due to memory limitations in the the reference platform.
@pytest.mark.parametrize("num_atoms", [33, 65, 231, 1050, 3080])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_correctness(
    rng, example_conf, example_box, example_nonbonded_potential, select_atom_indices, num_atoms, precision, atol, rtol
):
    """
    Test against the reference jax platform for correctness.
    """
    nonbonded_fn = example_nonbonded_potential.potential

    test_conf = example_conf[:num_atoms].astype(precision)

    # strip out parts of the system
    test_exclusions = []
    test_scales = []
    for (i, j), (sa, sb) in zip(nonbonded_fn.exclusion_idxs, nonbonded_fn.scale_factors):
        if i < num_atoms and j < num_atoms:
            test_exclusions.append((i, j))
            test_scales.append((sa, sb))

    test_exclusions = np.array(test_exclusions, dtype=np.int32)
    test_scales = np.array(test_scales, dtype=precision)
    test_params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)

    atom_idxs: Optional[NDArray] = None
    if select_atom_indices:
        atom_idxs = np.array(rng.choice(num_atoms, num_atoms // 2, replace=False), dtype=np.int32)
    potential = potentials.Nonbonded(
        num_atoms, test_exclusions, test_scales, nonbonded_fn.beta, nonbonded_fn.cutoff, atom_idxs=atom_idxs
    )

    test_box = example_box.astype(precision)

    GradientTest().compare_forces(
        test_conf, test_params, test_box, potential, potential.to_gpu(precision), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("select_atom_indices", [False, True])
@pytest.mark.parametrize("num_atoms", [33, 65, 231, 1050])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_systems", [1, 2, 4, 8])
@pytest.mark.parametrize("random_idx_lengths", [True, False])
def test_batch_correctness(
    rng,
    example_conf,
    example_box,
    example_nonbonded_potential,
    select_atom_indices,
    num_atoms,
    precision,
    num_systems,
    random_idx_lengths,
):
    """
    Test against the unbatched GPU platform, which should be identical. Don't trust results if test_correctness does not also pass.
    """
    nonbonded_fn = example_nonbonded_potential.potential

    # strip out parts of the system
    test_exclusions = []
    test_scales = []
    for (i, j), (sa, sb) in zip(nonbonded_fn.exclusion_idxs, nonbonded_fn.scale_factors):
        if i < num_atoms and j < num_atoms:
            test_exclusions.append((i, j))
            test_scales.append((sa, sb))

    atom_idxs: Optional[NDArray] = None
    if select_atom_indices:
        if random_idx_lengths:
            sizes = rng.integers(1, num_atoms, size=num_systems)
            atom_idxs = [rng.choice(num_atoms, size, replace=False).astype(np.int32) for size in sizes]
        else:
            atom_idxs = [
                np.asarray(rng.choice(num_atoms, num_atoms // 2, replace=False), dtype=np.int32)
                for _ in range(num_systems)
            ]

    exclusion_idxs = [np.array(test_exclusions, dtype=np.int32)] * num_systems
    scales = [np.array(test_scales, dtype=precision)] * num_systems

    coords = np.stack(
        [
            (example_conf[:num_atoms] + rng.uniform(-1e3, 1e3, size=(num_atoms, 3))).astype(precision)
            for _ in range(num_systems)
        ]
    ).astype(precision)
    boxes = np.stack([example_box for _ in coords]).astype(precision)
    params = np.stack(
        [example_nonbonded_potential.params[:num_atoms, :].astype(precision) for _ in range(num_systems)]
    ).astype(precision)

    potential = potentials.Nonbonded(
        num_atoms, exclusion_idxs, scales, nonbonded_fn.beta, nonbonded_fn.cutoff, atom_idxs=atom_idxs
    )

    non_zero_nrg = False
    for w_params in gen_nonbonded_params_with_4d_offsets(rng, params, nonbonded_fn.cutoff):
        batch_ub = potential.to_gpu(precision).unbound_impl
        batch_du_dx, batch_du_dp, batch_u = batch_ub.execute_dim(coords, w_params, boxes, 1, 1, 1)

        comp_batch_du_dx, comp_batch_du_dp, comp_batch_u = batch_ub.execute_dim(coords, w_params, boxes, 1, 1, 1)
        # Verify running the potential twice gives the same result
        np.testing.assert_array_equal(comp_batch_du_dx, batch_du_dx)
        np.testing.assert_array_equal(comp_batch_du_dp, batch_du_dp)
        np.testing.assert_array_equal(comp_batch_u, batch_u)
        for i in range(num_systems):
            ref_nb = potentials.Nonbonded(
                num_atoms,
                exclusion_idxs[i],
                scales[i],
                nonbonded_fn.beta,
                nonbonded_fn.cutoff,
                atom_idxs=atom_idxs[i] if atom_idxs is not None else None,
            )
            ref_du_dx, ref_du_dp, ref_u = ref_nb.to_gpu(precision).unbound_impl.execute(
                coords[i], w_params[i], boxes[i]
            )
            np.testing.assert_array_equal(batch_du_dx[i], ref_du_dx, err_msg=f"Batch {i}")
            np.testing.assert_array_equal(batch_du_dp[i], ref_du_dp, err_msg=f"Batch {i}")
            np.testing.assert_array_equal(batch_u[i], ref_u, err_msg=f"Batch {i}")
            if ref_u != 0.0:
                non_zero_nrg = True
    assert non_zero_nrg, "Didn't validate energies"


@pytest.mark.parametrize(
    "precision, rtol, atol, du_dp_rtol, du_dp_atol",
    [
        (np.float64, 1e-8, 1e-10, 1e-6, 1e-9),
        (np.float32, 1e-4, 3e-5, 1e-3, 1e-3),
    ],
)
def test_nblist_box_resize(precision, rtol, atol, du_dp_rtol, du_dp_atol):
    # test that running the coordinates under two different boxes produces correct results
    # since we should be rebuilding the nblist when the box sizes change.
    ff = Forcefield.load_default()
    host_config = builders.build_water_system(3.0, ff.water_ff)
    box = host_config.box
    test_bp = host_config.host_system.nonbonded_all_pairs
    assert test_bp.params is not None

    big_box = box + np.eye(3) * 1000

    # (ytz): note the ordering should be from large box to small box. though in the current code
    # the rebuild is triggered as long as the box *changes*.
    for test_box in [big_box, box]:
        GradientTest().compare_forces(
            host_config.conf.astype(precision),
            test_bp.params.astype(precision),
            test_box.astype(precision),
            test_bp.potential,
            test_bp.potential.to_gpu(precision),
            rtol=rtol,
            atol=atol,
            du_dp_rtol=du_dp_rtol,
            du_dp_atol=du_dp_atol,
        )


@pytest.mark.parametrize("cutoff", [1.2])
@pytest.mark.parametrize("size", [33, 231, 1050])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
def test_nonbonded_water(size, cutoff, precision, rtol, atol):
    np.random.seed(4321)
    ff = Forcefield.load_default()
    host_config = builders.build_water_system(3.0, ff.water_ff)
    coords = host_config.conf[:size]

    # E = 0 # DEBUG!
    charge_params, potential = prepare_water_system(coords, p_scale=5.0, cutoff=cutoff)
    test_impl = potential.to_gpu(precision)
    for params in gen_nonbonded_params_with_4d_offsets(np.random.default_rng(2022), charge_params, cutoff):
        GradientTest().compare_forces(
            coords.astype(precision),
            params.astype(precision),
            host_config.box.astype(precision),
            potential,
            test_impl,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("precision,rtol", [(np.float64, 1e-8), (np.float32, 1e-4)])
def test_nonbonded_exclusions(precision, rtol):
    """This test verifies behavior when two particles are arbitrarily
    close but are marked as excluded to ensure proper cancellation
    of exclusions occur in the fixed point math.
    """

    np.random.seed(2020)
    ff = Forcefield.load_default()

    host_config = builders.build_water_system(3.0, ff.water_ff)
    N = 126  # multiple of 3
    test_coords = host_config.conf[:N]
    box = np.eye(3) * 3

    EA = 10

    atom_idxs = np.arange(test_coords.shape[0])

    # pick a set of atoms that will be mutually excluded from each other.
    # we will need to set their exclusions manually
    exclusion_atoms = np.random.choice(atom_idxs, size=EA, replace=False)
    exclusion_idxs = exclusion_atoms_to_exclusion_pairs(exclusion_atoms)

    E = len(exclusion_idxs)
    exclusion_idxs = np.array(exclusion_idxs, dtype=np.int32)
    scales = np.ones((E, 2), dtype=np.float64)
    # perturb the system
    for idx in exclusion_atoms:
        test_coords[idx] = np.zeros(3) + np.random.rand() / 1000 + 2

    beta = 2.0
    cutoff = 1.2

    potential = potentials.Nonbonded(N, exclusion_idxs, scales, beta, cutoff)

    params = prepare_system_params(test_coords, cutoff)

    GradientTest().compare_forces(
        test_coords.astype(precision),
        params.astype(precision),
        box.astype(precision),
        potential,
        potential.to_gpu(precision),
        rtol,
    )
