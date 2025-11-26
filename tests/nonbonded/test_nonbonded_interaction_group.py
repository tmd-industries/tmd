import re

import jax.numpy as jnp
import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets

from tmd.potentials import FanoutSummedPotential, Nonbonded, NonbondedInteractionGroup

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_interaction_group_invalid_indices():
    with pytest.raises(RuntimeError, match=re.escape("provide at least one batch of row atom indices")):
        NonbondedInteractionGroup(1, [], 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("atom indices must be unique")):
        NonbondedInteractionGroup(3, np.array([1, 1], dtype=np.int32), 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("index values must be greater or equal to zero")):
        NonbondedInteractionGroup(3, np.array([1, -1], dtype=np.int32), 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("index values must be less than N")):
        NonbondedInteractionGroup(3, np.array([1, 100], dtype=np.int32), 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("col_atom_idxs must be nonempty")):
        NonbondedInteractionGroup(3, np.array([0, 1, 2], dtype=np.int32), 1.0, 1.0).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("num row idxs must be <= N(3)")):
        NonbondedInteractionGroup(
            3, np.array([0, 1, 2, 3], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([5], dtype=np.int32)
        ).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("index values must be less than N(3)")):
        NonbondedInteractionGroup(
            3, np.array([0, 1], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([2, 3, 4, 5], dtype=np.int32)
        ).to_gpu(np.float64).unbound_impl

    # okay; overlapping
    NonbondedInteractionGroup(
        3, np.array([0, 1], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([0, 1], dtype=np.int32)
    ).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("index values must be less than N(3)")):
        NonbondedInteractionGroup(
            3, np.array([0, 1], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([2, 3], dtype=np.int32)
        ).to_gpu(np.float64).unbound_impl

    # non disjoint, and non-overlapping
    with pytest.raises(RuntimeError, match=re.escape("row and col indices are neither disjoint nor overlapping")):
        NonbondedInteractionGroup(
            3, np.array([1, 2], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([0, 1], dtype=np.int32)
        ).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match=re.escape("num row atoms(5) must be <= num col atoms(3) if non-disjoint")):
        NonbondedInteractionGroup(
            6, np.array([5, 1, 3, 2, 4], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([3, 2, 4], dtype=np.int32)
        ).to_gpu(np.float64).unbound_impl

    # Ok for different idxs
    NonbondedInteractionGroup(
        3, np.array([0, 1], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([2], dtype=np.int32)
    ).to_gpu(np.float64).unbound_impl

    with pytest.raises(RuntimeError, match="each batch of column indices must be one dimensional"):
        NonbondedInteractionGroup(
            3, [np.array([0], dtype=np.int32)], 1.0, 1.0, col_atom_idxs=[np.array([1, 2], dtype=np.int32).reshape(1, 2)]
        ).to_gpu(np.float64).unbound_impl

    # Test that if we have disjoint row_atom/atom_idxs we're not allowed to set compute_col_grads=False
    impl = (
        NonbondedInteractionGroup(
            6, np.array([0, 1, 2], dtype=np.int32), 1.0, 1.0, col_atom_idxs=np.array([3, 4, 5], dtype=np.int32)
        )
        .to_gpu(np.float64)
        .unbound_impl
    )
    with pytest.raises(
        RuntimeError, match=re.escape("compute_col_grads must be true if interaction_type_ is DISJOINT")
    ):
        impl.set_compute_col_grads(False)  # type: ignore


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_nonbonded_interaction_group_zero_interactions(rng: np.random.Generator, precision):
    num_atoms = 33
    num_atoms_ligand = 15
    beta = 2.0
    cutoff = 1.1
    box = 10.0 * np.eye(3)
    conf = rng.uniform(0, 1, size=(num_atoms, 3)).astype(precision)
    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    # shift ligand atoms in x by twice the cutoff
    conf[ligand_idxs, 0] += 2 * cutoff

    params = rng.uniform(0, 1, size=(num_atoms, 4)).astype(precision)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)

    du_dx, du_dp, u = potential.to_gpu(precision).unbound_impl.execute(conf, params, box.astype(precision))

    assert (du_dx == 0).all()
    assert (du_dp == 0).all()
    assert u == 0


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_subset", [None, 33])
@pytest.mark.parametrize("num_atoms", [33, 65, 231])
def test_nonbonded_interaction_group_all_pairs_correctness(
    num_atoms,
    num_atoms_subset,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Verify that NonbondedInteractionGroup behaves correctly when used as an all pairs nonbonded potential"

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    atom_idxs = (
        rng.choice(num_atoms, size=(num_atoms_subset,), replace=False).astype(np.int32)
        if num_atoms_subset
        else np.arange(num_atoms, dtype=np.int32)
    )

    potential = NonbondedInteractionGroup(num_atoms, atom_idxs, beta, cutoff, col_atom_idxs=atom_idxs)

    test_impl = potential.to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms", [50, 231])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_col_atoms", [0, 1, 10, 33, None])
@pytest.mark.parametrize("num_systems", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("random_idx_lengths", [True, False])
def test_nonbonded_interaction_group_batch_correctness(
    random_idx_lengths,
    num_systems,
    num_col_atoms,
    num_atoms_ligand,
    num_atoms,
    precision,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng,
):
    "Verify that batch implementation of NonbondedInteractionGroup is identical to non-batched"

    coord_idxs = [rng.choice(example_conf.shape[0], num_atoms, replace=False) for _ in range(num_systems)]

    coords = np.stack([example_conf[idxs] for idxs in coord_idxs]).astype(precision)
    boxes = np.stack([example_box] * num_systems).astype(precision)
    params = np.stack([example_nonbonded_potential.params[idxs].astype(precision) for idxs in coord_idxs]).astype(
        precision
    )

    idxs_per_batch = np.array([num_atoms_ligand] * num_systems)
    if random_idx_lengths and num_atoms_ligand > 1:
        idxs_per_batch = rng.integers(1, num_atoms_ligand, size=num_systems)

    ligand_idxs = [rng.choice(num_atoms, size=(size,), replace=False).astype(np.int32) for size in idxs_per_batch]

    if num_col_atoms is None:  # means all the rest
        num_col_atoms = num_atoms - num_atoms_ligand

    col_atom_idxs = None
    if num_col_atoms:
        host_idxs = [np.setdiff1d(np.arange(num_atoms), lig_idxs).astype(np.int32) for lig_idxs in ligand_idxs]
        if random_idx_lengths and num_col_atoms > 1:
            col_atom_counts = rng.integers(1, min(num_col_atoms, num_atoms), size=num_systems)
        else:
            col_atom_counts = [num_col_atoms] * num_systems
        col_atom_idxs = [
            rng.choice(idxs, size=(size,), replace=False).astype(np.int32)
            for idxs, size in zip(host_idxs, col_atom_counts)
        ]

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff, col_atom_idxs=col_atom_idxs)

    for w_params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        batch_du_dx, batch_du_dp, batch_u = potential.to_gpu(precision).unbound_impl.execute_dim(
            coords, w_params, boxes, 1, 1, 1
        )
        for i in range(num_systems):
            ref_nb = NonbondedInteractionGroup(
                num_atoms,
                ligand_idxs[i],
                beta,
                cutoff,
                col_atom_idxs=col_atom_idxs[i] if col_atom_idxs is not None else None,
            )
            ref_du_dx, ref_du_dp, ref_u = ref_nb.to_gpu(precision).unbound_impl.execute(
                coords[i], w_params[i], boxes[i]
            )
            np.testing.assert_array_equal(batch_du_dx[i], ref_du_dx, err_msg=f"Batch {i}")
            np.testing.assert_array_equal(batch_du_dp[i], ref_du_dp, err_msg=f"Batch {i}")
            np.testing.assert_array_equal(batch_u[i], ref_u, err_msg=f"Batch {i}")


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms", [50, 231])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_col_atoms", [0, 1, 10, 33, None])
def test_nonbonded_interaction_group_correctness(
    num_col_atoms,
    num_atoms_ligand,
    num_atoms,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng,
):
    "Compares with jax reference implementation."

    conf = example_conf[:num_atoms].astype(precision).astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    if num_col_atoms is None:  # means all the rest
        num_col_atoms = num_atoms - num_atoms_ligand

    col_atom_idxs = None
    if num_col_atoms:
        host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs).astype(np.int32)
        col_atom_idxs = rng.choice(host_idxs, size=(num_col_atoms,), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff, col_atom_idxs=col_atom_idxs)

    test_impl = potential.to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms", [231])
@pytest.mark.parametrize("num_atoms_ligand", [128])
@pytest.mark.parametrize("num_col_atoms", [1, 10, 33])
def test_nonbonded_interaction_group_neighborlist_rebuild(
    num_col_atoms,
    num_atoms_ligand,
    num_atoms,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng,
):
    "Verify that randomizing the column indices will correctly trigger a neighborlist rebuild"

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    if num_col_atoms is None:  # means all the rest
        num_col_atoms = num_atoms - num_atoms_ligand

    col_atom_idxs = None
    if num_col_atoms:
        host_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs).astype(np.int32)
        col_atom_idxs = rng.choice(host_idxs, size=(num_col_atoms,), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff, col_atom_idxs=col_atom_idxs)

    test_impl = potential.to_gpu(precision)

    # Test that if we compare the potentials then randomize the column indices that the potentials still agree.
    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)

        # Randomize the coordinates of the column atoms to trigger a nblist rebuild
        conf[col_atom_idxs] += rng.random(size=(len(col_atom_idxs), 3)) * (cutoff**2)

        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231, 1050])
def test_nonbonded_interaction_group_consistency_allpairs_4d_decoupled(
    num_atoms,
    num_atoms_ligand,
    precision,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potential, which computes the sum of
    all pairwise interactions. This uses the identity

      U = U_A + U_B + U_AB

    where
    * U is the all-pairs potential over all atoms
    * U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    * U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B

    * U is computed using the reference potential over all atoms
    * U_A + U_B is computed using the reference potential over all atoms,
      separated into 2 non-interacting groups in the 4th dimension
    """

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    ref_all_pairs = NonbondedInteractionGroup(
        num_atoms,
        row_atom_idxs=np.arange(num_atoms, dtype=np.int32),
        col_atom_idxs=np.arange(num_atoms, dtype=np.int32),
        beta=beta,
        cutoff=cutoff,
    ).to_gpu(precision)

    raw_ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False)
    raw_other_idxs = list(range(num_atoms))

    for x in raw_ligand_idxs:
        raw_other_idxs.remove(x)

    ligand_idxs = np.array(raw_ligand_idxs, dtype=np.int32)
    other_idxs = np.array(raw_other_idxs, dtype=np.int32)

    ref_U_A = NonbondedInteractionGroup(
        num_atoms,
        row_atom_idxs=ligand_idxs,
        col_atom_idxs=ligand_idxs,
        beta=beta,
        cutoff=cutoff,
    )
    ref_U_B = NonbondedInteractionGroup(
        num_atoms,
        row_atom_idxs=other_idxs,
        col_atom_idxs=other_idxs,
        beta=beta,
        cutoff=cutoff,
    )
    test_ixn_group = FanoutSummedPotential([ref_U_A, ref_U_B]).to_gpu(precision)

    def ref_ixn_group(x, p, b):
        p = jnp.asarray(p).at[ligand_idxs, 3].set(3.01 * cutoff)
        return ref_all_pairs(x, p, b)

    GradientTest().compare_forces(
        conf,
        params,
        box,
        ref_potential=ref_ixn_group,
        test_potential=test_ixn_group,
        rtol=0,  # why doesn't 0 pass?
        atol=0,
    )


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 2e-4, 5e-4)])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_consistency_allpairs_constant_shift(
    num_atoms,
    num_atoms_ligand,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potential, which computes the sum of
    all pairwise interactions. This uses the identity

      U(x') - U(x) = U_AB(x') - U_AB(x)

    where
    * U is the all-pairs potential over all atoms
    * U_A, U_B are all-pairs potentials for interacting groups A and
      B, respectively
    * U_AB is the "interaction group" potential, i.e. the sum of
      pairwise interactions (a, b) where "a" is in A and "b" is in B
    * the transformation x -> x' does not affect U_A or U_B (e.g. a
      constant translation applied to each atom in one group)
    """

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    def ref_allpairs(conf):
        U_ref = Nonbonded(
            num_atoms,
            exclusion_idxs=np.array([], dtype=np.int32),
            scale_factors=np.zeros((0, 2), dtype=np.float64),
            beta=beta,
            cutoff=cutoff,
        )

        return U_ref(conf, params, box)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    test_impl = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff).to_gpu(precision).unbound_impl

    def test_ixngroups(conf):
        _, _, u = test_impl.execute(conf, params, box)
        return u

    conf_prime = np.array(conf)
    conf_prime[ligand_idxs] += rng.normal(0, 0.01, size=(3,))

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        ref_delta = ref_allpairs(conf_prime) - ref_allpairs(conf)
        test_delta = test_ixngroups(conf_prime) - test_ixngroups(conf)
        np.testing.assert_allclose(ref_delta, test_delta, rtol=rtol, atol=atol)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_set_atom_idxs(
    num_atoms, num_atoms_ligand, precision, cutoff, beta, rng: np.random.Generator
):
    box = 3.0 * np.eye(3)
    box = box.astype(precision)
    conf = rng.uniform(0, cutoff * 10, size=(num_atoms, 3)).astype(precision)
    params = rng.uniform(0, 1, size=(num_atoms, 4)).astype(precision)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)
    other_idxs = np.setdiff1d(np.arange(num_atoms), ligand_idxs)

    # Pick a subset to compare against, should produce different values
    secondary_ligand_set = rng.choice(other_idxs, size=(1), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)
    unbound_pot = potential.to_gpu(precision).unbound_impl

    ref_du_dx, ref_du_dp, ref_u = unbound_pot.execute(conf, params, box)

    # Set to first particle not in ligand_idxs, should produce different values
    col_atom_idxs = np.setdiff1d(np.arange(num_atoms, dtype=np.int32), secondary_ligand_set)
    unbound_pot.set_atom_idxs(secondary_ligand_set, col_atom_idxs)  # type: ignore
    assert np.all(unbound_pot.get_row_idxs() == secondary_ligand_set)
    assert np.all(unbound_pot.get_col_idxs() == col_atom_idxs)

    diff_du_dx, diff_du_dp, diff_u = unbound_pot.execute(conf, params, box)
    assert np.any(diff_du_dx != ref_du_dx)
    assert np.any(diff_du_dp != ref_du_dp)
    assert not np.allclose(ref_u, diff_u)

    # Reconstructing an Ixn group with the other set of atoms should be identical.
    potential2 = NonbondedInteractionGroup(num_atoms, secondary_ligand_set, beta, cutoff)
    unbound_pot2 = potential2.to_gpu(precision).unbound_impl

    diff_ref_du_dx, diff_ref_du_dp, diff_ref_u = unbound_pot2.execute(conf, params, box)
    np.testing.assert_array_equal(diff_ref_du_dx, diff_du_dx)
    np.testing.assert_array_equal(diff_ref_du_dp, diff_du_dp)
    np.testing.assert_equal(diff_ref_u, diff_u)

    # Set back to the indices, but shuffled, should be identical to reference
    rng.shuffle(ligand_idxs)
    col_atom_idxs = np.setdiff1d(np.arange(num_atoms, dtype=np.int32), ligand_idxs)
    unbound_pot.set_atom_idxs(ligand_idxs, col_atom_idxs)  # type: ignore

    test_du_dx, test_du_dp, test_u = unbound_pot.execute(conf, params, box)
    np.testing.assert_array_equal(test_du_dx, ref_du_dx)
    np.testing.assert_array_equal(test_du_dp, ref_du_dp)
    np.testing.assert_equal(test_u, ref_u)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_interaction_group_set_nblist_padding(
    num_atoms, num_atoms_ligand, precision, cutoff, beta, rng: np.random.Generator
):
    box = 3.0 * np.eye(3)
    box = box.astype(precision)
    conf = rng.uniform(0, cutoff * 10, size=(num_atoms, 3)).astype(precision)
    params = rng.uniform(0, 1, size=(num_atoms, 4)).astype(precision)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    potential = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)
    unbound_pot = potential.to_gpu(precision).unbound_impl

    def verify_padding(expected_padding: float):
        if precision == np.float64:
            assert expected_padding == unbound_pot.get_nblist_padding()  # type: ignore
        else:
            # Gets converted to a float under the hood, may be every so slightly different
            np.testing.assert_allclose(expected_padding, unbound_pot.get_nblist_padding())  # type: ignore

    verify_padding(potential.nblist_padding)

    ref_du_dx, ref_du_dp, ref_u = unbound_pot.execute(conf, params, box)

    # Running again should be identical
    test_du_dx, test_du_dp, test_u = unbound_pot.execute(conf, params, box)
    np.testing.assert_array_equal(test_du_dx, ref_du_dx)
    np.testing.assert_array_equal(test_du_dp, ref_du_dp)
    np.testing.assert_equal(test_u, ref_u)

    for new_padding in np.linspace(0.0, 1.2, 10):
        unbound_pot.set_nblist_padding(new_padding)  # type: ignore
        verify_padding(new_padding)

        test_du_dx, test_du_dp, test_u = unbound_pot.execute(conf, params, box)
        np.testing.assert_array_equal(test_du_dx, ref_du_dx)
        np.testing.assert_array_equal(test_du_dp, ref_du_dp)
        np.testing.assert_equal(test_u, ref_u)


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision", [np.float64, np.float32])
@pytest.mark.parametrize("num_atoms_ligand", [1, 15])
@pytest.mark.parametrize("num_atoms", [33, 231])
def test_nonbonded_ixn_group_order_independent(
    num_atoms,
    num_atoms_ligand,
    precision,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    "Verifies that with and without hilbert sorting the nonbonded potential is bitwise deterministic."

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    box = example_box.astype(precision)

    ligand_idxs = rng.choice(num_atoms, size=(num_atoms_ligand,), replace=False).astype(np.int32)

    sorted_pot = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff)
    unsorted_pot = NonbondedInteractionGroup(num_atoms, ligand_idxs, beta, cutoff, disable_hilbert_sort=True)

    sorted_impl = sorted_pot.to_gpu(precision).unbound_impl
    unsorted_impl = unsorted_pot.to_gpu(precision).unbound_impl

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        a_du_dx, a_du_dp, a_u = sorted_impl.execute(conf, params, box)
        b_du_dx, b_du_dp, b_u = unsorted_impl.execute(conf, params, box)
        np.testing.assert_array_equal(a_du_dx, b_du_dx)
        np.testing.assert_array_equal(a_du_dp, b_du_dp)
        assert a_u == b_u


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.2])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("num_free_idxs", [1, 7, 35, 63, 102])
@pytest.mark.parametrize("num_frozen_idxs", [1, 9, 15, 108, 0])  # 0 means "everything"
@pytest.mark.parametrize("num_atoms", [231, 1050])
def test_nonbonded_interaction_group_consistency_local_md(
    num_atoms,
    num_free_idxs,
    num_frozen_idxs,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    example_nonbonded_potential,
    example_conf,
    example_box,
    rng: np.random.Generator,
):
    """Compares with reference nonbonded potentials:

    Let A = free_idxs
    Let B = frozen_idxs

    U_AAB = U_AA + U_AB

    Where U_AA is an all-pairs potential
      and U_AB is an ixn-group potential
      and U_AAB is an upper triagonal ixn_group
    """

    if num_frozen_idxs == 0:
        num_frozen_idxs = num_atoms - num_free_idxs

    conf = example_conf[:num_atoms].astype(precision)
    params = example_nonbonded_potential.params[:num_atoms, :].astype(precision)
    params[:, 1] /= 2
    box = example_box.astype(precision)

    random_idxs = rng.choice(num_atoms, size=num_free_idxs + num_frozen_idxs, replace=False)
    free_idxs = random_idxs[:num_free_idxs].astype(np.int32)
    frozen_idxs = random_idxs[num_free_idxs : (num_free_idxs + num_frozen_idxs)].astype(np.int32)

    row_idxs = free_idxs
    col_idxs = np.concatenate([free_idxs, frozen_idxs])

    # test potential
    U_AAB = NonbondedInteractionGroup(num_atoms, row_idxs, beta, cutoff, col_idxs).to_gpu(precision)

    # build ref potential
    U_AA = Nonbonded(
        num_atoms,
        exclusion_idxs=np.array([], dtype=np.int32),
        scale_factors=np.zeros((0, 2), dtype=precision),
        beta=beta,
        cutoff=cutoff,
        atom_idxs=free_idxs,
    )
    U_AB = NonbondedInteractionGroup(num_atoms, free_idxs, beta, cutoff, frozen_idxs)
    U_sum = FanoutSummedPotential([U_AA, U_AB]).to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(
            conf,
            params,
            box,
            ref_potential=U_AAB,
            test_potential=U_sum,
            rtol=0,  # bitwise identical
            atol=0,  # bitwise identical
        )

    # compare just the ixn group itself to reference Python potential.
    U_AAB_jax = NonbondedInteractionGroup(num_atoms, row_idxs, beta, cutoff, col_idxs)
    U_AAB_gpu = U_AAB_jax.to_gpu(precision)

    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        GradientTest().compare_forces(
            conf,
            params,
            box,
            ref_potential=U_AAB_jax,
            test_potential=U_AAB_gpu,
            rtol=rtol,
            atol=atol,
        )

    # test that du_dx and du_dp of free atoms are bitwise identical, and that energies are identical
    ref_du_dx, ref_du_dp, ref_u = U_AAB_gpu.unbound_impl.execute(conf, params, box, True, True, True)
    U_AAB_gpu.unbound_impl.set_compute_col_grads(False)  # type: ignore
    test_du_dx, test_du_dp, test_u = U_AAB_gpu.unbound_impl.execute(conf, params, box, True, True, True)

    assert ref_u == test_u
    np.testing.assert_array_equal(ref_du_dx[free_idxs], test_du_dx[free_idxs])
    np.testing.assert_array_equal(ref_du_dp[free_idxs], test_du_dp[free_idxs])

    # for smaller system sizes we can occasionally still compute gradients on frozen_idxs spuriously
    # due to inclusion in a tile containing free_idxs, this test specifically checks for the typical
    # local MD use case.
    if num_frozen_idxs == num_atoms - num_free_idxs:
        assert not np.array_equal(ref_du_dx[frozen_idxs], test_du_dx[frozen_idxs])
        assert not np.array_equal(ref_du_dp[frozen_idxs], test_du_dp[frozen_idxs])
