import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets, shift_random_coordinates_by_box

from tmd.potentials import NonbondedPairList

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_pair_list_invalid_pair_idxs():
    with pytest.raises(RuntimeError, match=r"pair_idxs.size\(\) must be even, but got 1"):
        NonbondedPairList(4, [0], [0], 2.0, 1.1).to_gpu(np.float32).unbound_impl

    with pytest.raises(RuntimeError, match=r"illegal pair with src == dst: 0, 0"):
        NonbondedPairList(4, [(0, 0)], [(1, 1)], 2.0, 1.1).to_gpu(np.float32).unbound_impl

    with pytest.raises(RuntimeError, match=r"expected same number of pairs and scale tuples, but got 1 != 2"):
        NonbondedPairList(4, [(0, 1)], [(1, 1), (2, 2)], 2.0, 1.1).to_gpu(np.float32).unbound_impl


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ixn_group_size", [2, 33, 231])
def test_nonbonded_pair_list_correctness(
    ixn_group_size,
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
    "Compares with jax reference implementation."
    seed = rng.integers(np.iinfo(np.int32).max)

    num_atoms, _ = example_conf.shape

    # randomly select 2 interaction groups and construct all pairwise interactions
    atom_idxs = rng.choice(num_atoms, size=(2, ixn_group_size), replace=False).astype(np.int32)

    pair_idxs = np.stack(np.meshgrid(atom_idxs[0, :], atom_idxs[1, :])).reshape(2, -1).T
    num_pairs, _ = pair_idxs.shape

    rescale_mask = rng.uniform(0, 1, size=(num_pairs, 2)).astype(precision)

    potential = NonbondedPairList(num_atoms, pair_idxs, rescale_mask, beta, cutoff)
    params = example_nonbonded_potential.params

    for params_ in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff):
        test_impl = potential.to_gpu(precision)
        conf = example_conf.astype(precision)
        box = example_box.astype(precision)
        params_ = params.astype(precision)

        np.testing.assert_array_equal(potential.idxs, test_impl.unbound_impl.get_idxs())
        GradientTest().compare_forces(conf, params_, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params_, box, test_impl)

        # Testing batching across multiple coords/params
        num_batches = 3

        coords = np.array(
            [shift_random_coordinates_by_box(example_conf, example_box, seed=seed + i) for i in range(num_batches)],
            dtype=precision,
        )
        batch_pair_idxs = [rng.choice(pair_idxs, size=rng.integers(1, num_pairs), axis=0) for _ in range(num_batches)]
        batch_rescale_mask = [rng.uniform(0, 1, size=(len(idxs), 2)).astype(precision) for idxs in batch_pair_idxs]

        batch_params = [
            (np.array(params) + rng.uniform(0.0, 1e-3, size=params.shape)).astype(precision) for idxs in batch_pair_idxs
        ]
        batch_boxes = [
            (np.array(example_box) + np.eye(3) * rng.uniform(-0.5, 1.0)).astype(precision) for _ in range(num_batches)
        ]

        batch_pot = NonbondedPairList(num_atoms, batch_pair_idxs, batch_rescale_mask, beta, cutoff)

        batch_impl = batch_pot.to_gpu(precision).unbound_impl
        assert batch_impl.batch_size() == num_batches
        batch_du_dx, batch_du_dp, batch_u = batch_impl.execute_dim(coords, batch_params, batch_boxes, 1, 1, 1)

        assert batch_du_dx.shape[0] == num_batches
        assert batch_du_dx.shape[0] == len(batch_du_dp) == batch_u.size
        for i, (idxs, rescale_mask, x, box, params) in enumerate(
            zip(batch_pair_idxs, batch_rescale_mask, coords, batch_boxes, batch_params)
        ):
            potential = NonbondedPairList(num_atoms, idxs, rescale_mask, beta, cutoff)
            ref_du_dx, ref_du_dp, ref_u = potential.to_gpu(precision).unbound_impl.execute(x, params, box, 1, 1, 1)
            np.testing.assert_array_equal(batch_du_dx[i], ref_du_dx)
            np.testing.assert_array_equal(batch_du_dp[i], ref_du_dp)
            np.testing.assert_array_equal(batch_u[i], ref_u)
