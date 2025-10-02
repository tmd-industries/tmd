import numpy as np
import pytest
from common import GradientTest, gen_nonbonded_params_with_4d_offsets, shift_random_coordinates_by_box

from tmd.potentials import NonbondedPairListPrecomputed

pytestmark = [pytest.mark.memcheck]


def test_nonbonded_precomputed_pair_list_invalid_pair_idxs():
    with pytest.raises(RuntimeError, match=r"idxs.size\(\) must be exactly 2\*B"):
        NonbondedPairListPrecomputed(1, [0], 2.0, 1.1).to_gpu(np.float32).unbound_impl

    with pytest.raises(RuntimeError, match="illegal pair with src == dst: 0, 0"):
        NonbondedPairListPrecomputed(1, [(0, 0)], 2.0, 1.1).to_gpu(np.float32).unbound_impl


@pytest.mark.parametrize("beta", [2.0])
@pytest.mark.parametrize("cutoff", [1.1, 10000.0])
@pytest.mark.parametrize("precision,rtol,atol", [(np.float64, 1e-8, 1e-8), (np.float32, 1e-4, 5e-4)])
@pytest.mark.parametrize("ixn_group_size", [4, 33, 231])
@pytest.mark.parametrize("num_atoms", [25358])
def test_nonbonded_pair_list_precomputed_correctness(
    ixn_group_size,
    precision,
    rtol,
    atol,
    cutoff,
    beta,
    num_atoms,
    rng: np.random.Generator,
):
    "Compares with jax reference implementation."

    _pair_idxs = []
    for _ in range(ixn_group_size):
        _pair_idxs.append(rng.choice(np.arange(num_atoms), 2, replace=False))
    pair_idxs = np.array(_pair_idxs, dtype=np.int32)
    num_pairs, _ = pair_idxs.shape

    params = rng.uniform(0, 1, size=(num_pairs, 4))
    params[:, 0] -= 0.5  # get some positive and negative charges
    params[:, 1] /= 5  # shrink lj sigma to avoid huge repulsive forces

    conf = rng.uniform(0, 1, size=(num_atoms, 3)) * 3

    box = np.diag(
        1 + rng.uniform(0, 1, size=3) * 3
    )  # box should be fully ignored tbh (just like all other bonded forces)

    potential = NonbondedPairListPrecomputed(num_atoms, pair_idxs, beta, cutoff)

    # delta_w positive by convention
    test_impl = potential.to_gpu(precision)
    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff, w_min=0.0):
        conf = conf.astype(precision)
        params = params.astype(precision)
        box = box.astype(precision)
        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)

    # test bare charges
    params[:, 1] = 0
    params[:, 2] = 0
    for params in gen_nonbonded_params_with_4d_offsets(rng, params, cutoff, w_min=0.0):
        conf = conf.astype(precision)
        params = params.astype(precision)
        box = box.astype(precision)
        GradientTest().compare_forces(conf, params, box, potential, test_impl, rtol=rtol, atol=atol)
        GradientTest().assert_differentiable_interface_consistency(conf, params, box, test_impl)

    seed = rng.integers(np.iinfo(np.int32).max)
    num_batches = 3

    coords = np.array(
        [
            shift_random_coordinates_by_box(rng.uniform(0, 1, size=(num_atoms, 3)) * 3, box, seed=seed + i)
            for i in range(num_batches)
        ],
        dtype=precision,
    )
    batch_pair_idxs = [rng.choice(pair_idxs, size=rng.integers(1, num_pairs), axis=0) for _ in range(num_batches)]

    batch_params = [rng.choice(params, size=len(idxs), replace=True).astype(precision) for idxs in batch_pair_idxs]
    batch_boxes = [(np.array(box) + np.eye(3) * rng.uniform(-0.5, 1.0)).astype(precision) for _ in range(num_batches)]

    batch_pot = NonbondedPairListPrecomputed(num_atoms, batch_pair_idxs, beta, cutoff)

    batch_impl = batch_pot.to_gpu(precision).unbound_impl
    assert batch_impl.batch_size() == num_batches
    batch_du_dx, batch_du_dp, batch_u = batch_impl.execute_dim(coords, batch_params, batch_boxes, 1, 1, 1)

    assert batch_du_dx.shape[0] == num_batches
    assert batch_du_dx.shape[0] == len(batch_du_dp) == batch_u.size
    for i, (idxs, x, box, params) in enumerate(zip(batch_pair_idxs, coords, batch_boxes, batch_params)):
        potential = NonbondedPairListPrecomputed(num_atoms, idxs, beta, cutoff)
        ref_du_dx, ref_du_dp, ref_u = potential.to_gpu(precision).unbound_impl.execute(x, params, box, 1, 1, 1)
        np.testing.assert_array_equal(batch_du_dx[i], ref_du_dx)
        np.testing.assert_array_equal(batch_du_dp[i], ref_du_dp)
        np.testing.assert_array_equal(batch_u[i], ref_u)
