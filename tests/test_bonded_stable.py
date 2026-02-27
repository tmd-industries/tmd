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

import jax
import numpy as np
import pytest
from common import GradientTest

from tmd.potentials import HarmonicAngle, bonded


def generate_system(n_particles, n_angles, seed, precision):
    np.random.seed(seed)
    coords = GradientTest().get_random_coords(n_particles, D=3)
    params = np.random.rand(n_angles, 3).astype(np.float64)

    angle_idxs = []
    for _ in range(n_angles):
        angle_idxs.append(np.random.choice(n_particles, size=3, replace=False))
    angle_idxs = np.array(angle_idxs, dtype=np.int32) if n_angles else np.zeros((0, 3), dtype=np.int32)

    return angle_idxs, coords.astype(precision), params.astype(precision)


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision,rtol", [(np.float32, 2e-5), (np.float64, 1e-9)])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle(n_particles, n_angles, precision, rtol, seed):
    """Validate HarmonicAngle reference on random triples of particles"""

    box = np.eye(3, dtype=precision)  # note: ignored
    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed, precision)
    potential = HarmonicAngle(n_particles, angle_idxs)
    test_impl = potential.to_gpu(precision)
    GradientTest().compare_forces(coords, params, box, potential, test_impl, rtol)
    GradientTest().assert_differentiable_interface_consistency(coords, params, box, test_impl)


@pytest.mark.parametrize("n_particles", [64])
@pytest.mark.parametrize("n_angles", [25])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [2022])
def test_harmonic_angle_bitwise_symmetric(n_particles, n_angles, precision, seed):
    "Test bitwise equality when angles are transformed like (i, j, k) -> (k, j, i)"

    angle_idxs, coords, params = generate_system(n_particles, n_angles, seed, precision)

    test_potential_impl = HarmonicAngle(n_particles, angle_idxs).to_gpu(precision).unbound_impl
    test_potential_rev_impl = HarmonicAngle(n_particles, angle_idxs[:, ::-1]).to_gpu(precision).unbound_impl

    box = np.eye(3, dtype=precision)  # note: ignored
    test_du_dx, test_du_dp, test_u = test_potential_impl.execute(coords, params, box, 1, 1, 1)
    test_du_dx_rev, test_du_dp_rev, test_u_rev = test_potential_rev_impl.execute(coords, params, box, 1, 1, 1)

    np.testing.assert_array_equal(test_u, test_u_rev)
    np.testing.assert_array_equal(test_du_dx, test_du_dx_rev)
    np.testing.assert_array_equal(test_du_dp, test_du_dp_rev)


@pytest.mark.parametrize(
    "potential,params",
    [
        pytest.param(HarmonicAngle, [(1, 1, 1)]),
        pytest.param(
            HarmonicAngle,
            [(1, 1, 0)],
            marks=pytest.mark.xfail(reason="expect singularity when eps=0", strict=True),
        ),
    ],
)
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_harmonic_angle_finite_force_with_vanishing_bond_length(potential, params, precision):
    "Check that forces do not blow up when a bond has length close to zero"

    angle_idxs = np.array([(0, 1, 2)], dtype=np.int32)
    coords = np.array([(0, 0, 0), (1e-9, 0, 0), (0, 1, 0)], dtype=precision)
    params = np.array(params, dtype=precision)
    impl = potential(len(coords), angle_idxs).to_gpu(precision).unbound_impl
    box = np.eye(3, dtype=precision) * 100.0
    du_dx, _, _ = impl.execute(coords, params, box, 1, 0, 0)
    assert (np.abs(du_dx) < 1e7).all()


@pytest.mark.nocuda
def test_harmonic_angle_jax():
    "Check that forces do not blow up when a bond has length close to zero"

    angle_idxs = np.array([(0, 1, 2)])
    coords = np.array([(0, 0, 0), (1e-9, 0, 0), (0, 1, 0)])
    params = np.array([(1, 1, 0.001)])

    grad_fn = jax.grad(bonded.harmonic_angle, argnums=(0,))
    g = grad_fn(coords, params, None, angle_idxs)
    assert (np.abs(g) < 1e7).all()

    # setting eps to 0 should lead to a blow-up
    params = np.array([(1, 1, 0)])

    grad_fn = jax.grad(bonded.harmonic_angle, argnums=(0,))
    g = grad_fn(coords, params, None, angle_idxs)
    assert (np.abs(g) > 1e7).any()
