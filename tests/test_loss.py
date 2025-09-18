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

from typing import Callable

import numpy as np
import pytest

from tmd.fe.loss import flat_bottom_loss, l1_loss, pseudo_huber_loss, truncated_residuals

pytestmark = [pytest.mark.nocuda]

np.random.seed(2021)


def _assert_nonnegative_loss(loss_fxn: Callable[[float], float]):
    residuals = np.linspace(-10, 10, 100)
    losses = np.array([loss_fxn(res) for res in residuals])
    np.testing.assert_array_equal(losses >= 0, True)


def _assert_zero_loss_when_perfect(loss_fxn: Callable[[float], float]):
    np.testing.assert_almost_equal(loss_fxn(0.0), 0)


def _assert_basic_loss_properties(loss_fxn: Callable[[float], float]):
    _assert_nonnegative_loss(loss_fxn)
    _assert_zero_loss_when_perfect(loss_fxn)


def test_l1_loss():
    _assert_basic_loss_properties(l1_loss)


def test_pseudo_huber_loss():
    _assert_basic_loss_properties(pseudo_huber_loss)


def test_flat_bottom_loss():
    _assert_basic_loss_properties(flat_bottom_loss)


def _scalar_truncated_residual(prediction, label, reliable_interval):
    lower, upper = reliable_interval
    assert lower < upper

    if (label >= lower) and (label <= upper):
        residual = prediction - label
    elif label < lower:
        residual = max(0, prediction - lower)
    elif label > upper:
        residual = min(0, prediction - upper)
    else:
        raise (RuntimeError("unsatisfiable reliable_range"))

    return residual


def assert_consistency_with_scalar_version(predictions, labels, reliable_interval):
    ref = np.array([_scalar_truncated_residual(p, l, reliable_interval) for (p, l) in zip(predictions, labels)])

    test = truncated_residuals(predictions, labels, reliable_interval)

    np.testing.assert_allclose(ref, test)


def test_compute_residuals_finite_interval():
    predictions = np.random.rand(10000) * 4 - 2
    labels = np.random.rand(10000) * 4 - 2

    reliable_interval = (-1, +1)

    assert_consistency_with_scalar_version(predictions, labels, reliable_interval)


def test_compute_residuals_infinite_interval():
    predictions = np.random.rand(10000) * 4 - 2
    labels = np.random.rand(10000) * 4 - 2

    reliable_interval = (-np.inf, +np.inf)

    assert_consistency_with_scalar_version(predictions, labels, reliable_interval)


def test_compute_residuals_sanity_check():
    reliable_interval = (-10, +np.inf)
    predictions = np.array([-100.0, 0.0])

    # assays bottom'd out
    labels = np.array([-15, -15])

    # if we predict -100, but our label of -15 bottomed out at -10, residual should be 0
    # if we predict    0, and our label of -15 bottomed out at -10, residual should be >=10
    expected_residual = np.array([0, 10])
    test_residual = truncated_residuals(predictions, labels, reliable_interval)
    np.testing.assert_array_equal(expected_residual, test_residual)  # passes
