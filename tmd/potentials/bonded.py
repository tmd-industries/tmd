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

import jax.numpy as jnp

from tmd.constants import DEFAULT_POSITIONAL_RESTRAINT_K
from tmd.potentials.jax_utils import delta_r
from tmd.potentials.types import Array


def centroid_restraint(conf, params, box, group_a_idxs, group_b_idxs, kb, b0):
    """Computes kb  * (r - b0)**2 where r is the distance between the centroids of group_a and group_b

    Notes
    -----
    * Geometric centroid, not mass-weighted centroid
    * Gradient undefined when `(r - b0) == 0` and `b0 != 0` (explicitly stabilized in case `b0 == 0`)
    * params, box unused
    * `kb` not `kb/2`
    """
    xi = conf[group_a_idxs]
    xj = conf[group_b_idxs]

    avg_xi = jnp.mean(xi, axis=0)
    avg_xj = jnp.mean(xj, axis=0)

    dx = avg_xi - avg_xj
    d2ij = jnp.sum(dx * dx)
    d2ij = jnp.where(d2ij == 0, 0, d2ij)  # stabilize derivative
    dij = jnp.sqrt(d2ij)
    delta = dij - b0

    # when b0 == 0 and dij == 0
    return jnp.where(b0 == 0, kb * d2ij, kb * delta**2)


def harmonic_bond(conf: Array, params: Array, box: Array, bond_idxs: Array):
    r"""
    Compute the harmonic bond energy given a collection of molecules.

    This implements a harmonic bond potential:
        V(conf) = 0.5 \sum_bond kbs[bond] * (distance[bond] - r0s[bond])^2

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.ndarray
        atomic coordinates

    params: shape [num_bonds, 2] np.ndarray
        force constants, eq lengths
        (kbs, r0s = params.T)

    box: shape [3, 3] np.ndarray
        periodic boundary vectors

    bond_idxs: [num_bonds, 2] np.ndarray
        each element (src, dst) is a unique bond in the conformation

    Notes
    -----
    """
    assert params.shape == bond_idxs.shape

    if bond_idxs.shape[0] == 0:
        return 0.0

    ci = conf[bond_idxs[:, 0]]
    cj = conf[bond_idxs[:, 1]]

    cij = delta_r(ci, cj, box)
    d2ij = jnp.sum(cij * cij, axis=-1)
    d2ij = jnp.where(d2ij == 0, 0, d2ij)  # stabilize derivative
    dij = jnp.sqrt(d2ij)
    kbs = params[:, 0]
    r0s = params[:, 1]

    # this is here to prevent a numerical instability
    # when b0 == 0 and dij == 0
    energy = jnp.where(r0s == 0, kbs / 2 * d2ij, kbs / 2 * jnp.power(dij - r0s, 2.0))

    return jnp.sum(energy)


def kahan_angle(ci, cj, ck, eps, box):
    """
    Compute the angle given three points, i,j,k, as defined by the vector j->i, j->k

    Note: eps is a small value used to stabilize computation when either r_ji or r_jk goes to zero.
    See: https://github.com/proteneer/timemachine/pull/935
    """
    rji = jnp.hstack([delta_r(ci, cj, box), jnp.expand_dims(eps, axis=-1)])
    rjk = jnp.hstack([delta_r(ck, cj, box), jnp.expand_dims(eps, axis=-1)])
    nji = jnp.linalg.norm(rji, axis=-1)
    njk = jnp.linalg.norm(rjk, axis=-1)
    nji = jnp.expand_dims(nji, axis=-1)
    njk = jnp.expand_dims(njk, axis=-1)
    y = jnp.linalg.norm(njk * rji - nji * rjk, axis=-1)
    x = jnp.linalg.norm(njk * rji + nji * rjk, axis=-1)
    angle = 2 * jnp.arctan2(y, x)
    return angle


def harmonic_angle(conf, params, box, angle_idxs):
    r"""
    Compute the harmonic angle energy using a numerically stable approximation.

    The functional form is identical to :py:func:`potentials.bonded.HarmonicAngle`, except that the following
    approximation is used for the intermediate computation of :math:`\cos(\theta)`:

    :math::

        \cos(\theta) \approx \frac{r_{ij} \cdot r_{kj}}{\sqrt{(r_{ij}^2 + \epsilon^2) (r_{kj}^2 + \epsilon^2)}}

    This reduces to the exact expression when :math:`\epsilon = 0`; When :math:`\epsilon > 0`, this avoids the
    singularities in the exact expression as :math:`r_{ij}` or :math:`r_{kj}` approach zero.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.ndarray
        atomic coordinates

    params: shape [num_angles, 3] np.ndarray
        force constants, eq angles, epsilons
        (kas, a0s, epsilons = params.T)

    box: shape [3, 3] np.ndarray
        periodic boundary vectors, if not None

    angle_idxs: shape [num_angles, 3] np.ndarray
        each element (i, j, k) is a unique angle in the conformation. Atom j is defined
        to be the middle atom.

    """
    if angle_idxs.shape[0] == 0:
        return 0.0
    ci, cj, ck = conf[angle_idxs.T]
    kas, a0s, eps = params.T
    angle = kahan_angle(ci, cj, ck, eps, box)
    energies = kas / 2 * jnp.power(angle - a0s, 2)
    return jnp.sum(energies, -1)


def signed_torsion_angle(ci, cj, ck, cl, box):
    """
    Batch compute the signed angle of a torsion angle.  The torsion angle
    between two planes should be periodic but not necessarily symmetric.

    Parameters
    ----------
    ci, cj, ck, cl: shape [num_torsions, 3] np.ndarrays
        atom coordinates defining torsion angle i-j-k-l

    Returns
    -------
    shape [num_torsions,] np.ndarray
        array of torsion angles
    """

    # Taken from the wikipedia arctan2 implementation:
    # https://en.wikipedia.org/wiki/Dihedral_angle

    # We use an identical but numerically stable arctan2
    # implementation as opposed to the OpenMM energy function to
    # avoid a singularity when the angle is zero.

    rij = delta_r(cj, ci, box)
    rkj = delta_r(cj, ck, box)
    rkl = delta_r(cl, ck, box)

    n1 = jnp.cross(rij, rkj)
    n2 = jnp.cross(rkj, rkl)

    y = jnp.sum(jnp.multiply(jnp.cross(n1, n2), rkj / jnp.linalg.norm(rkj, axis=-1, keepdims=True)), axis=-1)
    x = jnp.sum(jnp.multiply(n1, n2), -1)

    return jnp.arctan2(y, x)


def periodic_torsion(conf, params, box, torsion_idxs):
    """
    Compute the periodic torsional energy.

    Parameters:
    -----------
    conf: shape [num_atoms, 3] np.ndarray
        atomic coordinates

    params: shape [num_torsions, 3] np.ndarray
        parameters

    box: shape [3, 3] np.ndarray
        periodic boundary vectors, if not None

    torsion_idxs: shape [num_torsions, 4] np.ndarray
        indices denoting the four atoms that define a torsion

    Notes
    -----
    * box argument unused
    * if conf has more than 3 dimensions, this function only depends on the first 3
    """
    if torsion_idxs.shape[0] == 0:
        return 0.0

    conf = conf[:, :3]  # this is defined only in 3d

    ci = conf[torsion_idxs[:, 0]]
    cj = conf[torsion_idxs[:, 1]]
    ck = conf[torsion_idxs[:, 2]]
    cl = conf[torsion_idxs[:, 3]]

    ks = params[:, 0]
    phase = params[:, 1]
    period = params[:, 2]
    angle = signed_torsion_angle(ci, cj, ck, cl, box)

    nrg = ks * (1 + jnp.cos(period * angle - phase))
    return jnp.sum(nrg, axis=-1)


def _flat_bottom_bond_impl(r, params, box):
    """
    U(r; k, r_min, r_max) =
        (k/4) * (r - r_max)**4 if r > r_max
        (k/4) * (r - r_min)**4 if r < r_min
    """
    # compute distances

    # compute energies
    k, r_min, r_max = params.T
    bond_nrgs = (k / 4) * ((r > r_max) * ((r - r_max) ** 4) + (r < r_min) * ((r - r_min) ** 4))
    return bond_nrgs


def flat_bottom_bond(conf, params, box, bond_idxs):
    """
    U(r; k, r_min, r_max) =
        (k/4) * (r - r_max)**4 if r > r_max
        (k/4) * (r - r_min)**4 if r < r_min
    """
    i, j = bond_idxs.T
    r = jnp.sqrt(jnp.sum(delta_r(conf[i], conf[j], box) ** 2, 1))
    bond_nrgs = _flat_bottom_bond_impl(r, params, box)
    return jnp.sum(bond_nrgs)


def flat_bottom_restraint(conf, params, box, atom_idxs, restraint_coords):
    """
    Identical to the flat_bottom_bond except that atoms are restrained to specific coordinates

    U(r; k, r_min, r_max) =
        (k/4) * (r - r_max)**4 if r > r_max
        (k/4) * (r - r_min)**4 if r < r_min
    """
    r = jnp.sqrt(jnp.sum(delta_r(conf[atom_idxs], restraint_coords, box) ** 2, 1))
    restraint_nrgs = _flat_bottom_bond_impl(r, params, box)
    return jnp.sum(restraint_nrgs)


def log_flat_bottom_bond(conf, params, box, bond_idxs, beta):
    """
    Implements the log inverse of the flat bottom potential
    """
    i, j = bond_idxs.T
    r = jnp.sqrt(jnp.sum(delta_r(conf[i], conf[j], box) ** 2, 1))
    nrgs = _flat_bottom_bond_impl(r, params, box)
    log_nrgs = -jnp.log(1 - jnp.exp(-beta * nrgs))
    # note the extra 1/beta is to be consistent with other potentials
    # so that energies have units of kJ/mol
    return jnp.sum(log_nrgs) / beta


def harmonic_positional_restraint(
    x_init: Array, x_new: Array, box: Array, k: float = DEFAULT_POSITIONAL_RESTRAINT_K
) -> Array:
    r"""Harmonic positional restraint useful for performing minimization to prevent initial conformations
    from changing too much.

    This implements a harmonic bond potential, while being PBC aware:
        V(x_new, x_init, k) = \sum k / 2 * sum((x_new - x_init)^2)
    """
    assert x_init.shape == x_new.shape

    d2ij = jnp.sum(delta_r(x_new, x_init, box=box) ** 2, axis=-1)
    d2ij = jnp.where(d2ij == 0, 0, d2ij)  # stabilize derivative
    return jnp.sum((k / 2) * d2ij)
