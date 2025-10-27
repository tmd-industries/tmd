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

from typing import Optional

import numpy as np
import pytest
from common import hilbert_sort
from numpy.typing import NDArray

from tmd.constants import DEFAULT_TEMP
from tmd.fe.free_energy import HostConfig, MDParams, sample
from tmd.fe.rbfe import setup_initial_states, setup_optimized_host
from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import get_romol_conf
from tmd.ff import Forcefield
from tmd.lib import custom_ops
from tmd.md.builders import build_protein_system, build_water_system
from tmd.potentials.jax_utils import delta_r
from tmd.testsystems.dhfr import setup_dhfr
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file


@pytest.mark.memcheck
def test_empty_neighborlist():
    with pytest.raises(RuntimeError, match="Neighborlist N must be at least 1"):
        custom_ops.Neighborlist_f32(0, True)


def reference_block_bounds(coords: NDArray, box: NDArray, block_size: int) -> tuple[NDArray, NDArray]:
    # Make a copy to avoid modify the coordinates that end up used later by the Neighborlist
    coords = coords.copy()
    N = coords.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    box_diag = np.diagonal(box)

    _ref_ctrs = []
    _ref_exts = []

    for bidx in range(num_blocks):
        start_idx = bidx * block_size
        end_idx = min((bidx + 1) * block_size, N)
        block_coords = coords[start_idx:end_idx]
        min_coords = block_coords[0]
        max_coords = block_coords[0]
        for new_coords in block_coords[1:]:
            center = 0.5 * (max_coords + min_coords)
            new_coords -= box_diag * np.floor((new_coords - center) / box_diag + 0.5)
            min_coords = np.minimum(min_coords, new_coords)
            max_coords = np.maximum(max_coords, new_coords)

        _ref_ctrs.append((max_coords + min_coords) / 2)
        _ref_exts.append((max_coords - min_coords) / 2)

    ref_ctrs = np.array(_ref_ctrs)
    ref_exts = np.array(_ref_exts)
    return ref_ctrs, ref_exts


@pytest.mark.memcheck
@pytest.mark.parametrize("batches", [1, 2])
@pytest.mark.parametrize("precision,atol,rtol", [(np.float32, 1e-6, 1e-6), (np.float64, 2e-7, 1e-7)])
@pytest.mark.parametrize("sort", [True, False])
def test_block_bounds_dhfr(batches, precision, atol, rtol, sort):
    _, _, coords, box = setup_dhfr()

    print("STARt\n")
    if precision == np.float32:
        nblist = custom_ops.Neighborlist_f32(batches, coords.shape[0], True)
    else:
        nblist = custom_ops.Neighborlist_f64(batches, coords.shape[0], True)

    if sort:
        perm = hilbert_sort(coords, box)
        coords = coords[perm]

    print("Here we go")
    block_size = 32
    ref_ctrs, ref_exts = reference_block_bounds(coords, box, block_size)
    print(len(ref_ctrs))

    test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

    for i, (ref_ctr, test_ctr) in enumerate(zip(ref_ctrs, test_ctrs)):
        np.testing.assert_allclose(ref_ctr, test_ctr, atol=atol, rtol=rtol, err_msg=f"Center {i} has mismatch")
    for i, (ref_ext, test_ext) in enumerate(zip(ref_exts, test_exts)):
        np.testing.assert_allclose(ref_ext, test_ext, atol=atol, rtol=rtol, err_msg=f"Extent {i} has mismatch")


@pytest.mark.memcheck
@pytest.mark.parametrize("precision,atol,rtol", [(np.float32, 1e-6, 1e-6), (np.float64, 1e-7, 1e-7)])
@pytest.mark.parametrize("size", [12, 128, 156, 298])
def test_block_bounds(precision, atol, rtol, size):
    np.random.seed(2020)
    block_size = 32
    D = 3

    if precision == np.float32:
        nblist = custom_ops.Neighborlist_f32(size, True)
    else:
        nblist = custom_ops.Neighborlist_f64(size, True)

    coords = np.random.randn(size, D)

    box_diag = np.random.rand(3) + 1
    box = np.eye(3) * box_diag

    ref_ctrs, ref_exts = reference_block_bounds(coords, box, block_size)

    test_ctrs, test_exts = nblist.compute_block_bounds(coords, box, block_size)

    np.testing.assert_allclose(ref_ctrs, test_ctrs, atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref_exts, test_exts, atol=atol, rtol=rtol)


def get_water_coords(D):
    x = np.load("tests/data/water.npy").astype(np.float32).astype(np.float64)
    x = x[:, :D]

    return x


def build_reference_ixn_list(coords: NDArray, box: NDArray, cutoff: float, padding: float) -> list[list[int]]:
    row_idxs = np.arange(len(coords))
    col_idxs = np.arange(len(coords))
    upper_triangular = True
    return build_reference_ixn_group_list(coords, box, cutoff, padding, row_idxs, col_idxs, upper_triangular)


def build_reference_ixn_group_list(
    coords: NDArray,
    box: NDArray,
    cutoff: float,
    padding: float,
    row_idxs: NDArray,
    col_idxs: NDArray,
    upper_triangular: bool,
) -> list[list[int]]:
    N = coords.shape[0]
    block_size = 32
    col_coords = coords[col_idxs]
    col_coords = np.expand_dims(col_coords, axis=0)
    # Compute the reference interactions of the ligand
    ref_ixn_list: list[list[int]] = []
    all_row_coords = coords[row_idxs]
    row_length = all_row_coords.shape[0]
    num_blocks = (row_length + block_size - 1) // block_size

    padded_cutoff = cutoff + padding
    half_padded_cutoff = cutoff + padding / 2

    NR = len(row_idxs)

    for rbidx in range(num_blocks):
        row_start = rbidx * block_size
        row_end = min((rbidx + 1) * block_size, N)
        row_coords = all_row_coords[row_start:row_end]
        row_coords = np.expand_dims(row_coords, axis=1)
        deltas = delta_r(row_coords, col_coords, box)
        dij = np.linalg.norm(deltas, axis=-1)
        if upper_triangular:
            dij[:, :row_start] = padded_cutoff  # hack to mimic upper_triangular
            j_cutoff = np.where(col_idxs >= NR, half_padded_cutoff, padded_cutoff)
            j_cutoff = np.expand_dims(j_cutoff, axis=0)  # vector
        else:
            # scalar type (but broadcastable nevertheless)
            j_cutoff = padded_cutoff  # type: ignore

        arg_idxs = np.argwhere(np.any(dij < j_cutoff, axis=0))
        # Get back the column indices that are ixns
        idxs = col_idxs[arg_idxs.reshape(-1)]
        ref_ixn_list.append(idxs.reshape(-1).tolist())  # type: ignore
    return ref_ixn_list


def build_reference_ixn_list_with_subset(
    coords: NDArray, box: NDArray, cutoff: float, padding: float, row_idxs: NDArray
) -> list[list[int]]:
    identity_idxs = np.arange(len(coords))
    col_idxs = np.delete(identity_idxs, row_idxs)
    return build_reference_ixn_group_list(coords, box, cutoff, padding, row_idxs, col_idxs, upper_triangular=False)


def assert_ixn_lists_are_equal(ref_ixn, test_ixn):
    for bidx, (a, b) in enumerate(zip(ref_ixn, test_ixn)):
        if sorted(a) != sorted(b):
            print("TESTING bidx", bidx)
            print("ref_ixn", sorted(a))
            print("test_ixn", sorted(b))
        np.testing.assert_equal(sorted(a), sorted(b))


@pytest.mark.memcheck
@pytest.mark.parametrize("num_atoms", [35, 64, 129, 1025, 1259, 2029])
def test_nblist_row_indices_are_order_independent(num_atoms):
    D = 3
    cutoff = 0.9
    padding = 0.1
    water_coords = get_water_coords(D)
    nblists = [custom_ops.Neighborlist_f32(num_atoms, False), custom_ops.Neighborlist_f64(num_atoms, False)]

    np.random.seed(1234)
    water_idxs = np.random.choice(np.arange(water_coords.shape[0]), num_atoms, replace=False)
    coords = water_coords[water_idxs]
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.diag(diag)

    atom_idxs = np.random.choice(np.arange(coords.shape[0]), num_atoms // 2, replace=False)
    atom_idxs = atom_idxs.astype(np.uint32)

    reference_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, padding, atom_idxs)
    # Shuffle idxs, should still have the same set of interactions
    shuffled_idxs = atom_idxs.copy()
    np.random.shuffle(shuffled_idxs)

    assert not np.all(shuffled_idxs == atom_idxs)

    shuffled_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, padding, shuffled_idxs)

    # Verify that the ixns are the same, different ordering so each block will be different
    reference_ixns_set = set(np.concatenate(reference_ixns).reshape(-1))
    shuffled_ixns_set = set(np.concatenate(shuffled_ixns).reshape(-1))

    np.testing.assert_array_equal(reference_ixns_set, shuffled_ixns_set)

    # Verify that the C++ agrees
    for nblist in nblists:
        nblist.set_row_idxs(atom_idxs)
        test_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)
        test_ixns_set = set(np.concatenate(test_ixn_list).reshape(-1))
        assert reference_ixns_set == test_ixns_set
        assert_ixn_lists_are_equal(reference_ixns, test_ixn_list)

        nblist.set_row_idxs(shuffled_idxs)
        test_shuffle_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)
        test_shuffle_ixns_set = set(np.concatenate(test_shuffle_ixn_list).reshape(-1))
        assert shuffled_ixns_set == test_shuffle_ixns_set
        assert_ixn_lists_are_equal(shuffled_ixns, test_shuffle_ixn_list)


@pytest.mark.memcheck
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("upper_triangular", [False, True])
@pytest.mark.parametrize("num_atoms", [35, 64, 129, 1025, 1259, 2029])
def test_nblist_with_arbitrary_row_idxs_and_col_idxs(num_atoms, upper_triangular, precision):
    D = 3
    cutoff = 0.9
    padding = 0.1
    water_coords = get_water_coords(D)

    if precision == np.float32:
        nblist = custom_ops.Neighborlist_f32(num_atoms, upper_triangular)
    elif precision == np.float64:
        nblist = custom_ops.Neighborlist_f64(num_atoms, upper_triangular)
    else:
        assert 0, "Unknown precision"

    np.random.seed(1234)
    water_idxs = np.random.choice(np.arange(water_coords.shape[0]), num_atoms, replace=False)
    coords = water_coords[water_idxs]
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.diag(diag)

    # test partial rows
    row_atom_idxs = np.random.choice(np.arange(coords.shape[0]), num_atoms // 3, replace=False)
    row_atom_idxs = row_atom_idxs.astype(np.uint32)

    col_atom_idxs = np.random.choice(np.arange(coords.shape[0]), num_atoms // 2, replace=False)
    col_atom_idxs = col_atom_idxs.astype(np.uint32)

    reference_ixns_list = build_reference_ixn_group_list(
        coords, box, cutoff, padding, row_atom_idxs, col_atom_idxs, upper_triangular
    )

    nblist.set_row_idxs_and_col_idxs(row_atom_idxs, col_atom_idxs)
    test_ixns_list = nblist.get_nblist(coords, box, cutoff, padding)

    assert_ixn_lists_are_equal(reference_ixns_list, test_ixns_list)

    # test complete rows
    col_atom_idxs = np.random.choice(np.arange(coords.shape[0]), num_atoms, replace=False)
    col_atom_idxs = col_atom_idxs.astype(np.uint32)
    row_atom_idxs = col_atom_idxs[: num_atoms // 4]
    reference_ixns_list = build_reference_ixn_group_list(
        coords, box, cutoff, padding, row_atom_idxs, col_atom_idxs, upper_triangular
    )

    nblist.set_row_idxs_and_col_idxs(row_atom_idxs, col_atom_idxs)
    test_ixns_list = nblist.get_nblist(coords, box, cutoff, padding)

    assert_ixn_lists_are_equal(reference_ixns_list, test_ixns_list)


@pytest.mark.memcheck
@pytest.mark.parametrize("sort", [True])
@pytest.mark.parametrize("num_atoms", [35, 64, 129, 1025, 1259, 2029])
def test_neighborlist(num_atoms, sort):
    water_coords = get_water_coords(3)
    nblists = [custom_ops.Neighborlist_f32(num_atoms, True), custom_ops.Neighborlist_f64(num_atoms, True)]

    np.random.seed(1234)
    atom_idxs = np.random.choice(np.arange(num_atoms), num_atoms, replace=False)
    coords = water_coords[atom_idxs]
    padding = 0.1
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.eye(3) * diag

    cutoff = 0.9
    padding = 0.1

    if sort:
        perm = hilbert_sort(coords, box)
        coords = coords[perm]

    ref_ixn_list = build_reference_ixn_list(coords, box, cutoff, padding)
    for nblist in nblists:
        # Run twice to ensure deterministic results
        for _ in range(2):
            test_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)

            assert len(ref_ixn_list) == len(test_ixn_list)

            assert_ixn_lists_are_equal(ref_ixn_list, test_ixn_list)


@pytest.mark.memcheck
def test_neighborlist_resize():
    N = 3

    # Verify that the sizes of the rows and columns match how the NBlist was constructed
    for nblist in (
        custom_ops.Neighborlist_f32(N, True),
        custom_ops.Neighborlist_f64(N, True),
    ):
        with pytest.raises(RuntimeError, match="size is must be at least 1"):
            nblist.resize(0)

        with pytest.raises(RuntimeError, match=f"size is greater than max size: {N + 1} > {N}"):
            nblist.resize(N + 1)

        nblist.resize(N - 1)


@pytest.mark.memcheck
def test_neighborlist_invalid_row_idxs():
    N = 3

    # Verify that the sizes of the rows and columns match how the NBlist was constructed
    for nblist in (
        custom_ops.Neighborlist_f32(N, True),
        custom_ops.Neighborlist_f64(N, True),
    ):
        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.zeros(0, dtype=np.uint32))
        assert "idxs can't be empty" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.zeros(2, dtype=np.uint32))
        assert "atom indices must be unique" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.arange(N * 5, dtype=np.uint32))
        assert "number of idxs must be less than N" == str(e.value)

        with pytest.raises(RuntimeError) as e:
            nblist.set_row_idxs(np.arange(N - 1, dtype=np.uint32) * N * 5)
        assert "indices values must be less than N" == str(e.value)


@pytest.mark.memcheck
def test_neighborlist_on_subset_of_system():
    ligand, _, _ = get_hif2a_ligand_pair_single_topology()
    ligand_coords = get_romol_conf(ligand)
    ff = Forcefield.load_default()

    host_coords = build_water_system(4.0, ff.water_ff, mols=[ligand]).conf
    num_host_atoms = host_coords.shape[0]
    host_coords = np.array(host_coords)

    coords = np.concatenate([host_coords, ligand_coords])
    N = coords.shape[0]

    cutoff = 0.9
    padding = 0.1

    np.random.seed(1234)
    diag = np.amax(coords, axis=0) - np.amin(coords, axis=0) + padding
    box = np.eye(3) * diag

    atom_idxs = np.arange(num_host_atoms, N, dtype=np.uint32)
    sort = True
    if sort:
        perm = hilbert_sort(coords, box)
        coords = coords[perm]
        # Get the new idxs of the ligand atoms
        atom_idxs = np.isin(perm, atom_idxs).nonzero()[0]
    atom_idxs = atom_idxs.astype(np.uint32)

    reference_subset_ixns = build_reference_ixn_list_with_subset(coords, box, cutoff, padding, atom_idxs)
    reference_complete_ixns = build_reference_ixn_list(coords, box, cutoff, padding)

    for nblist in (
        custom_ops.Neighborlist_f32(N, False),
        custom_ops.Neighborlist_f64(N, False),
    ):
        nblist.set_row_idxs(atom_idxs)
        for _ in range(2):
            test_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)
            # compute the sparsity of the tile
            assert len(reference_subset_ixns) == len(test_ixn_list), "Number of blocks with interactions don't agree"

            assert_ixn_lists_are_equal(reference_subset_ixns, test_ixn_list)
        # Verify that you can reset the indices and go back to the regular neighborlist
        nblist.reset_row_idxs()
        nblist.set_compute_upper_triangular(True)

        test_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)
        # compute the sparsity of the tile
        assert len(reference_complete_ixns) == len(test_ixn_list), "Number of blocks with interactions don't agree"

        assert_ixn_lists_are_equal(reference_complete_ixns, test_ixn_list)


@pytest.mark.memcheck
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("tiles", [2, 10, 100])
def test_nblist_max_interactions(block_size, tiles):
    """Verify that if all particles in the system interact, that the neighborlist correctly assigns large enough buffers"""
    rng = np.random.default_rng(2023)
    cutoff = 9.9  # Set a large cutoff, so everything overlaps
    padding = 0.1

    coords = rng.random(size=(block_size * tiles, 3))
    box = np.eye(3) * 100.0

    nblist = custom_ops.Neighborlist_f32(coords.shape[0], True)
    max_ixn_count = nblist.get_max_ixn_count()
    test_ixn_list = nblist.get_nblist(coords, box, cutoff, padding)
    assert np.mean(compute_tile_densities(nblist, coords, box, cutoff, padding)) == 1.0
    assert len(test_ixn_list) == tiles
    for tile_ixns in test_ixn_list:
        assert len(tile_ixns) > 0
    assert nblist.get_tile_ixn_count() * block_size == max_ixn_count


@pytest.fixture(
    scope="module",
    params=["solvent", "complex"],
)
def hif2a_single_topology_leg(request):
    return setup_hif2a_initial_state(request.param)


def setup_hif2a_initial_state(host_name: str):
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config: Optional[HostConfig] = None

    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    if host_name == "complex":
        with path_to_internal_file("tmd.testsystems.data", "hif2a_nowater_min.pdb") as protein_path:
            host_config = build_protein_system(
                str(protein_path), forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b], box_margin=0.1
            )
    elif host_name == "solvent":
        host_config = build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b], box_margin=0.1)
    else:
        assert 0, "Invalid host name"

    st = SingleTopology(mol_a, mol_b, core, forcefield)
    host = setup_optimized_host(host_config, [mol_a, mol_b], forcefield)
    lambda_grid = np.array([0.0])
    initial_state = setup_initial_states(
        st, host, DEFAULT_TEMP, lambda_grid, seed=2024, min_cutoff=None, verify_constraints=True
    )[0]

    return st, host, host_name, initial_state


def compute_tile_densities(
    nblist,
    frame: NDArray,
    box: NDArray,
    cutoff: float,
    padding: float = 0.0,
    column_block: int = 32,
    row_block: int = 32,
) -> NDArray:
    """
    Compute the average density of tiles when building a neighborlist, possibly with a padding. The density
    of tiles is computed in the absence of the padding.
    """
    assert column_block == 32
    assert column_block == row_block
    N = frame.shape[0]
    # assert nblist.get_num_row_idxs() == N
    ixn_list = nblist.get_nblist(frame, box, cutoff, padding)
    tile_densities = []
    num_blocks = (N + row_block - 1) // row_block
    ixns_per_tile = min(N, column_block) * min(N, row_block)

    row_block_offset = 0
    # Each ixn_block is the column atoms in the tile that interact with at least one
    # row block atom
    for i, ixn_block in enumerate(ixn_list):
        row_coords = frame[row_block_offset : row_block_offset + row_block]
        row_coords = np.expand_dims(row_coords, axis=0)
        column_ixn_atoms = np.array(ixn_block)
        column_block_offset = 0
        for _ in range(num_blocks - i):
            # Find the interactions that would make up a tile, if the tile is empty
            # don't compute density.
            tile_column_ixns = column_ixn_atoms[
                (column_ixn_atoms >= column_block_offset) & (column_ixn_atoms < (column_block_offset + column_block))
            ]
            column_block_offset += column_block
            if tile_column_ixns.size == 0:
                continue
            col_coords = np.expand_dims(frame[tile_column_ixns], axis=1)
            deltas = delta_r(row_coords, col_coords, box)
            dij = np.linalg.norm(deltas, axis=-1)
            ixns = np.sum(dij < cutoff)
            tile_densities.append(ixns / ixns_per_tile)
        row_block_offset += row_block
    return np.array(tile_densities)


@pytest.mark.memcheck
@pytest.mark.parametrize("cutoff", [1.0, 1.2])
@pytest.mark.parametrize("precision", [np.float32])
def test_nblist_density_dhfr(cutoff, precision):
    _, _, coords, box = setup_dhfr()
    padding = 0.1
    if precision == np.float32:
        nblist = custom_ops.Neighborlist_f32(coords.shape[0], True)
    else:
        nblist = custom_ops.Neighborlist_f64(coords.shape[0], True)
    unsorted_density = np.mean(compute_tile_densities(nblist, coords, box, cutoff, padding))
    print(f"DHFR NBList Occupancy with Cutoff {cutoff}, no sort: {unsorted_density * 100.0:.3g}%")
    perm = hilbert_sort(coords, box)
    coords = coords[perm]
    density = np.mean(compute_tile_densities(nblist, coords, box, cutoff, padding))
    print(f"DHFR NBList Occupancy with Cutoff {cutoff}: {density * 100.0:.3g}%")
    assert density > unsorted_density
    assert density > 0.10


@pytest.mark.parametrize("frames", [20])
@pytest.mark.parametrize("precision", [np.float32])
def test_nblist_density(hif2a_single_topology_leg, frames, precision):
    cutoff = 1.2
    padding = 0.1
    st, host, host_name, initial_state = hif2a_single_topology_leg

    md_params = MDParams(n_eq_steps=0, steps_per_frame=200, n_frames=frames, seed=2024)

    traj = sample(initial_state, md_params, max_buffer_frames=md_params.n_frames)

    if precision == np.float32:
        nblist = custom_ops.Neighborlist_f32(initial_state.x0.shape[0], True)
    else:
        nblist = custom_ops.Neighborlist_f64(initial_state.x0.shape[0], True)
    densities = []
    perm = hilbert_sort(initial_state.x0, initial_state.box0)
    density = np.mean(compute_tile_densities(nblist, initial_state.x0[perm], initial_state.box0, cutoff, padding))
    densities.append(density)
    for i, (frame, box) in enumerate(zip(traj.frames, traj.boxes)):
        perm = hilbert_sort(frame, box)
        density = np.mean(compute_tile_densities(nblist, frame[perm], box, cutoff, padding))
        densities.append(density)
    total_steps = md_params.steps_per_frame * md_params.n_frames + md_params.n_eq_steps
    print(f"Starting density {densities[0]}")
    print(f"Final density after {total_steps} steps {densities[-1]}")
    # After equilibration the tile densities are about 17% for both solvent and complex if padding=0.1
    # and about 22% if padding=0. Note that even though this looks like it's only a 3% increase in the density,
    # adding a padding also increase the total # of tiles as well.
    assert densities[-1] >= 0.185

    # import matplotlib.pyplot as plt

    # steps = [i * md_params.steps_per_frame for i in range(len(densities))]
    # plt.plot(steps, np.array(densities) * 100.0)
    # plt.ylabel("Occupancy (%)")
    # plt.xlabel("Steps")
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # boxes = [initial_state.box0]
    # boxes.extend(traj.boxes)
    # ax2.plot(steps, [np.linalg.det(box) for box in boxes], label="Box Volume", color="red")
    # ax2.set_ylabel("Box Volume (nm^3)")
    # plt.title(f"Occupancy by Frame\n{host_name} Precision {precision.__name__}")
    # plt.savefig(f"{host_name}_density_{precision.__name__}.png", dpi=150)
    # plt.clf()
