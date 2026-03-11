"""Tests for configurable nonbonded cutoff (WP6).

Verifies that:
- compute_beta is correct and special-cases cutoff=1.2
- Python switch_fn respects the cutoff parameter
- CUDA nonbonded potential works with non-default cutoffs
- CUDA and Python reference agree for non-default cutoffs
"""

import numpy as np
import pytest
from scipy.special import erfc as scipy_erfc

from tmd.constants import (
    DEFAULT_NONBONDED_BETA,
    DEFAULT_NONBONDED_CUTOFF,
    compute_beta,
)

pytestmark = [pytest.mark.memcheck]


class TestComputeBeta:
    def test_default_cutoff_returns_exact_default_beta(self):
        assert compute_beta(DEFAULT_NONBONDED_CUTOFF) == DEFAULT_NONBONDED_BETA

    def test_special_case_1_2(self):
        """cutoff=1.2 must return exactly 2.0, not an approximation."""
        assert compute_beta(1.2) == 2.0

    @pytest.mark.parametrize("cutoff", [0.8, 1.0, 1.5, 2.0])
    def test_erfc_product_is_conserved(self, cutoff):
        """beta * cutoff should give the same erfc value as default."""
        beta = compute_beta(cutoff)
        ref_erfc = scipy_erfc(DEFAULT_NONBONDED_BETA * DEFAULT_NONBONDED_CUTOFF)
        actual_erfc = scipy_erfc(beta * cutoff)
        np.testing.assert_allclose(actual_erfc, ref_erfc, rtol=1e-10)

    @pytest.mark.parametrize("cutoff", [0.8, 1.0, 1.2, 1.5, 2.0])
    def test_beta_positive(self, cutoff):
        assert compute_beta(cutoff) > 0

    def test_smaller_cutoff_larger_beta(self):
        """Smaller cutoff requires larger beta to maintain erfc suppression."""
        assert compute_beta(0.8) > compute_beta(1.2) > compute_beta(2.0)


class TestSwitchFn:
    def test_switch_fn_respects_cutoff(self):
        """switch_fn should go to zero at the given cutoff, not hardcoded 1.2."""
        import jax.numpy as jnp
        from tmd.potentials.nonbonded import switch_fn

        # With cutoff=0.8, distances between 0.8 and 1.2 should be zero
        dij = jnp.array([0.5, 0.79, 0.81, 1.0, 1.19])
        result_small_cutoff = switch_fn(dij, cutoff=0.8)
        result_default_cutoff = switch_fn(dij, cutoff=1.2)

        # At d=0.5, both should be nonzero
        assert float(result_small_cutoff[0]) > 0
        assert float(result_default_cutoff[0]) > 0

        # At d=0.81, small cutoff should be zero, default should be nonzero
        assert float(result_small_cutoff[2]) == 0.0
        assert float(result_default_cutoff[2]) > 0

        # At d=1.19, both should differ (small=0, default>0)
        assert float(result_small_cutoff[4]) == 0.0
        assert float(result_default_cutoff[4]) > 0

    def test_switch_fn_at_cutoff_is_zero(self):
        import jax.numpy as jnp
        from tmd.potentials.nonbonded import switch_fn

        for cutoff in [0.8, 1.0, 1.2, 1.5]:
            val = float(switch_fn(jnp.array(cutoff), cutoff=cutoff))
            assert val == 0.0, f"switch_fn({cutoff}, cutoff={cutoff}) = {val}, expected 0.0"

    def test_switch_fn_below_cutoff_is_one(self):
        import jax.numpy as jnp
        from tmd.potentials.nonbonded import switch_fn

        for cutoff in [0.8, 1.0, 1.2, 1.5]:
            # At r=0, switch_fn should be ~1.0
            val = float(switch_fn(jnp.array(0.01), cutoff=cutoff))
            np.testing.assert_allclose(val, 1.0, atol=1e-6)


class TestCUDANonbondedCutoff:
    """Test that the CUDA nonbonded implementation works with non-default cutoffs."""

    @pytest.fixture()
    def small_system(self):
        """Create a small system with random coords and params."""
        rng = np.random.default_rng(42)
        N = 64
        coords = rng.random((N, 3)).astype(np.float64) * 2.0  # spread in 2nm box
        box = np.eye(3) * 3.0  # 3nm cubic box

        # Generate params: q, sig, eps, w
        params = np.zeros((N, 4), dtype=np.float64)
        params[:, 0] = (rng.random(N) - 0.5) * 0.1  # charges
        params[:, 1] = rng.random(N) * 0.05  # sigma/2
        params[:, 2] = rng.random(N) * 0.1  # sqrt(eps)
        params[:, 3] = 0.0  # w coords

        exclusion_idxs = np.zeros((0, 2), dtype=np.int32)
        scale_factors = np.zeros((0, 2), dtype=np.float64)

        return N, coords, params, box, exclusion_idxs, scale_factors

    @pytest.mark.parametrize("cutoff", [0.9, 1.0, 1.2, 1.5])
    def test_different_cutoffs_give_finite_energies(self, small_system, cutoff):
        """Verify that CUDA nonbonded produces finite energies for different cutoffs."""
        from tmd.potentials import Nonbonded

        N, coords, params, box, exclusion_idxs, scale_factors = small_system
        beta = compute_beta(cutoff)

        nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
        impl = nb.to_gpu(np.float64).unbound_impl

        _, _, u = impl.execute(coords, params, box)
        assert np.isfinite(u), f"Energy not finite for cutoff={cutoff}: {u}"

    def test_different_cutoffs_give_different_energies(self, small_system):
        """Verify that changing the cutoff actually changes the computed energy."""
        from tmd.potentials import Nonbonded

        N, coords, params, box, exclusion_idxs, scale_factors = small_system

        energies = {}
        for cutoff in [0.9, 1.2, 1.5]:
            beta = compute_beta(cutoff)
            nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
            impl = nb.to_gpu(np.float64).unbound_impl
            _, _, u = impl.execute(coords, params, box)
            energies[cutoff] = u

        # Different cutoffs should give different energies
        assert energies[0.9] != energies[1.2], "cutoff 0.9 and 1.2 gave same energy"
        assert energies[1.2] != energies[1.5], "cutoff 1.2 and 1.5 gave same energy"

    @pytest.mark.parametrize("cutoff", [1.0, 1.2])
    def test_cuda_matches_python_reference(self, small_system, cutoff):
        """CUDA and Python reference should agree for the given cutoff."""
        from tmd.potentials import Nonbonded
        from tmd.potentials.nonbonded import nonbonded_block_unsummed

        N, coords, params, box, exclusion_idxs, scale_factors = small_system
        beta = compute_beta(cutoff)

        # CUDA energy
        nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
        impl = nb.to_gpu(np.float64).unbound_impl
        _, _, u_cuda = impl.execute(coords, params, box)

        # Python reference: compute full NxN block and sum upper triangle
        u_ref_block = np.array(nonbonded_block_unsummed(coords, coords, box, params, params, beta, cutoff))
        # Sum upper triangle (avoid double counting and self-interactions)
        u_ref = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                u_ref += float(u_ref_block[i, j])

        np.testing.assert_allclose(u_cuda, u_ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("cutoff", [0.9, 1.0, 1.2, 1.5])
    def test_forces_are_finite(self, small_system, cutoff):
        """Verify forces are finite for different cutoffs."""
        from tmd.potentials import Nonbonded

        N, coords, params, box, exclusion_idxs, scale_factors = small_system
        beta = compute_beta(cutoff)

        nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
        impl = nb.to_gpu(np.float64).unbound_impl

        du_dx, _, _ = impl.execute(coords, params, box)
        assert np.all(np.isfinite(du_dx)), f"Forces not finite for cutoff={cutoff}"

    @pytest.mark.parametrize("cutoff", [0.9, 1.5])
    def test_interaction_group_with_non_default_cutoff(self, small_system, cutoff):
        """Verify NonbondedInteractionGroup works with non-default cutoffs."""
        from tmd.potentials import NonbondedInteractionGroup

        N, coords, params, box, _, _ = small_system
        beta = compute_beta(cutoff)

        ligand_idxs = np.arange(8, dtype=np.int32)
        env_idxs = np.arange(8, N, dtype=np.int32)

        ixn = NonbondedInteractionGroup(N, ligand_idxs, beta, cutoff, col_atom_idxs=env_idxs)
        impl = ixn.to_gpu(np.float64).unbound_impl

        du_dx, _, u = impl.execute(coords, params, box)
        assert np.isfinite(u), f"IxnGroup energy not finite for cutoff={cutoff}"
        assert np.all(np.isfinite(du_dx)), f"IxnGroup forces not finite for cutoff={cutoff}"


class TestBlockCountScaling:
    """Verify that the cutoff-scaled kernel block heuristic produces correct results.

    The heuristic scales as (cutoff / 1.2)^3 so that larger cutoffs launch
    enough CUDA blocks to cover the increased number of interacting tiles.
    We test this indirectly: if the block count were too low the kernel would
    silently drop tile interactions and the energy would disagree with the
    Python reference.
    """

    @pytest.fixture()
    def dense_system(self):
        """A moderately dense system that stresses the block-count heuristic."""
        rng = np.random.default_rng(2026)
        N = 128
        coords = rng.random((N, 3)).astype(np.float64) * 2.5
        box = np.eye(3) * 4.0  # large enough for cutoff=1.4 + padding

        params = np.zeros((N, 4), dtype=np.float64)
        params[:, 0] = (rng.random(N) - 0.5) * 0.2   # charges
        params[:, 1] = rng.random(N) * 0.04            # sigma/2
        params[:, 2] = rng.random(N) * 0.08            # sqrt(eps)
        params[:, 3] = 0.0                              # w

        exclusion_idxs = np.zeros((0, 2), dtype=np.int32)
        scale_factors = np.zeros((0, 2), dtype=np.float64)
        return N, coords, params, box, exclusion_idxs, scale_factors

    @pytest.mark.parametrize("cutoff", [1.0, 1.2, 1.4])
    def test_cuda_matches_reference_with_scaled_blocks(self, dense_system, cutoff):
        """CUDA energy must match Python reference at non-default cutoffs.

        A too-small block count would cause the CUDA kernel to miss tile
        interactions, producing an energy that diverges from the reference.
        """
        from tmd.potentials import Nonbonded
        from tmd.potentials.nonbonded import nonbonded_block_unsummed

        N, coords, params, box, exclusion_idxs, scale_factors = dense_system
        beta = compute_beta(cutoff)

        nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
        impl = nb.to_gpu(np.float64).unbound_impl
        _, _, u_cuda = impl.execute(coords, params, box)

        u_ref_block = np.array(nonbonded_block_unsummed(coords, coords, box, params, params, beta, cutoff))
        u_ref = np.sum(np.triu(u_ref_block, k=1))

        np.testing.assert_allclose(u_cuda, u_ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("cutoff", [1.0, 1.2, 1.4])
    def test_ixn_group_matches_reference_with_scaled_blocks(self, dense_system, cutoff):
        """NonbondedInteractionGroup energy must match reference at cutoff=1.4."""
        from tmd.potentials import NonbondedInteractionGroup
        from tmd.potentials.nonbonded import nonbonded_block_unsummed

        N, coords, params, box, _, _ = dense_system
        beta = compute_beta(cutoff)

        ligand_idxs = np.arange(16, dtype=np.int32)
        env_idxs = np.arange(16, N, dtype=np.int32)

        ixn = NonbondedInteractionGroup(N, ligand_idxs, beta, cutoff, col_atom_idxs=env_idxs)
        impl = ixn.to_gpu(np.float64).unbound_impl

        _, _, u_cuda = impl.execute(coords, params, box)

        # Python reference: ligand-env block
        u_ref_block = np.array(
            nonbonded_block_unsummed(
                coords[ligand_idxs], coords[env_idxs], box,
                params[ligand_idxs], params[env_idxs], beta, cutoff,
            )
        )
        u_ref = float(np.sum(u_ref_block))

        np.testing.assert_allclose(u_cuda, u_ref, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("cutoff", [1.0, 1.2, 1.4])
    def test_forces_finite_with_scaled_blocks(self, dense_system, cutoff):
        """Forces must remain finite when block count is scaled for larger cutoffs."""
        from tmd.potentials import Nonbonded

        N, coords, params, box, exclusion_idxs, scale_factors = dense_system
        beta = compute_beta(cutoff)

        nb = Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff)
        impl = nb.to_gpu(np.float64).unbound_impl

        du_dx, _, u = impl.execute(coords, params, box)
        assert np.isfinite(u), f"Energy not finite for cutoff={cutoff}"
        assert np.all(np.isfinite(du_dx)), f"Forces not finite for cutoff={cutoff}"
