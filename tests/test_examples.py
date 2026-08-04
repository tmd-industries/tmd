# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025-2026, Forrest York
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

import json
import os
import pickle
import subprocess
import sys
from collections.abc import Sequence
from csv import DictReader
from importlib import resources
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
import pytest
from common import ARTIFACT_DIR_NAME, hash_file, ligand_from_smiles, temporary_working_dir
from rdkit import Chem

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_FF, KCAL_TO_KJ
from tmd.fe.free_energy import assert_deep_eq
from tmd.fe.utils import get_mol_experimental_value, get_mol_name, read_sdf, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.utils import path_to_internal_file

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def run_example(
    example_name: str, cli_args: list[str], env: Optional[dict[str, str]] = None, cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Runs an example script

    Parameters
    ----------

    example_name: Name of the example
            The name of a file within the examples/ directory

    cli_args: List of command line arguments to pass

    env: Dictionary to override environment variables

    cwd: Directory to run in, defaults in current directory

    Returns
    -------

    Returns the completed subprocess
    """
    example_path = EXAMPLES_DIR / example_name
    assert example_path.is_file(), f"No such example {example_path}"
    subprocess_env = os.environ.copy()
    if env is not None:
        subprocess_env.update(env)
    subprocess_args = [sys.executable, str(example_path), *cli_args]
    print("Running with args:", " ".join(subprocess_args))
    proc = subprocess.run(
        subprocess_args,
        env=subprocess_env,
        check=True,
        cwd=cwd,
    )
    return proc


def get_cli_args(config: dict) -> list[str]:
    return [(f"--{key}={val}" if val is not None else f"--{key}") for (key, val) in config.items()]


@pytest.mark.parametrize(
    "n_steps, n_windows, n_frames, n_eq_steps, mps_workers",
    [(100, 4, 50, 0, 4)],
)
@pytest.mark.parametrize("seed", [2025])
def test_run_rbfe_graph_local(
    n_steps,
    n_windows,
    n_frames,
    n_eq_steps,
    mps_workers,
    seed,
):
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        mols = read_sdf(hif2a_dir / "ligands.sdf")

    config = dict(
        pdb_path=hif2a_dir / "5tbm_prepared.pdb",
        seed=seed,
        n_eq_steps=n_eq_steps,
        n_frames=n_frames,
        n_windows=n_windows,
        steps_per_frame=n_steps,
        local_md_steps=n_steps,
        forcefield=DEFAULT_FF,
        mps_workers=mps_workers,
        output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_graph_local_{seed}",
        experimental_field="IC50[uM](SPA)",
        experimental_units="uM",
        force_overwrite=None,  # Force overwrite any existing data
    )

    temp_mols = NamedTemporaryFile(suffix=".sdf")
    writer = Chem.SDWriter(temp_mols.name)
    rng = np.random.default_rng(seed)
    num_mols = 3
    for i, mol in enumerate(rng.choice(mols, replace=False, size=num_mols)):
        if i == num_mols - 1:
            # One mol should not have an experimental field
            mol.ClearProp(config["experimental_field"])
        writer.write(mol)

    writer.close()

    mols_by_name = read_sdf_mols_by_name(temp_mols.name)

    def verify_run(edges: Sequence[dict], output_dir: Path):
        assert output_dir.is_dir()
        leg_names = ["vacuum", "solvent", "complex"]
        for edge in edges:
            mol_a = edge["mol_a"]
            mol_b = edge["mol_b"]
            mol_dir = f"{mol_a}_{mol_b}"
            edge_dir = output_dir / mol_dir
            pair_by_name = read_sdf_mols_by_name(edge_dir / "mols.sdf")

            assert len(pair_by_name) == 2
            assert mol_a in pair_by_name
            assert mol_b in pair_by_name
            assert (edge_dir / "md_params.pkl").is_file()
            assert (edge_dir / "atom_mapping.svg").is_file()
            assert (edge_dir / "core.pkl").is_file()
            assert (edge_dir / "ff.py").is_file()
            assert (edge_dir / "rest_region.svg").is_file()

            assert Forcefield.load_from_file(edge_dir / "ff.py") is not None

            for leg in leg_names:
                leg_dir = edge_dir / leg
                assert leg_dir.is_dir()
                assert (leg_dir / "results.npz").is_file()
                if config.get("write_trajectories", False):
                    assert (leg_dir / "lambda0_traj.npz").is_file()
                    assert (leg_dir / "lambda1_traj.npz").is_file()
                else:
                    assert not (leg_dir / "lambda0_traj.npz").is_file()
                    assert not (leg_dir / "lambda1_traj.npz").is_file()

                assert (leg_dir / "final_pairbar_result.pkl").is_file()
                if leg in ["solvent", "complex"]:
                    assert (leg_dir / "host_config.pkl").is_file()
                else:
                    assert not (leg_dir / "host_config.pkl").is_file()
                assert (leg_dir / "hrex_transition_matrix.png").is_file()
                assert (leg_dir / "hrex_replica_state_distribution_heatmap.png").is_file()
                assert (leg_dir / "dg_errors.png").is_file()
                assert (leg_dir / "overlap_summary.png").is_file()
                assert (leg_dir / "forward_and_reverse_dg.png").is_file()
                if leg == "complex":
                    assert (leg_dir / "water_sampling_acceptances.png").is_file()

                results = np.load(str(leg_dir / "results.npz"))
                assert results["pred_dg"].size == 1
                assert results["pred_dg"].dtype == np.float64
                assert results["pred_dg"] != 0.0

                assert results["pred_dg_err"].size == 1
                assert results["pred_dg_err"].dtype == np.float64
                assert results["pred_dg_err"] != 0.0

                assert results["n_windows"].size == 1
                assert results["n_windows"].dtype == np.intp
                assert 2 <= results["n_windows"] <= config["n_windows"]
                assert isinstance(results["overlaps"], np.ndarray)
                assert all(isinstance(overlap, float) for overlap in results["overlaps"])
        assert (output_dir / "dg_results.csv").is_file()
        dg_rows = list(DictReader(open(output_dir / "dg_results.csv")))
        assert len(dg_rows) == num_mols
        expected_dg_keys = {"mol", "smiles", "pred_dg (kcal/mol)", "pred_dg_err (kcal/mol)", "exp_dg (kcal/mol)"}
        for row in dg_rows:
            assert set(row.keys()) == expected_dg_keys
            assert len(row["mol"]) > 0
            assert len(row["smiles"]) > 0
            exp_dg = row["exp_dg (kcal/mol)"]
            if exp_dg != "":
                assert np.isfinite(float(exp_dg))
                mol = mols_by_name[row["mol"]]
                ref_exp = (
                    get_mol_experimental_value(mol, config["experimental_field"], config["experimental_units"])
                    / KCAL_TO_KJ
                )
                np.testing.assert_almost_equal(float(exp_dg), ref_exp)

        assert (output_dir / "ddg_results.csv").is_file()
        ddg_rows = list(DictReader(open(output_dir / "ddg_results.csv")))
        expected_ddg_keys = {
            "mol_a",
            "mol_b",
            "pred_ddg (kcal/mol)",
            "pred_ddg_err (kcal/mol)",
            "exp_ddg (kcal/mol)",
            "mle_ddg (kcal/mol)",
            "mle_ddg_err (kcal/mol)",
        }
        for leg in leg_names:
            expected_ddg_keys.add(f"{leg}_pred_dg (kcal/mol)")
            expected_ddg_keys.add(f"{leg}_pred_dg_err (kcal/mol)")
        for row in ddg_rows:
            assert set(row.keys()) == expected_ddg_keys
            assert len(row["mol_a"]) > 0
            assert len(row["mol_b"]) > 0
            exp_ddg = row["exp_ddg (kcal/mol)"]
            if exp_ddg != "":
                assert np.isfinite(float(exp_ddg))

    with NamedTemporaryFile(suffix=".json") as temp:
        # Build a graph
        proc = run_example("build_rbfe_graph.py", [temp_mols.name, temp.name])
        assert proc.returncode == 0
        with open(temp.name) as ifs:
            edges = json.load(ifs)
            assert len(edges) == 3
            assert all(isinstance(edge, dict) for edge in edges)
            for expected_key in ["mol_a", "mol_b", "core"]:
                assert all(expected_key in edge for edge in edges)
        config["sdf_path"] = temp_mols.name
        config["graph_json"] = temp.name
        proc = run_example("run_rbfe_graph.py", get_cli_args(config))
        assert proc.returncode == 0
        verify_run(edges, Path(config["output_dir"]))


@pytest.mark.parametrize(
    "n_steps, n_windows, n_frames, n_eq_steps, mps_workers",
    [(100, 4, 50, 0, 3)],
)
@pytest.mark.parametrize("seed", [2026])
def test_run_rbfe_graph_gpcr_with_membrane_local(
    n_steps,
    n_windows,
    n_frames,
    n_eq_steps,
    mps_workers,
    seed,
):
    """Note that this is not bitwise deterministic because building the membrane system is not deterministic"""
    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)

    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "a2a_hip278.pdb") as pdb_path:
        config = dict(
            pdb_path=str(pdb_path),
            seed=seed,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            steps_per_frame=n_steps,
            local_md_steps=n_steps,
            forcefield="smirnoff_2_0_0_amber_am1ccc_amber14.py",
            mps_workers=mps_workers,
            output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_graph_local_gpcr_{seed}",
            experimental_field="r_exp_dg",
            experimental_units="kcal/mol",
            add_membrane=None,  # Add a membrane to the protein
            legs="complex",
            force_overwrite=None,  # Force overwrite any existing data
        )

    temp_mols = NamedTemporaryFile(suffix=".sdf")
    writer = Chem.SDWriter(temp_mols.name)
    rng = np.random.default_rng(seed)
    num_mols = 3
    for i, mol in enumerate(rng.choice(mols, replace=False, size=num_mols)):
        if i == num_mols - 1:
            # One mol should not have an experimental field
            mol.ClearProp(config["experimental_field"])
        writer.write(mol)

    writer.close()

    def verify_run(edges: Sequence[dict], output_dir: Path):
        assert output_dir.is_dir()
        leg_names = ["complex"]
        for edge in edges:
            mol_a = edge["mol_a"]
            mol_b = edge["mol_b"]
            mol_dir = f"{mol_a}_{mol_b}"
            edge_dir = output_dir / mol_dir
            pair_by_name = read_sdf_mols_by_name(edge_dir / "mols.sdf")

            assert len(pair_by_name) == 2
            assert mol_a in pair_by_name
            assert mol_b in pair_by_name
            assert (edge_dir / "md_params.pkl").is_file()
            assert (edge_dir / "atom_mapping.svg").is_file()
            assert (edge_dir / "core.pkl").is_file()
            assert (edge_dir / "ff.py").is_file()
            assert (edge_dir / "rest_region.svg").is_file()

            assert Forcefield.load_from_file(edge_dir / "ff.py") is not None

            for leg in leg_names:
                leg_dir = edge_dir / leg
                with open(leg_dir / "host_config.pkl", "rb") as ifs:
                    host_config = pickle.load(ifs)
                assert host_config.num_membrane_atoms > 20000

        assert (output_dir / "ddg_results.csv").is_file()
        ddg_rows = list(DictReader(open(output_dir / "ddg_results.csv")))
        expected_ddg_keys = {
            "mol_a",
            "mol_b",
            "exp_ddg (kcal/mol)",
        }
        for leg in leg_names:
            expected_ddg_keys.add(f"{leg}_pred_dg (kcal/mol)")
            expected_ddg_keys.add(f"{leg}_pred_dg_err (kcal/mol)")
        for row in ddg_rows:
            assert set(row.keys()) == expected_ddg_keys
            assert len(row["mol_a"]) > 0
            assert len(row["mol_b"]) > 0
            exp_ddg = row["exp_ddg (kcal/mol)"]
            if exp_ddg != "":
                assert np.isfinite(float(exp_ddg))

    with NamedTemporaryFile(suffix=".json") as temp:
        # Build a graph
        proc = run_example("build_rbfe_graph.py", [temp_mols.name, temp.name])
        assert proc.returncode == 0
        with open(temp.name) as ifs:
            edges = json.load(ifs)
            assert len(edges) == 3
            assert all(isinstance(edge, dict) for edge in edges)
            for expected_key in ["mol_a", "mol_b", "core"]:
                assert all(expected_key in edge for edge in edges)
        config["sdf_path"] = temp_mols.name
        config["graph_json"] = temp.name
        proc = run_example("run_rbfe_graph.py", get_cli_args(config))
        assert proc.returncode == 0
        verify_run(edges, Path(config["output_dir"]))


@pytest.mark.parametrize(
    "n_steps, n_windows, n_frames, n_eq_steps, mps_workers",
    [(100, 4, 50, 0, 1)],
)
@pytest.mark.parametrize("seed", [2026])
def test_run_abfe_gpcr_with_membrane_local(
    n_steps,
    n_windows,
    n_frames,
    n_eq_steps,
    mps_workers,
    seed,
):
    """Note that this is not bitwise deterministic because building the membrane system is not deterministic"""
    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "ligands.sdf") as sdf_path:
        mols = read_sdf(sdf_path)

    with path_to_internal_file("tmd.testsystems.gpcrs.a2a_hip278", "a2a_hip278.pdb") as pdb_path:
        config = dict(
            pdb_path=str(pdb_path),
            seed=seed,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            steps_per_frame=n_steps,
            local_md_steps=n_steps,
            forcefield="smirnoff_2_0_0_amber_am1ccc_amber14.py",
            mps_workers=mps_workers,
            output_dir=f"{ARTIFACT_DIR_NAME}/abfe_graph_local_gpcr_{seed}",
            experimental_field="r_exp_dg",
            experimental_units="kcal/mol",
            add_membrane=None,  # Add a membrane to the protein
            legs="complex",
            force_overwrite=None,  # Force overwrite any existing data
        )

    temp_mols = NamedTemporaryFile(suffix=".sdf")
    writer = Chem.SDWriter(temp_mols.name)
    rng = np.random.default_rng(seed)
    num_mols = 1
    for i, mol in enumerate(rng.choice(mols, replace=False, size=num_mols)):
        writer.write(mol)

    writer.close()

    def verify_run(output_dir: Path):
        assert output_dir.is_dir()
        mols = read_sdf(output_dir / "mols.sdf")
        assert Forcefield.load_from_file(output_dir / "ff.py") is not None
        assert (output_dir / "ff.py").is_file()
        leg_names = ["complex"]
        for mol in mols:
            mol_dir = output_dir / get_mol_name(mol)

            assert (mol_dir / "md_params.pkl").is_file()

            for leg in leg_names:
                leg_dir = mol_dir / leg
                with open(leg_dir / "host_config.pkl", "rb") as ifs:
                    host_config = pickle.load(ifs)
                assert host_config.num_membrane_atoms > 20000

        assert (output_dir / "dg_results.csv").is_file()
        ddg_rows = list(DictReader(open(output_dir / "dg_results.csv")))
        expected_ddg_keys = {
            "mol",
        }
        for leg in leg_names:
            expected_ddg_keys.add(f"{leg}_pred_dg (kcal/mol)")
            expected_ddg_keys.add(f"{leg}_pred_dg_err (kcal/mol)")
        for row in ddg_rows:
            assert set(row.keys()) == expected_ddg_keys
            assert len(row["mol"]) > 0

    config["sdf_path"] = temp_mols.name
    proc = run_example("run_abfe.py", get_cli_args(config))
    assert proc.returncode == 0
    verify_run(Path(config["output_dir"]))


@pytest.mark.fixed_output
@pytest.mark.parametrize(
    "leg, n_steps, n_windows, n_frames, n_eq_steps, mps_workers",
    [("solvent", 100, 4, 50, 0, 2), ("complex", 100, 4, 50, 0, 2)],
)
@pytest.mark.parametrize("seed", [2025])
def test_run_abfe(
    leg,
    n_steps,
    n_windows,
    n_frames,
    n_eq_steps,
    mps_workers,
    seed,
):
    leg_results_hashes = {
        "solvent": (
            "98712af1d9a1e960b4d0f4a9898acde0767d55736fac3622aa9439daeb866e4c",
            "9d718cae41d68a04bc39f3102ce45445c1c69f84983d3b85118ae8eedc8cbfe4",
        ),
        "complex": (
            "b768c61995c59501afb8113ec7229b6994e2ca425e66d16e3af9c86b2d3f82de",
            "da5fb456aafe6b3cacdcccb8e20b27bff228eb6b464cc70508027a7e1fa7c681",
        ),
    }

    def verify_endstate_hashes(leg_dir: Path, expected_hash: str):
        results_path = leg_dir / "results.npz"
        assert results_path.is_file()
        summary_data = dict(np.load(results_path))
        with NamedTemporaryFile(suffix=".npz") as temp:
            # The time changes, so need to remove prior to hashing
            summary_data.pop("time")
            np.savez(temp.name, **summary_data)
            summary_hash = hash_file(temp.name)
        endstate_hash = hash_file(leg_dir / "lambda0_traj.npz")
        # Load the summary, so we can see what changed
        assert (summary_hash, endstate_hash) == expected_hash, summary_data

    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        mols = read_sdf(hif2a_dir / "ligands.sdf")

        config = dict(
            seed=seed,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            steps_per_frame=n_steps,
            local_md_steps=n_steps,
            forcefield=DEFAULT_FF,
            mps_workers=mps_workers,
            output_dir=f"{ARTIFACT_DIR_NAME}/abfe_graph_local_{seed}",
            experimental_field="IC50[uM](SPA)",
            experimental_units="uM",
            legs=leg,
            force_overwrite=None,
            store_trajectories=None,
            target_overlap=0.1,
            min_overlap=0.1,
        )

        if leg == "complex":
            config["pdb_path"] = hif2a_dir / "5tbm_prepared.pdb"

        rng = np.random.default_rng(seed)
        mols_to_run = rng.choice(mols, replace=False, size=1)
        with NamedTemporaryFile(suffix=".sdf") as temp_mols:
            with Chem.SDWriter(temp_mols.name) as writer:
                for mol in mols_to_run:
                    writer.write(mol)

            config["sdf_path"] = temp_mols.name
            proc = run_example("run_abfe.py", get_cli_args(config))
            assert proc.returncode == 0

        output_dir = Path(config["output_dir"])
        assert Forcefield.load_from_file(output_dir / "ff.py") is not None
        assert output_dir.is_dir()
        assert len(mols_to_run) == 1
        for mol in mols_to_run:
            mol_dir = output_dir / get_mol_name(mol)
            mols_by_name = read_sdf_mols_by_name(mol_dir / "mol.sdf")
            assert len(mols_by_name) == 1
            assert (mol_dir / "md_params.pkl").is_file()
            leg_dir = mol_dir / leg

            assert (leg_dir / "results.npz").is_file()
            if "force_overwrite" in config:
                assert (leg_dir / "lambda0_traj.npz").is_file()
            else:
                assert not (leg_dir / "lambda0_traj.npz").is_file()

            assert (leg_dir / "final_pairbar_result.pkl").is_file()
            assert (leg_dir / "host_config.pkl").is_file()
            assert (leg_dir / "hrex_transition_matrix.png").is_file()
            assert (leg_dir / "hrex_replica_state_distribution_heatmap.png").is_file()
            assert (leg_dir / "dg_errors.png").is_file()
            assert (leg_dir / "overlap_summary.png").is_file()
            assert (leg_dir / "forward_and_reverse_dg.png").is_file()
            if leg == "complex":
                assert (leg_dir / "water_sampling_acceptances.png").is_file()

            results = np.load(str(leg_dir / "results.npz"))
            assert results["pred_dg"].size == 1
            assert results["pred_dg"].dtype in (np.float64, np.float32)
            assert results["pred_dg"] != 0.0

            assert results["pred_dg_err"].size == 1
            assert results["pred_dg_err"].dtype in (np.float64, np.float32)
            assert results["pred_dg_err"] != 0.0

            if leg == "complex":
                assert results["correction"].size == 1
                assert results["correction"].dtype in (np.float64, np.float32)
                assert results["correction"] != 0.0
            else:
                assert "correction" not in results

            assert results["n_windows"].size == 1
            assert results["n_windows"].dtype == np.intp
            assert 2 <= results["n_windows"] <= config["n_windows"]
            assert isinstance(results["overlaps"], np.ndarray)
            assert all(isinstance(overlap, float) for overlap in results["overlaps"])
            verify_endstate_hashes(leg_dir, leg_results_hashes[leg])


@pytest.mark.nocuda
@pytest.mark.parametrize("scoring_method, expected_edges", [("best", 58), ("jaccard", 59), ("dummy_atoms", 58)])
def test_build_rbfe_graph(scoring_method, expected_edges):
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        base_args = [str(hif2a_dir / "ligands.sdf"), "--greedy_scoring", scoring_method]
        with NamedTemporaryFile(suffix=".json") as temp:
            # Build a graph
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                ref_edges = json.load(ifs)
                # The number of edges changes based on the mapping
                assert len(ref_edges) == expected_edges
                assert all(isinstance(edge, dict) for edge in ref_edges)
                for expected_key in ["mol_a", "mol_b", "core"]:
                    assert all(expected_key in edge for edge in ref_edges)

        with NamedTemporaryFile(suffix=".json") as temp:
            # Re-build the graph, make sure the results are deterministic
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                comp_edges = json.load(ifs)

        assert len(ref_edges) == len(comp_edges)
        for ref_edge, comp_edge in zip(ref_edges, comp_edges):
            assert ref_edge == comp_edge


@pytest.mark.nocuda
def test_build_rbfe_graph_charge_hop():
    with NamedTemporaryFile(suffix=".sdf") as temp_sdf:
        with Chem.SDWriter(temp_sdf.name) as writer:
            mol_a = ligand_from_smiles("Cc1cc[nH]c1", seed=2025)
            writer.write(mol_a)

            mol_b = ligand_from_smiles("C[n+]1cc[nH]c1", seed=2025)
            writer.write(mol_b)

        base_args = [temp_sdf.name, "--enable_charge_hops"]
        with NamedTemporaryFile(suffix=".json") as temp:
            # Build a graph
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                ref_edges = json.load(ifs)
                # Only two compounds, so there will only be a single edge
                assert len(ref_edges) == 1
                assert all(isinstance(edge, dict) for edge in ref_edges)
                for expected_key in ["mol_a", "mol_b", "core"]:
                    assert all(expected_key in edge for edge in ref_edges)

        with NamedTemporaryFile(suffix=".json") as temp:
            # Re-build the graph, make sure the results are deterministic
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                comp_edges = json.load(ifs)

        assert len(ref_edges) == len(comp_edges)
        for ref_edge, comp_edge in zip(ref_edges, comp_edges):
            assert ref_edge == comp_edge

        # Should fail due to having only one ligand per charge set
        with NamedTemporaryFile(suffix=".json") as temp:
            with pytest.raises(subprocess.CalledProcessError):
                run_example("build_rbfe_graph.py", [temp_sdf.name, temp.name])


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "parameters_to_adjust, expected_edges",
    [
        ({"ring_matches_ring_only": True, "constrain_hydrogens": False}, 58),
        ({"max_connected_components": 2, "constrain_hydrogens": False}, 58),
        ({"enforce_core_core": False, "constrain_hydrogens": False}, 57),
        ({"enforce_core_core": False, "constrain_hydrogens": True}, 59),
    ],
)
def test_build_rbfe_graph_atom_mapping_parameters(parameters_to_adjust, expected_edges):
    atom_mapping_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    # Parameters to update should be in the base atom mapping set
    assert set(atom_mapping_kwargs.keys()).union(parameters_to_adjust.keys()) == set(atom_mapping_kwargs.keys())
    atom_mapping_kwargs.update(parameters_to_adjust)
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        base_args = [str(hif2a_dir / "ligands.sdf")]
        for key, val in atom_mapping_kwargs.items():
            # The initial mapping isn't exposed through the CLI
            if key == "initial_mapping":
                continue
            base_args.append(f"--atom_map_{key}")
            if not isinstance(val, bool):
                base_args.append(str(val))
            else:
                base_args.append("1" if val else "0")
        with NamedTemporaryFile(suffix=".json") as temp:
            # Build a graph
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                ref_edges = json.load(ifs)
                # The number of edges changes based on the mapping
                assert len(ref_edges) == expected_edges
                assert all(isinstance(edge, dict) for edge in ref_edges)
                for expected_key in ["mol_a", "mol_b", "core"]:
                    assert all(expected_key in edge for edge in ref_edges)

        with NamedTemporaryFile(suffix=".json") as temp:
            # Re-build the graph, make sure the results are deterministic
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                comp_edges = json.load(ifs)

        assert len(ref_edges) == len(comp_edges)
        for ref_edge, comp_edge in zip(ref_edges, comp_edges):
            assert ref_edge == comp_edge


@pytest.mark.fixed_output
@pytest.mark.parametrize("batch_size", [1, 1000])
@pytest.mark.parametrize(
    "insertion_type, last_frame_sha",
    [("untargeted", "e4200ddbeb8c6d473f141a2cfc553204f8c123e61d085670d72c192b45efe2d2")],
)
def test_water_sampling_mc_bulk_water(batch_size, insertion_type, last_frame_sha):
    with resources.as_file(resources.files("tmd.testsystems.water_exchange")) as water_exchange:
        config = dict(
            out_cif="bulk.cif",
            water_pdb=water_exchange / "bb_0_waters.pdb",
            iterations=5,
            md_steps_per_batch=1000,
            mc_steps_per_batch=1000,
            equilibration_steps=5000,
            insertion_type=insertion_type,
            batch_size=batch_size,
            use_hmr=1,
            save_last_frame="comp_frame.npz",
        )

    with temporary_working_dir() as temp_dir:
        # expect running this script to write summary_result_result_{mol_name}_*.pkl files
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / str(config["out_cif"])).is_file()
        last_frame = Path(temp_dir) / str(config["save_last_frame"])
        assert last_frame.is_file()
        assert hash_file(last_frame) == last_frame_sha


@pytest.mark.fixed_output
@pytest.mark.parametrize("batch_size", [1, 250, 512, 1000])
@pytest.mark.parametrize(
    "insertion_type, last_frame_sha",
    [
        ("targeted", "2a911f7a58f7bd1c20fc0dd5a8e1fa0aa6cfb49fc0f48b8b6df3d86189afbea2"),
        ("untargeted", "d5a12c4c429748746a825964d5c7ba4234e0c3b4649156af292532eeee85e3a4"),
    ],
)
def test_water_sampling_mc_buckyball(batch_size, insertion_type, last_frame_sha):
    # Expectations of the test:
    # 1) ggifferent batch_sizes produces identical final frames
    # 2) Different insertion_types produces different final frames, but bitwise identical to a reference final frame.

    # setup cli kwargs for the run_example_script
    with resources.as_file(resources.files("tmd.testsystems.water_exchange")) as water_exchange:
        config = dict(
            out_cif="bulk.cif",
            water_pdb=water_exchange / "bb_6_waters.pdb",
            ligand_sdf=water_exchange / "bb_centered_espaloma.sdf",
            iterations=50,
            md_steps_per_batch=1000,
            mc_steps_per_batch=5000,
            equilibration_steps=5000,
            insertion_type=insertion_type,
            use_hmr=1,
            batch_size=batch_size,
            save_last_frame="comp_frame.npz",
            # save_last_frame=reference_data_path, # uncomment me to manually update the data folders.
        )

    with temporary_working_dir() as temp_dir:
        proc = run_example("water_sampling_mc.py", get_cli_args(config), cwd=temp_dir)
        assert proc.returncode == 0
        assert (Path(temp_dir) / str(config["out_cif"])).is_file()
        last_frame = Path(temp_dir) / str(config["save_last_frame"])
        assert last_frame.is_file()
        assert hash_file(last_frame) == last_frame_sha


def verify_leg_results_hashes(leg_dir: Path, expected_hash: str):
    result_path = leg_dir / "results.npz"
    assert result_path.is_file()
    results = dict(np.load(result_path))
    with NamedTemporaryFile(suffix=".npz") as temp:
        # The time changes, so need to remove prior to hashing
        results.pop("time")
        np.savez(temp.name, **results)
        results_hash = hash_file(temp.name)
    endstate_0_hash = hash_file(leg_dir / "lambda0_traj.npz")
    endstate_1_hash = hash_file(leg_dir / "lambda1_traj.npz")
    # Load the results, so we can see what changed
    assert (results_hash, endstate_0_hash, endstate_1_hash) == expected_hash, results


@pytest.mark.fixed_output
@pytest.mark.parametrize("enable_batching", [False, True])
@pytest.mark.parametrize(
    "leg, n_windows, n_frames, n_eq_steps",
    [("vacuum", 24, 50, 1000), ("solvent", 5, 50, 1000), ("complex", 5, 50, 1000)],
)
@pytest.mark.parametrize("dt", [2.5, 4.0])
@pytest.mark.parametrize("mol_a, mol_b", [("15", "30")])
@pytest.mark.parametrize("seed", [2025])
def test_run_rbfe_legs(
    enable_batching,
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
    dt,
    mol_a,
    mol_b,
    seed,
):
    # To update the leg result hashes, refer to the hashes generated from CI runs.
    # TBD: GENERATE THE ARCHIVE WHEN CI IS MORE ROBUST
    # The CI jobs produce an artifact for the results stored at ARTIFACT_DIR_NAME
    # which can be used to investigate the results that generated the hashes.
    # Hashes are of results.npz, lambda0_traj.npz and lambda1_traj.npz respectively.
    leg_results_hashes = {
        (2.5, False, "vacuum"): (
            "09928a11e5ec43ef85f58fd9245466bc2fac6749260a7ae3c1371860f96b105c",
            "31cc1a5501330c41456688f68b12f7a2535bcd75b14a1bc248797ad872554597",
            "f77e1d024b2335f62694986a936dbea401f8a0505ae0b69df2cafb15ee5c59d6",
        ),
        (2.5, False, "solvent"): (
            "c6f801045048f259c98cd8d27f45998c136b56708087176fdc202bfe303c5a17",
            "6306404d0b6ef96f4aa6507d8b91b975238833293f53c41e685d365d291f9450",
            "c050d8da98b3bc5eb8ee26ccdf04a4d141c7562462d486d7f4789fdd288c56a0",
        ),
        (2.5, False, "complex"): (
            "19218149d3711cb224e30a607b153d5951a075a7bf4cbac5e41b7fb0df1af45d",
            "b1f03fb72889edbddd3ad3fac82ddadb060110502b7805edb25d660f6a5ed132",
            "26420580debfe248105c142e3da8c0e80ef5602d88324b9760ba7d1ac0c6301c",
        ),
        (2.5, True, "vacuum"): (
            "fc8306759626aef422ecc439f0aca4e11477c8fbcc3a4e29a71fca813ad687bd",
            "36703376f6aadb9bbd9c1bbc974be7a8318f7fb083e0b284e1e7cc664dcd1a9e",
            "d0970bdbb2351128e045d05917d3949b9bb63963aa0ab58db6fde9f03b0a0888",
        ),
        (2.5, True, "solvent"): (
            "8375d69a35e99c82de60b2fffae37ccc1743ab6d2dda691af0cf9167df2dc555",
            "3d5aa463ddf47998f560986948634e9aca426acabf902f079778e68f1c3fe77e",
            "ea301573e9dd6114848b9a18510a4f62c6c781cbee083d8b67a1e1a5d4ef0e13",
        ),
        (2.5, True, "complex"): (
            "fbe75acfedff0fde58c7b079fb7485c78b48186e72480d7ab113d1b1fceae4c5",
            "e12d4469586ce882137409ad3b3a9b4739e939d7f87be04fc95b2e5f14cd745d",
            "c6e3ab9a59bb312a6992176d3f40e80f8126ed69a42ead4123cf2cb342f3f4ed",
        ),
        (4.0, True, "vacuum"): (
            "1fae557ec3ef62ccf7b4bf31689d8cfce2e90e17de7ed922fcd5349e30ee4d3b",
            "5f4264c882e02aba89761689a9772e5db46e8651bf723ecaaaf07000de0d4ac8",
            "7aff51042e7be8ef66bb0f44d458cdfe7710422abdd9f2b141787cc265d39a0c",
        ),
        (4.0, True, "solvent"): (
            "0b2e36b90d1841445ced106442fdab3142edfcb144daa4402a7f9f914fd339c2",
            "e1148db6fa5ca07e6685b8fd2288bf0ff8a6661da05a6ac1ac6e1f5bc3564c4e",
            "3bf716d534a4528ae049c2336e2aaded6b61896ad3a0a87ce7b8c6159854f504",
        ),
        (4.0, True, "complex"): (
            "21183eef436e091aa126ca0c41350a4f16bfa869b0d4bb0b1aa718ea573c22c1",
            "e8b0d0bf4b2ff0287cbe24054206eaa666c2912be5ed0552723bd5462ea8c3b9",
            "02dd6e021d9417b5550cc2b9cc3727a1bc2e792804fcb837e3a28ec6d7dbbf65",
        ),
        (4.0, False, "vacuum"): (
            "f4aaa92848fcbd087dd4ed7f64965f6b01d0b5cd7d71960dadda6538899dcfdc",
            "c3940241d0843c5399eaf23cd25f7abcfabfb87cde2b54d9c9e8121f102688ea",
            "655a7b38f0624d6e9e1ab8066388e70233beb4d95e95167922ea9ca583cff7e4",
        ),
        (4.0, False, "solvent"): (
            "867189c73e3a8e2d3bc4c6d58826de7db4b8e9087308dea94b8f8de3978434ee",
            "a43f36e7200ae9f451c51b0ec500292cc0a6fe3530338de2a2ae9f00c9b82798",
            "6c34950200dd5883d9ba16ee72072ccb7a1bb6b4c6bcc3143c1168ae44ab5a00",
        ),
        (4.0, False, "complex"): (
            "0ad184aefc855464416952965816e6aa8e8cfd624a3209190eb9f39a6a8f6205",
            "201cdc833f861792ffee87da38f62b83b46ea87febce9516253c22f80f8bf721",
            "76a1e99d88734aed0321152e98f0bb65edc479cab01a694940d7e4d507bd5b86",
        ),
    }
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        config = dict(
            mol_a=mol_a,
            mol_b=mol_b,
            sdf_path=hif2a_dir / "ligands.sdf",
            pdb_path=hif2a_dir / "5tbm_prepared.pdb",
            seed=seed,
            legs=leg,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            forcefield=DEFAULT_FF,
            output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_{mol_a}_{mol_b}_{leg}_{seed}_{enable_batching}_{dt}",
            force_overwrite=None,  # Force overwrite any existing data
            experimental_field="IC50[uM](SPA)",
            experimental_units="uM",
            dt_fs=dt,
        )

        def verify_run(output_dir: Path):
            assert output_dir.is_dir()
            mols_by_name = read_sdf_mols_by_name(output_dir / "mols.sdf")
            assert len(mols_by_name) == 2
            assert mol_a in mols_by_name
            assert mol_b in mols_by_name
            assert (output_dir / "md_params.pkl").is_file()
            assert (output_dir / "atom_mapping.svg").is_file()
            assert (output_dir / "core.pkl").is_file()
            assert (output_dir / "ff.py").is_file()
            assert (output_dir / "ddg_results.csv").is_file()
            assert (output_dir / "rest_region.svg").is_file()

            assert Forcefield.load_from_file(output_dir / "ff.py") is not None

            leg_dir = output_dir / leg
            assert leg_dir.is_dir()
            assert (leg_dir / "results.npz").is_file()
            assert (leg_dir / "lambda0_traj.npz").is_file()
            assert (leg_dir / "lambda1_traj.npz").is_file()

            assert (leg_dir / "final_pairbar_result.pkl").is_file()
            if leg in ["solvent", "complex"]:
                assert (leg_dir / "host_config.pkl").is_file()
            else:
                assert not (leg_dir / "host_config.pkl").is_file()
            assert (leg_dir / "hrex_transition_matrix.png").is_file()
            assert (leg_dir / "hrex_replica_state_distribution_heatmap.png").is_file()
            assert (leg_dir / "dg_errors.png").is_file()
            assert (leg_dir / "overlap_summary.png").is_file()
            assert (leg_dir / "forward_and_reverse_dg.png").is_file()
            if leg == "complex":
                assert (leg_dir / "water_sampling_acceptances.png").is_file()

            results = np.load(str(leg_dir / "results.npz"))
            assert results["pred_dg"].size == 1
            assert results["pred_dg"].dtype == np.float64
            assert results["pred_dg"] != 0.0

            assert results["pred_dg_err"].size == 1
            assert results["pred_dg_err"].dtype == np.float64
            assert results["pred_dg_err"] != 0.0

            assert results["n_windows"].size == 1
            assert results["n_windows"].dtype == np.intp
            if not enable_batching:
                assert 2 <= results["n_windows"] <= config["n_windows"]
            else:
                batch_size = 8
                # If batching, can get config["n_windows"] // 8
                assert 2 <= results["n_windows"] <= max(1, config["n_windows"] // batch_size) * batch_size
            assert isinstance(results["overlaps"], np.ndarray)
            assert all(isinstance(overlap, float) for overlap in results["overlaps"])

            assert results["time"].dtype == np.float64
            assert results["time"] > 0.0

            assert results["total_ns"].dtype == np.float64
            assert results["total_ns"] > 0.0

            assert results["bisected_windows"].dtype == np.intp
            assert results["bisected_windows"] >= results["n_windows"]

            assert results["normalized_kl_divergence"].dtype == np.float64
            assert results["normalized_kl_divergence"] > 0.0

            for lamb in [0, 1]:
                traj_data = np.load(str(leg_dir / f"lambda{lamb:d}_traj.npz"))
                assert len(traj_data["coords"]) == n_frames
                assert len(traj_data["boxes"]) == n_frames
            ddg_rows = list(DictReader(open(output_dir / "ddg_results.csv")))
            assert len(ddg_rows) == 1
            assert ddg_rows[0]["mol_a"] == mol_a
            assert ddg_rows[0]["mol_b"] == mol_b

        env = {"TMD_BATCH_MODE": "on" if enable_batching else "off"}

        config_a = config.copy()
        config_a["output_dir"] = config["output_dir"] + "_a"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_a), env=env)
        assert proc.returncode == 0
        verify_run(Path(config_a["output_dir"]))
        verify_leg_results_hashes(Path(config_a["output_dir"]) / leg, leg_results_hashes[(dt, enable_batching, leg)])

        config_b = config.copy()
        config_b["output_dir"] = config["output_dir"] + "_b"
        assert config_b["output_dir"] != config_a["output_dir"], "Runs are writing to the same output directory"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_b), env=env)
        assert proc.returncode == 0
        verify_run(Path(config_b["output_dir"]))

        def verify_simulations_match(ref_dir: Path, comp_dir: Path):
            with open(ref_dir / "md_params.pkl", "rb") as ifs:
                ref_md_params = pickle.load(ifs)
            with open(comp_dir / "md_params.pkl", "rb") as ifs:
                comp_md_params = pickle.load(ifs)
            assert ref_md_params == comp_md_params, "MD Parameters don't match"
            assert ref_md_params.local_md_params is None

            with open(ref_dir / "core.pkl", "rb") as ifs:
                ref_core = pickle.load(ifs)
            with open(comp_dir / "core.pkl", "rb") as ifs:
                comp_core = pickle.load(ifs)
            assert np.all(ref_core == comp_core), "Atom mappings don't match"

            ref_results = np.load(str(ref_dir / leg / "results.npz"))
            comp_results = np.load(str(comp_dir / leg / "results.npz"))
            np.testing.assert_equal(ref_results["pred_dg"], comp_results["pred_dg"])
            np.testing.assert_equal(ref_results["pred_dg_err"], comp_results["pred_dg_err"])
            np.testing.assert_array_equal(ref_results["overlaps"], comp_results["overlaps"])
            np.testing.assert_equal(ref_results["n_windows"], comp_results["n_windows"])

            with open(ref_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
                ref_final_pairbar = pickle.load(ifs)
            with open(comp_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
                comp_final_pairbar = pickle.load(ifs)
            assert len(ref_final_pairbar.initial_states) == ref_results["n_windows"]
            assert len(ref_final_pairbar.initial_states) == len(comp_final_pairbar.initial_states)

            for ref_state, comp_state in zip(ref_final_pairbar.initial_states, comp_final_pairbar.initial_states):
                np.testing.assert_array_equal(ref_state.x0, comp_state.x0)
                np.testing.assert_array_equal(ref_state.v0, comp_state.v0)
                np.testing.assert_array_equal(ref_state.box0, comp_state.box0)
                np.testing.assert_array_equal(ref_state.ligand_idxs, comp_state.ligand_idxs)
                np.testing.assert_array_equal(ref_state.protein_idxs, comp_state.protein_idxs)
                assert_deep_eq(ref_state.potentials, comp_state.potentials)

            for lamb in [0, 1]:
                ref_traj = np.load(str(ref_dir / leg / f"lambda{lamb}_traj.npz"))
                comp_traj = np.load(str(comp_dir / leg / f"lambda{lamb}_traj.npz"))
                np.testing.assert_array_equal(ref_traj["coords"], comp_traj["coords"])
                np.testing.assert_array_equal(ref_traj["boxes"], comp_traj["boxes"])

        verify_simulations_match(Path(config_a["output_dir"]), Path(config_b["output_dir"]))


@pytest.mark.fixed_output
@pytest.mark.parametrize("enable_batching", [False, True])
@pytest.mark.parametrize(
    "leg, n_windows, n_frames, n_eq_steps",
    [("vacuum", 24, 50, 1000), ("solvent", 5, 50, 1000), ("complex", 5, 50, 1000)],
)
@pytest.mark.parametrize("mol", ["15"])
@pytest.mark.parametrize("seed", [2026])
def test_rest_ligand_flexibility(
    enable_batching,
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
    mol,
    seed,
):
    leg_hashes = {
        (False, "vacuum"): (
            "14951ebea2429b7c917fd7fdce1e2954eb7181d4c52483c3ab527720dfde73af",
            "4e36e1e16f120cde7a6a18c0a2479c851bb53ba594ccaeef3c9a4e174845861f",
        ),
        (False, "solvent"): (
            "4dfec70ec5a5487c3547685b31e21547debede02fd6f75fb1d750724faeee0ab",
            "b3dc0611afad5875cb70bea3ee876ef522bb93e5c461958606ab9ecf3d525e6d",
        ),
        (False, "complex"): (
            "c8eb9fc9ecf751a0e1adb2aeebba37858b7fd622aac90dbf10936d195be4652f",
            "c0d695f22af9b20b3b69fc6c5ed5489806da0ae6c59b4611922559287f407a1a",
        ),
        (True, "vacuum"): (
            "29f333673d66dfbc0d4c280840a8d1a77b06041883e9c85f02aa98eccc1035a6",
            "d3017192543aae2caaa8bede01f8055cc42b050cf2edbccf74010eca1c214b08",
        ),
        (True, "solvent"): (
            "aafae98745e4c90b91ddfa264f1433287cba5588cbffcb12328a842dab8306df",
            "f322e1bffdb5df936550cd98bb1a442e7dede8ebc082bcdbf2617d35b795f05d",
        ),
        (True, "complex"): (
            "a1d2e307dea63ef82cf4aced178fc7d138f410badf9d2f47d80f5af3509cb9eb",
            "8ed1f332c6ff52359de024ae69d705956667be23dbfd664e421f7a49f4e15c40",
        ),
    }

    def verify_endstate_hashes(leg_dir: Path, expected_hash: str):
        summary_path = leg_dir / "summary.npz"
        assert summary_path.is_file()
        summary_data = dict(np.load(summary_path))
        with NamedTemporaryFile(suffix=".npz") as temp:
            # The time changes, so need to remove prior to hashing
            summary_data.pop("time")
            np.savez(temp.name, **summary_data)
            summary_hash = hash_file(temp.name)
        endstate_hash = hash_file(leg_dir / "endstate_traj.npz")
        # Load the summary, so we can see what changed
        assert (summary_hash, endstate_hash) == expected_hash, summary_data

    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        mols = read_sdf_mols_by_name(hif2a_dir / "ligands.sdf")
        with NamedTemporaryFile(suffix=".sdf") as temp:
            with Chem.SDWriter(temp.name) as writer:
                writer.write(mols[mol])
            config = dict(
                sdf_path=temp.name,
                pdb_path=hif2a_dir / "5tbm_prepared.pdb",
                seed=seed,
                leg=leg,
                n_eq_steps=n_eq_steps,
                n_frames=n_frames,
                n_windows=n_windows,
                forcefield=DEFAULT_FF,
                output_dir=f"{ARTIFACT_DIR_NAME}/rest_endstate_{mol}_{leg}_{seed}_{enable_batching}",
            )

            env = {"TMD_BATCH_MODE": "on" if enable_batching else "off"}

            config_a = config.copy()
            config_a["output_dir"] = config["output_dir"] + "_a"
            proc = run_example("rest_ligand_flexibility.py", get_cli_args(config_a), env=env)
            assert proc.returncode == 0
            verify_endstate_hashes(Path(config_a["output_dir"]) / mol, leg_hashes[(enable_batching, leg)])

            config_b = config.copy()
            config_b["output_dir"] = config["output_dir"] + "_b"
            assert config_b["output_dir"] != config_a["output_dir"], "Runs are writing to the same output directory"
            proc = run_example("rest_ligand_flexibility.py", get_cli_args(config_b), env=env)
            assert proc.returncode == 0
            verify_endstate_hashes(Path(config_b["output_dir"]) / mol, leg_hashes[(enable_batching, leg)])


@pytest.mark.fixed_output
@pytest.mark.parametrize("enable_batching", [False, True])
@pytest.mark.parametrize(
    "leg, n_windows, n_frames, n_eq_steps, local_steps",
    [
        ("solvent", 5, 50, 1000, 400),
        ("complex", 5, 50, 1000, 400),
        ("solvent", 5, 50, 1000, 390),
        ("complex", 5, 50, 1000, 390),
    ],
)
@pytest.mark.parametrize("mol_a, mol_b", [("15", "30")])
@pytest.mark.parametrize("seed", [2025])
def test_run_rbfe_legs_local(
    enable_batching,
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
    local_steps,
    mol_a,
    mol_b,
    seed,
):
    # To update the leg result hashes, refer to the hashes generated from CI runs.
    # TBD: GENERATE THE ARCHIVE WHEN CI IS MORE ROBUST
    # The CI jobs produce an artifact for the results stored at ARTIFACT_DIR_NAME
    # which can be used to investigate the results that generated the hashes.
    # Hashes are of results.npz, lambda0_traj.npz and lambda1_traj.npz respectively.
    leg_results_hashes = {
        ("solvent", 400, True): (
            "71d4d926a5f346b465aebeba6e14396b9c015281b533f541c9e15d7e7f4c20b9",
            "3947481d8d2c929f8a4bed77fa960770cf269c1413efbab97a6ea7958a7e484e",
            "68368192311c2d4b3852e69311c69b83d5fe71f96b969585bbd44ad2ddaf1795",
        ),
        ("complex", 400, True): (
            "d56dd1bf9e2773e00b5fd519c924d14a105b6e5a18ef5bba615216bba055e955",
            "3584f6e4861b16af191d6208336ceed87ab0fbaae368d44677e6383ce1759adb",
            "64e09c63e5b56df12e0690542e81d35cedcf594d10d98f6fd5a1724bcf2e0206",
        ),
        ("solvent", 390, True): (
            "e53364fd68097112ceec04c0fce3ea01721b4d8b277abc8b0f01cd93d2c1724a",
            "0c3cf9cb74ae9d2ec0a248359cca5d94c8f21962aa3de34cff710cb7a85d4ef8",
            "35729dc6dfdf700c32b66ac7e577eb8eb33aefb5ae9fa32de083691f30f84060",
        ),
        ("complex", 390, True): (
            "60b0c56048879d11a170d3cd2dacec4492690d73080a5729d2115d325aa32324",
            "c7885d8622f7758052a5f5a1ad925f59af4673feb354b3a093cdf9038a90835b",
            "47e2d71644595ec53b05bc1a2ff4b8a6da3cd84d992e3089aff54c7e6b61e0d0",
        ),
        ("solvent", 400, False): (
            "12c08f13cf72a3d67fe2dd6f5f730921fe2442380e568b29f38b81a085fe5975",
            "8af66f6a0ad779f53218bedba983952c9f74ca585c55a9699950b4f9d3812ba0",
            "cb207ae1bb4363ba264ec097845551487f330ca1123036f92efe4ad5fcd73f0f",
        ),
        ("complex", 400, False): (
            "3a239e3b3ed61a66753118de108fc463f181dc376de44f23cba242bc7c1db0e9",
            "ecf57d0377826bd8b54878d76867561159aad46607c38bf456f7ab21fecb8f30",
            "29e991772e6568bbd5faa5f30efd7e5ab60d07c60abc4a12f5f7f14ff3d1e600",
        ),
        ("solvent", 390, False): (
            "817de34496737931605b2077a31b18a2531661ad7e82b0ffb83300650b4957db",
            "784d78376afdfbfba44100dbc2ca0e705ab386e7978c73ee1b98e2224c0514f8",
            "0a82006c5a937f2634d7ec31511840d07828a2caa1bdf140c49a48c51d0cf74a",
        ),
        ("complex", 390, False): (
            "260856d06955bf9a6e4ac75af9ec1f8bc8e780da66149f1f780cf749383c63bf",
            "be784c3753ca3b421c9f3d9cdc23831e084678ee87e570230b4fb0121ca228f0",
            "702ca6253e5ef1160e2c96b91c2b24ccc84bbf584c5db9c847971ce83db2f3ab",
        ),
    }
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        config = dict(
            mol_a=mol_a,
            mol_b=mol_b,
            sdf_path=hif2a_dir / "ligands.sdf",
            pdb_path=hif2a_dir / "5tbm_prepared.pdb",
            seed=seed,
            legs=leg,
            n_eq_steps=n_eq_steps,
            n_frames=n_frames,
            n_windows=n_windows,
            forcefield=DEFAULT_FF,
            output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_local_{mol_a}_{mol_b}_{leg}_{seed}_{local_steps}_{enable_batching}",
            local_md_steps=local_steps,
            local_md_radius=2.0,
            force_overwrite=None,  # Force overwrite any existing data
            experimental_field="IC50[uM](SPA)",
            experimental_units="uM",
        )

        def verify_run(output_dir: Path):
            assert output_dir.is_dir()
            mols_by_name = read_sdf_mols_by_name(output_dir / "mols.sdf")
            assert len(mols_by_name) == 2
            assert mol_a in mols_by_name
            assert mol_b in mols_by_name
            assert (output_dir / "md_params.pkl").is_file()
            assert (output_dir / "atom_mapping.svg").is_file()
            assert (output_dir / "core.pkl").is_file()
            assert (output_dir / "ff.py").is_file()
            assert (output_dir / "ddg_results.csv").is_file()

            assert Forcefield.load_from_file(output_dir / "ff.py") is not None

            leg_dir = output_dir / leg
            assert leg_dir.is_dir()
            assert (leg_dir / "results.npz").is_file()
            assert (leg_dir / "lambda0_traj.npz").is_file()
            assert (leg_dir / "lambda1_traj.npz").is_file()

            assert (leg_dir / "final_pairbar_result.pkl").is_file()
            if leg in ["solvent", "complex"]:
                assert (leg_dir / "host_config.pkl").is_file()
            else:
                assert not (leg_dir / "host_config.pkl").is_file()
            assert (leg_dir / "hrex_transition_matrix.png").is_file()
            assert (leg_dir / "hrex_replica_state_distribution_heatmap.png").is_file()
            if leg == "complex":
                assert (leg_dir / "water_sampling_acceptances.png").is_file()
            assert (leg_dir / "forward_and_reverse_dg.png").is_file()

            results = np.load(str(leg_dir / "results.npz"))
            assert results["pred_dg"].size == 1
            assert results["pred_dg"].dtype == np.float64
            assert results["pred_dg"] != 0.0

            assert results["pred_dg_err"].size == 1
            assert results["pred_dg_err"].dtype == np.float64
            assert results["pred_dg_err"] != 0.0

            assert results["n_windows"].size == 1
            assert results["n_windows"].dtype == np.intp
            if not enable_batching:
                assert 2 <= results["n_windows"] <= config["n_windows"]
            else:
                batch_size = 8
                # If batching, can get config["n_windows"] // 8
                assert 2 <= results["n_windows"] <= max(1, config["n_windows"] // batch_size) * batch_size
            assert isinstance(results["overlaps"], np.ndarray)
            assert all(isinstance(overlap, float) for overlap in results["overlaps"])

            for lamb in [0, 1]:
                traj_data = np.load(str(leg_dir / f"lambda{lamb:d}_traj.npz"))
                assert len(traj_data["coords"]) == n_frames
                assert len(traj_data["boxes"]) == n_frames

        env = {"TMD_BATCH_MODE": "on" if enable_batching else "off"}

        config_a = config.copy()
        config_a["output_dir"] = config["output_dir"] + "_a"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_a), env=env)
        assert proc.returncode == 0
        verify_run(Path(config_a["output_dir"]))
        verify_leg_results_hashes(
            Path(config_a["output_dir"]) / leg, leg_results_hashes[(leg, local_steps, enable_batching)]
        )

        config_b = config.copy()
        config_b["output_dir"] = config["output_dir"] + "_b"
        assert config_b["output_dir"] != config_a["output_dir"], "Runs are writing to the same output directory"
        proc = run_example("run_rbfe_legs.py", get_cli_args(config_b), env=env)
        assert proc.returncode == 0
        verify_run(Path(config_b["output_dir"]))

        def verify_simulations_match(ref_dir: Path, comp_dir: Path):
            with open(ref_dir / "md_params.pkl", "rb") as ifs:
                ref_md_params = pickle.load(ifs)
            with open(comp_dir / "md_params.pkl", "rb") as ifs:
                comp_md_params = pickle.load(ifs)
            assert ref_md_params == comp_md_params, "MD Parameters don't match"
            assert ref_md_params.local_md_params is not None

            with open(ref_dir / "core.pkl", "rb") as ifs:
                ref_core = pickle.load(ifs)
            with open(comp_dir / "core.pkl", "rb") as ifs:
                comp_core = pickle.load(ifs)
            assert np.all(ref_core == comp_core), "Atom mappings don't match"

            ref_results = np.load(str(ref_dir / leg / "results.npz"))
            comp_results = np.load(str(comp_dir / leg / "results.npz"))
            np.testing.assert_equal(ref_results["pred_dg"], comp_results["pred_dg"])
            np.testing.assert_equal(ref_results["pred_dg_err"], comp_results["pred_dg_err"])
            np.testing.assert_array_equal(ref_results["overlaps"], comp_results["overlaps"])
            np.testing.assert_equal(ref_results["n_windows"], comp_results["n_windows"])

            with open(ref_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
                ref_final_pairbar = pickle.load(ifs)
            with open(comp_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
                comp_final_pairbar = pickle.load(ifs)
            assert len(ref_final_pairbar.initial_states) == ref_results["n_windows"]
            assert len(ref_final_pairbar.initial_states) == len(comp_final_pairbar.initial_states)

            for ref_state, comp_state in zip(ref_final_pairbar.initial_states, comp_final_pairbar.initial_states):
                np.testing.assert_array_equal(ref_state.x0, comp_state.x0)
                np.testing.assert_array_equal(ref_state.v0, comp_state.v0)
                np.testing.assert_array_equal(ref_state.box0, comp_state.box0)
                np.testing.assert_array_equal(ref_state.ligand_idxs, comp_state.ligand_idxs)
                np.testing.assert_array_equal(ref_state.protein_idxs, comp_state.protein_idxs)
                assert_deep_eq(ref_state.potentials, comp_state.potentials)

            for lamb in [0, 1]:
                ref_traj = np.load(str(ref_dir / leg / f"lambda{lamb}_traj.npz"))
                comp_traj = np.load(str(comp_dir / leg / f"lambda{lamb}_traj.npz"))
                np.testing.assert_array_equal(ref_traj["coords"], comp_traj["coords"])
                np.testing.assert_array_equal(ref_traj["boxes"], comp_traj["boxes"])

        verify_simulations_match(Path(config_a["output_dir"]), Path(config_b["output_dir"]))


@pytest.mark.nightly
@pytest.mark.parametrize("system", ["dhfr", "hif2a-rbfe"])
def test_dhfr_benchmark(system):
    with temporary_working_dir() as temp_dir:
        proc = run_example("benchmark.py", ["--processes", "1", "2", "--local_md", "--system", system], cwd=temp_dir)
        assert proc.returncode == 0
