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
            "cda7fe864328c93ed9753e0324a969ba8dd8c95989c5c4dda0431caa68356916",
            "9d718cae41d68a04bc39f3102ce45445c1c69f84983d3b85118ae8eedc8cbfe4",
        ),
        "complex": (
            "f5a12e88a123e634935efc03a2498763d52b85598a837df9dd639c7c14439a72",
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
            "dd1e18de84e869705667c762b1f43ff77ecbb029fbb864126600ba72c8150714",
            "495b73e2f814394dd3f325e47de0627a78f7a7139e48dcff249c2b4755c73579",
            "a26899b6ec00859b9bb45186e055626191e0b6684b9867ee9ca1e8fe4be8e91a",
        ),
        (2.5, False, "solvent"): (
            "65ce588e283448322dd6d8d5dbd9e1f8e298c4240a0f67f6b7f07721b150d05c",
            "314c05fce90938ad2551753b23c4f08080e08351bd377654e95fa25ca95b589e",
            "cfb56ec04f2bb7e391aca1be3df8b3070b4be83165746edb21b648375b6782e2",
        ),
        (2.5, False, "complex"): (
            "ee7ddf3bc59452b6bee3e229f921409be0c29ec47031a3bc311e7a13c162d41f",
            "a41598c921e7ddf81ab562d5cf11e302c37db0376dc6ad0cf3e85ef62b3848d5",
            "f97df56a5d1c25ad7b11026c1d5cd3fe7cb97b2475741cc1cb6c08905a1c8527",
        ),
        (2.5, True, "vacuum"): (
            "46c70e6f0f572e82b2ee76cdbc09e9818d34d1e8c628ea368680cccb0df96d73",
            "b99461b6e038e4b17e8add32a497c9af3d75af486f9c5ce055bea37bbb142541",
            "3bd714a2162ed5ea69e576d789967fb730dc9aa10cc53620feed9db6a955c681",
        ),
        (2.5, True, "solvent"): (
            "a6491be77ca4b019f4a12db2dd0b6211d24746d6e4aa5f26bb478e84497158cc",
            "e5fe80f21abae7a8eb8d83c7e9e766ec8b8aba69b05dcd772fcfe6298d5d3702",
            "8dfeb7b0464ce789f9a917d2be55fc5c8ba94423f16d906a0d683c0af06dd7bf",
        ),
        (2.5, True, "complex"): (
            "4f90e1cdf432c68216045c16e71c0135a68941b0ec059f8ced0fd94a63dd1fce",
            "59871639fc5de98fbfec23b5760bcc816a8014db89fd4aa48097aef51e11363f",
            "ddf7550cd0852ad87c886102507e2b1a30d0309caa8b384abd2e457384b347e2",
        ),
        (4.0, True, "vacuum"): (
            "5f050627f513ce96f1dfc1b86dd734d9cb094c34a68188eb4cdf08bc1b1a8363",
            "d153e1f943dc3ae78c5a5dbfb3eca2f07f553e7d1f9eab08714467230f33219a",
            "e4c5200a02219bfd2009f1d49a83e88f6bf5054e43815d7c691e70940118558a",
        ),
        (4.0, True, "solvent"): (
            "333f900aba383b83bf649760e3f4f8c32d7a1a728a7212689b5c7d64c9f8062a",
            "19e7814d4448b7def28d320b5fe63a24dffab6daf758775e2921e0230a0f9a37",
            "66b31befa7b0389cc9ec1c193676a31cf2bc3a47b8bbbd099a4c128154d258e0",
        ),
        (4.0, True, "complex"): (
            "6d6818d5881170664287bdd65f72e38a6f6a722caf72598f899745508e613cfd",
            "28d5e7ecda895714e5ada99f28f87dbb22dba373efcd057c62e0b76ef4421126",
            "81fc0be4dd5ffebd48c040a3116ce142275c3f2ac8b4e056b47736493e321aee",
        ),
        (4.0, False, "vacuum"): (
            "fbaf698b84ac5c88fa5f5017be7c22094cee6e1a4afe7bf165f894d4f069cb6c",
            "1f7edaab761334c0ed31a83310ab4996337daafe1259b808c7cb1a30cc58bd6b",
            "caaa160bfb07b6cc5359be5ce7a7446ec71c01d81c01b5cc82853c0c823b549b",
        ),
        (4.0, False, "solvent"): (
            "805f7e885a1059e38e53f9fbeb89eeacedfaf432931c930b64f2372a29aa985b",
            "35615cb72133fdaeaaef27e8cbf2b5cdfdd7cf2f055f6c7e182481ccf9284bd6",
            "27eaf1553258163dc3560d2160fa7f6e5edc7ecca3b220331569e367bfad5d82",
        ),
        (4.0, False, "complex"): (
            "59e42cbe624e8d5123f0018bc981239b97236ecca8cd5ddf0f7e8cfecfb608fa",
            "b5612cdc5a2f4fcfc0308454ffebc316640fae29bf524e488650465072ad1656",
            "a39071008a3a1d8e90fb00470d0c0c77203691fe9892ce49aeaa5c7e55e8ef3f",
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


@pytest.mark.parametrize(
    "leg, n_windows, n_frames, n_eq_steps",
    [("solvent", 5, 10, 200), ("complex", 5, 10, 200)],
)
@pytest.mark.parametrize("mol_a, mol_b", [("15", "30")])
@pytest.mark.parametrize("seed", [2025])
def test_run_septop_legs_deterministic(
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
    mol_a,
    mol_b,
    seed,
):
    """Two runs of run_septop with the same seed must agree bitwise."""
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
            force_overwrite=None,  # Force overwrite any existing data
        )

        output_dirs = []
        for suffix in ["a", "b"]:
            run_config = config.copy()
            run_config["output_dir"] = f"{ARTIFACT_DIR_NAME}/septop_{mol_a}_{mol_b}_{leg}_{seed}_{suffix}"
            proc = run_example("run_septop_legs.py", get_cli_args(run_config))
            assert proc.returncode == 0
            output_dirs.append(Path(run_config["output_dir"]))

        ref_dir, comp_dir = output_dirs
        assert ref_dir != comp_dir, "Runs are writing to the same output directory"

        ref_results = np.load(str(ref_dir / leg / "results.npz"))
        comp_results = np.load(str(comp_dir / leg / "results.npz"))
        # The solvent leg carries no restraint correction, so pred_dg must equal raw_dg there.
        if leg == "solvent":
            np.testing.assert_equal(ref_results["correction"], 0.0)
            np.testing.assert_equal(ref_results["pred_dg"], ref_results["raw_dg"])
        for key in ["pred_dg", "pred_dg_err", "raw_dg", "correction", "correction_a", "correction_b", "n_windows"]:
            np.testing.assert_equal(ref_results[key], comp_results[key], err_msg=f"{key} is not deterministic")
        np.testing.assert_array_equal(ref_results["overlaps"], comp_results["overlaps"])

        with open(ref_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
            ref_final_pairbar = pickle.load(ifs)
        with open(comp_dir / leg / "final_pairbar_result.pkl", "rb") as ifs:
            comp_final_pairbar = pickle.load(ifs)
        assert len(ref_final_pairbar.initial_states) == len(comp_final_pairbar.initial_states)

        for ref_state, comp_state in zip(ref_final_pairbar.initial_states, comp_final_pairbar.initial_states):
            np.testing.assert_array_equal(ref_state.x0, comp_state.x0)
            np.testing.assert_array_equal(ref_state.v0, comp_state.v0)
            np.testing.assert_array_equal(ref_state.box0, comp_state.box0)
            np.testing.assert_array_equal(ref_state.ligand_idxs, comp_state.ligand_idxs)
            assert_deep_eq(ref_state.potentials, comp_state.potentials)

        for lamb in [0, 1]:
            ref_traj = np.load(str(ref_dir / leg / f"lambda{lamb}_traj.npz"))
            comp_traj = np.load(str(comp_dir / leg / f"lambda{lamb}_traj.npz"))
            np.testing.assert_array_equal(ref_traj["coords"], comp_traj["coords"])
            np.testing.assert_array_equal(ref_traj["boxes"], comp_traj["boxes"])


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
            "21aef59ab429b90d40b719ea79e43e675e0edbac8d3f7e3b36e53360128ac90b",
            "4d2344370162d4eb4efa9de1c9fa3788ab21939d1af6f13cebdd0f74f073e348",
        ),
        (False, "solvent"): (
            "e4d6a57ef06d9e2c68f6919b095a70023e4d50e536d96722bc3b9c9e26fc9fb6",
            "dfd49d28092ce3ec6f9d603aa279a5d6689741c5b3dcddbe4e69935f864a3bdf",
        ),
        (False, "complex"): (
            "aa5b9701189e640dd3746bef319e28c8a99c69571a238148a4cfed381114cc6a",
            "f700c19c12a6ab02b59906448bc5b61d216cfda2d9b83db8b56e816e1f54ba15",
        ),
        (True, "vacuum"): (
            "dec9db1b3265cc32b49fbea85a18b0317b4069f1537d8d719823bf999ca21e38",
            "487d1403ba47e32386c0d4c687e6f791e9bc225c1c499fd22f0511a2153254af",
        ),
        (True, "solvent"): (
            "49cf0ae8adf008af8644b4c24de945c81fd22320c020a7c55540e39c026781b6",
            "d8d4b87579f3dabd6f5096900a6a15242d890cdc378925128564cb0e42a81b7b",
        ),
        (True, "complex"): (
            "657e64344b19ec3e0996d92c25f7ec3d87ee4d8588c138795a0f621cfb8caa99",
            "8af6361aa154494f881d7d3083cb08dc77183a500610fdaa3fe04174dbb70933",
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
            "76e67a5ef88a8e63c25028ebd71c6feeb819c002255b597442657260f1fdc98d",
            "3d00abcbebf536267ca9bc37a7a384f56fc6e97e000827593f16160e519f6f5f",
            "98a85be4a4ebdc160e49a18b3b3a3c3ce6a16908d5aa24aac4d1712097da2903",
        ),
        ("complex", 400, True): (
            "2b1d2f2c2edc5cfcbd51ae83fc4c50a87083361ec509f5ad36720374ff634d19",
            "db1f610ada0a39d641e3227d100633a24ed25519e6df7d94ee253203e3ed1f7d",
            "e6931cd29a05eaa88a369099db5f262b2618b1c66c66ce88fe365b1c4693ae45",
        ),
        ("solvent", 390, True): (
            "152d613d2e77eadf52df1c3de6ff5768786315232c3b18fea4410f6ca0392cc1",
            "6a1bd8821469c766c7b37a85d10508592c13e209089d727a05911aea2831ba5b",
            "dd9af718de58d5d9d9a3ad80b457a760031783f45fb33880c8b1af2f11b321b3",
        ),
        ("complex", 390, True): (
            "f16fb63370c848ebfe8aee6c4108a57889857c272933e13d97fbd7c91a2d6802",
            "ba0cda2494f307b5ea166239f74b228ad68dd2207718b582c2af7725ee205824",
            "ef97e28802badae800eb8e9d470ec90e69fa6f217d9f4cb90021ba9cb0305179",
        ),
        ("solvent", 400, False): (
            "c31671bfbdcf2a5289d1548dff031239ea1c952f9b0d336f64846816ec1ce2ed",
            "961f8e85b87f40f4734b3536afbed74ec0b32c190c85981c6538f54af75cc9ea",
            "c2e3f406c7d028ce33a25f4565cdc9c169198cb678eea7e50d01301a2a4f5655",
        ),
        ("complex", 400, False): (
            "16dda4d5bf3c0d00556cd2834b6ef0cc0e44a1afaf36658cd057ba51b3cc50f4",
            "67a3daa7ad21480f58e19ffacae6dc8336a0cf8569697fea14b4658a8db0d190",
            "5a8a802e4a7a3bf3ff0c9346752dd1df15548261550701b029f9395c4a3d1378",
        ),
        ("solvent", 390, False): (
            "f0231a86c8b003b776fafd663ba0a46aeaa78c59903eef28693e74f43a575535",
            "0c0c7b3b8a7f99eebad622df487bae2fcaa7119ee7ed1dd754d02a454f8e5488",
            "b2cba1e7ca255efb39ca173275205e3f62dc0887355cafcd9e353e4a9f122224",
        ),
        ("complex", 390, False): (
            "b4bfb4f624f22cd5635458bff2cf5736283c1a918263ec6a983575951f44410d",
            "c64758e09b551fc4bc229bb76348620e87a5115b655ffa68504daef656654121",
            "ad6a8752cf3e0bc3fa837062ea713d0f425a7f40cfd8cd1cf011e57049ea3309",
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
