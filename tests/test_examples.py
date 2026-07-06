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
            "19266d7b260631d725a4ff7f2c10f64aca9d07d1191abce813a8cbda29c882c7",
            "c889a49cf0dc03147c9b74f760f7d2573b731ca52ba0acdc3df25d46278291cf",
        ),
        "complex": (
            "1904bff39d7ea9a19f0a61d28c73b350611e8ed46108a6f274cda6efb3afd8ea",
            "1ce0ecf50731129fde7fa8508bb5a4ae3eacda726a12a52109a7245605c5f82d",
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
    [({"ring_matches_ring_only": True}, 58), ({"max_connected_components": 2}, 58), ({"enforce_core_core": False}, 57)],
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
@pytest.mark.parametrize("mol_a, mol_b", [("15", "30")])
@pytest.mark.parametrize("seed", [2025])
def test_run_rbfe_legs(
    enable_batching,
    leg,
    n_windows,
    n_frames,
    n_eq_steps,
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
        (False, "vacuum"): (
            "eed2d235a6a28bc584ca118a45742acf7288c1eef75701cb9eae07d0a6166294",
            "e38f24e61370203fb75a31eeff55a0ae65b5bbbbad836f077a6ddbb76426c750",
            "d9584ac5c1e113f01e8e26a47cc38c4dfeb29421ebbefadc1c64ef6259ee7d99",
        ),
        (False, "solvent"): (
            "f44c2a26fbd0d49d3349fc0573ff2162fbceb330d27617ee8a28b59afcafee83",
            "38b32c4f820166fd89f1dde219a85a77e72d58ef8718d8ebd2599733fbef7b83",
            "67da8e6da98a3556b9cc442fe80f49e61f9aa7226f6830f9c7e79b3d432f61da",
        ),
        (False, "complex"): (
            "5b47a57ff4c7864414ee66f3c82de60ce948d43d0d9bdcda8b667a7e0b20ce40",
            "588804cebb4bf4a9013d753acc21181a4f690242ba4bee44c5e63bc5d51c72e6",
            "a10b1cbe3d4cc48a1dc45d7f3f02d998bd084e5d856e007894f5f89b518f3c63",
        ),
        (True, "vacuum"): (
            "2e57af3a402e7ad6faf9d56c4af6259ee8f9cc0f7d61944ae376691a48da0b36",
            "7bc53f3acb2dabc83fe73532243f276a9f209827105b5103aad0c7033707c6e3",
            "742970e49dcf5abc00ad52cad2ca29b31cca7a9dfb8f9ce4eb8fa1b4d7e0724f",
        ),
        (True, "solvent"): (
            "7d019fa34bc6a264c68ed8c14df04e6d141fd79a47b359e40c1a342ad9cc729e",
            "7ad93e0844fb569f723d231d3cc35e079bc58428398519e944c518cf90f890d3",
            "61cfa56ed59ee01626281f1af354638c5136ebd7984b3fcf3030cbe63396d159",
        ),
        (True, "complex"): (
            "76f9525a3384e3721f6459ed6ffae5ff9ec826e95fa263f7731e3246759aba5a",
            "1be4d635b8787905a6403ca493209d629bacef4843bc38c88c2a3ea4519711b8",
            "f5849cb6825c8f86d53eb0dffe6538113d4094f8fc08efda9cf04a22a1ac8747",
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
            output_dir=f"{ARTIFACT_DIR_NAME}/rbfe_{mol_a}_{mol_b}_{leg}_{seed}_{enable_batching}",
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
        verify_leg_results_hashes(Path(config_a["output_dir"]) / leg, leg_results_hashes[(enable_batching, leg)])

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
            "4e4efdbda7ade105f77fc08c7f3edf9dabbe42b6eab1a17cab3ef9c550a29e1c",
            "b43f4e7184b030a8e8c3b9a2aacfeab9819ae29c8be43f139c5e58aec4b3e42b",
        ),
        (False, "solvent"): (
            "63bff54dec081d86fd19952b267b310642294711a2fde58ac0dfd1df7c4e31c6",
            "487066e570e3dfa0e657be402fc5ee2c9f3cf1a21c4f4109b30683d21b318243",
        ),
        (False, "complex"): (
            "2a365930d49575b7abcaf7387c7a65cbae5d1a96997ebfe7a871a5b302ee5cef",
            "6d073cadd80b4d51cb41f3e3e251f005a1932ac6904b54829768d98b19075b44",
        ),
        (True, "vacuum"): (
            "ad421823f975569bb64c6ccc898bd4a624fedf60abb1f62b9fced0c70f873b7c",
            "adc01f04f3295bc671d7f6bb1273d310041d3259a337e97bf1838c0c48576a00",
        ),
        (True, "solvent"): (
            "b3546bfeb77101820c8f37ee370abd6e55a401a4b96a29e3749189e3e92a5713",
            "9dcae94dbc2f896e9252edc2541ea49590acb78158899267a052600e9c33e75d",
        ),
        (True, "complex"): (
            "17877de1dd7332ab98d220e027448ebd5351038bb148abc8297c8aa59e1d24ea",
            "24c76e26ef63572ded01171875c1d0042e4efb50bf9cb29486bcd47bc30c669b",
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
            "9e962575bad71f96ad8f3cfc18d9ee548cc806f2b8d52bdc437726bbe740b4bb",
            "fca9911ff013409f754fe3c421dfa8cb60564b311ce6519ce79341fa824082c1",
            "51b2ae289042aa1f2f12e1c67a42618dac1d336b804e4e616664e92fb1e85bec",
        ),
        ("complex", 400, True): (
            "6674aa859b93065d0c4572a2efc6d5c45ce9625b3cbc8851801b3872be1741da",
            "a80fd51f943f2055177d243fbb68ffb136c3b27eae0b01a1aaf3ed2b4fdd6176",
            "1dc69addaddd95af7fcbd27882038e6bfd8e7df5071c7e285a04dff97d1bb587",
        ),
        ("solvent", 390, True): (
            "39b8c4b3ed510ab414bb6e71a8ac6a1c097bc1db77a10ef8506559da0c116025",
            "a4cac1ce1f08ad95dcd1b4974b3a5ae0efb553eceeedc0eca24faf7ac26f189a",
            "3aa185cf453422d8542a95fbab9daed8e5c31f220fb84319d2150672769b2598",
        ),
        ("complex", 390, True): (
            "eecf475b6ec916d1f186759fdcca8078846adb1b3a6a0fb39eee95c7bb965252",
            "7501fc80e6b2e34122b892dc022cd6297bc0bea8d74e040f2985bb900cf9e2f4",
            "f31b531520c5c21ebf9fa523de0fdb4d562e6d7d806c6666d31d0fa45144f51b",
        ),
        ("solvent", 400, False): (
            "f52dc812a63ceea4968df568376d184a3dde32cabb1a75fbbd6b7bf5ab770204",
            "231857b2bc7f97d92706d8a74246f79df3ac0e23af94fe6394f68e07c9dbfee8",
            "49b698dc57859eb844476fde61d791c74d8b76673ff6a8444b248fa618ce3f63",
        ),
        ("complex", 400, False): (
            "92144a977a13d471bfd1ff2739242d8ea19c8e0cd23b0042fa76684888863939",
            "af2744d69ec866c4b072abef7e4e85bad955387a22fbda58c0f47c19f28f1c26",
            "670ee5d30784e35eccc20c09e841478d39a9dbbffd50809282ed0ce0f9059aed",
        ),
        ("solvent", 390, False): (
            "cc60ddcb72a4995a5cb5673db08a2f5946b8ba2ee1eb99d93594dc2357ec0f01",
            "868f23addc4ce05d03d5f92f6efda107492bc29da1a92ad936846e4005d058a5",
            "ff74fd682fbab315b94290ada4833015f82e520881d2d128b3b7cbacaf3a0527",
        ),
        ("complex", 390, False): (
            "c95b48f9c4ec564e9c7ccd9d24a2b3649317b757b87bc9347948b9b991ccb4d5",
            "0b25ef486a6ff145f04045acfa8d719b536811dd2332973d403e031eeaac088d",
            "a227a86124891caf8ca66e37f44922f65b61b3803a60c6c79a2c07a02a0c7795",
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
