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


def test_run_abfe_restraints(tmp_path):
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        mols = read_sdf(hif2a_dir / "ligands.sdf")
        mol = np.random.default_rng(2025).choice(mols, replace=False, size=1)[0]
        sdf_path = tmp_path / "mol.sdf"
        with Chem.SDWriter(sdf_path) as writer:
            writer.write(mol)
        output_dir = tmp_path / "output"
        config = dict(
            sdf_path=sdf_path,
            pdb_path=hif2a_dir / "5tbm_prepared.pdb",
            output_dir=output_dir,
            legs="complex",
            restraint_selection="protein and name CA",
            seed=2025,
            n_eq_steps=0,
            n_frames=50,
            n_windows=4,
            steps_per_frame=100,
            local_md_steps=100,
            forcefield=DEFAULT_FF,
            mps_workers=2,
            target_overlap=0.1,
            min_overlap=0.1,
        )
        proc = run_example(
            "run_abfe.py",
            get_cli_args(config),
        )
        assert proc.returncode == 0
        assert (output_dir / get_mol_name(mol) / "complex" / "results.npz").is_file()


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
        ({"enforce_core_core": False, "constrain_hydrogens": True}, 58),
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
            "344546b5520f99954f85330a147eaf2f936a12b04fd143c2a5f46cd712f8a61c",
            "5927bc43193c193c3d0f69e6e065ff26473aae70665f487a4d0e4eb0d80d5da8",
            "d773340569ab8a5e68d6b16b76f50c326b1919ccd6cfc8fb126dc8425c4d899a",
        ),
        (2.5, False, "solvent"): (
            "d3257d3f1a921183544ec9d9ce575f1faa64b9af74f22c2fcaf3d0aaeea93c3e",
            "365403cceb2b9da17bc685f83ffe2fd25df927e96a0db8a484b96e9b366a2d56",
            "a35183bed6623e8697a97686d420d381b9586e0f3a80c7211e1ca6cd9ff8bfbf",
        ),
        (2.5, False, "complex"): (
            "09062323da33dce2700558ce582a4fea6e958ff5cc1c77e72579eea0eb2d60e2",
            "6352a58b1b0642e2dc73d2b3fb93667fa1561f9f2463e93a70acac39641375ff",
            "429a33f76c3e171217119e7c36fb1bd5606f22b32fd4996acf282a3e75427b71",
        ),
        (2.5, True, "vacuum"): (
            "260dc9293ea6a36fc80bd9b9381ba2cdf461969add761c1ed97fdedf04e9739f",
            "a9c95d509c2d16ce0d2a339fe47f061326be8dd6fee2efc8653c248993ccaa18",
            "610bc1ce427824105101b96b10195bb7c95e9788eaf8769b1a06b0f86c6cf937",
        ),
        (2.5, True, "solvent"): (
            "0859d5ba42b8952122260fc6096db1e9006a765215b71320c295ad487d3229e1",
            "a104308a7593b339f3ab3184ca6dd494b995404b418f0c4d5dbb9b7e4e82febb",
            "79272827436a86267ddb76e433335b5686c9723ed2cb2debe26ba8b6aa49fe24",
        ),
        (2.5, True, "complex"): (
            "501df04b6c80ccf1fc3512d71aaf3862f748dfd6863999267d9a6aed00d1ada0",
            "1d26090c3659168f08aee3aaf05b7e188b327581f8511ab53d8e3c3db586e600",
            "d895bebface4ec67c205cadb0c3cae0badb435ccf3a3edd217380ab15f80aaf9",
        ),
        (4.0, True, "vacuum"): (
            "d5e3092c4c2b3457740c4dd4fe0796c7a0cedef5266c56bd40455731ef0360c7",
            "0d83eb91d490e17b43ec8c0b08410a1caa5f9adf6020eb5be9a88df39a17f1f9",
            "87d3b3961c735e60c4abfef0d027cb79c10a582ceb57e8307b3f0b18890d6ac7",
        ),
        (4.0, True, "solvent"): (
            "75c0a54cf57717827eb9bda814a37667da4032879783f52ac9cd028bfd3f665d",
            "124fb76dac69118bb4f291307fcfe83136b3729e5530d42a8cddf73c16c39f75",
            "7fdafb6698945b3b855c8a6ec10cae4f9d6d6313db13c6124181f0be30346fa8",
        ),
        (4.0, True, "complex"): (
            "05b447d1a9f6b84fb6cb23f57a3e61c7f2db4e8dc25025664ea25072953c1666",
            "18dc00c8999a446dd1345e1cf6c847db1f146c64397ea72b146833ca2f3f1d2f",
            "8cb6091dc2e9661193bf57672865ffc3874c51af4cb7630a8867b33eeea1734a",
        ),
        (4.0, False, "vacuum"): (
            "f8e7a6656e1b28f041a376d6e7acd853360eba43087d63b3ac0ffe81572a03a3",
            "b03f0f405a5c750e47d2c00933d95e7574f693f1d919f09bccba37e8bb1bd8c1",
            "c761281afc297c6390818cd539350c61208970cd5eb99ae644ce9403eb228b2d",
        ),
        (4.0, False, "solvent"): (
            "cd201363d26a9fa1c97f3d0162bbf7b872a0018a0215fcbfca0373dc4a034d99",
            "b67510954d87f20550838a501f0add39084adc6de2ddf5aad77d427bad303c4c",
            "6a716c27d9eaac68841451c9bfe7e22e09f567f18ed38a02455e1397510dc9f5",
        ),
        (4.0, False, "complex"): (
            "e41d5480d2a232fd129467469c02150a8bfc492b9a3ba5a70324882e9249bcdd",
            "706edf019d8c5c847d66c55993dbb5e94b8ca0fb67408564ac1a0e464743f560",
            "bd7118119c563b0def1365722439129a77e3b404260789c8f19146f0a3b44483",
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
            "2feb38420e3ae156432ca00591ba32ebd633253f872272488379fb5fa9ce3917",
            "da964f31de444643ed73e33d48233f7cd3b3120bc6359a46702d1328e154a05f",
        ),
        (False, "solvent"): (
            "e4d6a57ef06d9e2c68f6919b095a70023e4d50e536d96722bc3b9c9e26fc9fb6",
            "dfd49d28092ce3ec6f9d603aa279a5d6689741c5b3dcddbe4e69935f864a3bdf",
        ),
        (False, "complex"): (
            "a2d505b6f214ecfbd852487c925df536db43fcef90a3d34cb4071086484d4a9d",
            "126a95cd2f01561b57308ba2f6f2630c5113b703d479814d7824efc33e4ce227",
        ),
        (True, "vacuum"): (
            "ba5d44e58b76b50e2f1b6705f30078fbf83b3f869c8d2ec556ed693cebf2d47b",
            "4060a24524d8a60d3c1aeb40f081db4447cb6b80035acd68ea946f859f09a2bd",
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
            "f56b02b75f0b73fd2c8695995f60f68aaee9f717eb6498df4d9c69c90ece5b99",
            "4e69d59c8163b6c5b13629adb267bb34c4110f8a578527ea0f0718fc53839a6e",
            "04da7a2462c11d4965e26a5995b7738bc7f500ebd83e7e61a87aef7cc01415ab",
        ),
        ("complex", 400, True): (
            "5836c22bd313a4800909a8aea3e3b36ea1c4c9acb90049e4ce4dedddf25574d3",
            "b06bc5a07ee09f94cc150a7e34896c867d1e4e83b952dcf51653679ee61a3446",
            "102383a6a397aabfc50ff0fefc1d8342c3f2a3073bb9a8b45f1450932df94305",
        ),
        ("solvent", 390, True): (
            "20ab27955612cd401d3926aed6b7d1ea9b06532817d5c2f4783ea8a1b0d6df5e",
            "26481b71cf0de2fb96b96ab5e1ddc47f734863b3b5af030cd7dbd0000782b4b5",
            "fa40caa7a04e8efb77c5928fe33f358385ed3b9159ebe18959259c5bad22c91c",
        ),
        ("complex", 390, True): (
            "2888d85571d689a80312cf87f82c3fdca00d020dac01c7d548da70de23a3ffb0",
            "70ea17701adb225e85106793050471ae81269395997b7c746a787efbe2b61657",
            "c7f193d60b82fd28d33f6ea9b2f8ab9e5807855fdfb45b6bc8a6f09a03134e38",
        ),
        ("solvent", 400, False): (
            "6242859025e539541a1ee88b846b7e324d228efe105fe58cd8dfebb632383bef",
            "c57af02a8ea101276399dd9e601c259b5152097533dde3fab071518938e08e3f",
            "dd9c477a334733c58b9b606b3c069fa35583851ed9378928de8ff082674e5fea",
        ),
        ("complex", 400, False): (
            "272f45c5b70c962e6440a89bdd017c4891d1fcd0ed2bf993abec4d45cd7fef67",
            "eae481d8accea664c1dc0a762720d2ee5d2a042fcfe0ceb68bd32617bbcfe78f",
            "6aaa70182e5358529dabd82214b6fb370b963e9d0fff301f0a2e4733801c24b3",
        ),
        ("solvent", 390, False): (
            "1c9de55a4a52bc391902e36d3765ea6660440d6eb4dbc01e890822040121feb0",
            "1d34d530aa836cf9d233d4863ac7fbeccb49309c149590d9b20019aee18b5c3c",
            "88b87c6388ed67cad83924c1cc366e450db3ca42905d6703d5f24345a346a568",
        ),
        ("complex", 390, False): (
            "70da27079467462045cbe0ee18c7cee681071c8e84d5395a82c950298f9948b7",
            "c8dd7484d16bb6bf2744928b30353c0926d7fcce5c9d5a404f599276a9f3b205",
            "4a74c49102b4c44fbe38f38eafaa42be002febbac6b42195de5d11ac3b8cdd65",
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
