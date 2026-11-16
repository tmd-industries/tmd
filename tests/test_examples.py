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
            "64925fb504e5357163fb861c4c6646ed7d4b7ec791139b91e382c8df45de7e30",
            "5e7777eba88f4a3af4ec3f4564a59c717e84f69ee9a89aa750162f156454e20b",
        ),
        "complex": (
            "6826b660c37a7a3e99febd429807379a4fabb511c93a82710966eac22b604cd9",
            "c2c3dd45fbaef5afdcf89ee073d76a9618e2e7ff82f1958bfbae3bc322dfed46",
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
            "7cd070d4248552b911fe36258669e206247810a2c2e39d2c2d0f0ab46cf5bb5f",
            "1a9490f7fa94ca974cb519b438fb23d10ef6ed91103b7faf78e2bfbe03b8dc7c",
            "57fe3b97cec712994bf2fbcb42f68b1b4775ebf2da1ef601e284c8024e8a28fd",
        ),
        (2.5, False, "solvent"): (
            "76cd8a768c5e3273a2043aeb219263103567341fda5cf3aad87c4f8272067439",
            "239fb8ffc1fb286c7cbfef3f906f5a2d03d0320f7c79542b799ac7ee86b889b9",
            "9a67a83dc6f6c414b7884a4c1016e9ab7de3596896612d2a3c4ede9f613122fd",
        ),
        (2.5, False, "complex"): (
            "6999c49128bf7000ffb9595fd5544dcf82da8ef005025500652dc2025cb737a2",
            "710ceb74ddc76d25dd753a547d75bf6f6b3ca480bbb7abb52de0c7ef8b67c16f",
            "736a7a47cc506c3287fccd64fa26d2d52da354e18163198cc82bfbe87d561e9e",
        ),
        (2.5, True, "vacuum"): (
            "7d582c077052fa5dad43049fef13b010c2ba813aa12e413f6ef2d2762c4d6894",
            "a2e3e9c4c48afc7175a6a6936591853ef70b4ce9a2d8295fd129f6770613d453",
            "5562c0ef657736da2befd521853989027baa8d7087c542ba4b8338db9417ab1f",
        ),
        (2.5, True, "solvent"): (
            "38dd07b9e8cc89b723b613ac094f672f8ef474cc706a17e074725c6c1a73d01c",
            "0777afa46a7036a11672497e4507954e07267c5ae039d5b61be13d354d16caf2",
            "6aef659d1f818fbf0acae2af3d8b35920a2ac1870e05eefe5967a8cf82ebd3d7",
        ),
        (2.5, True, "complex"): (
            "3023baa31631af949dba82b9a39153f4c69629580bd492935be32e66cc069847",
            "6f5b36f16b92a71dc2b7125ea895982686a7f1e1413a46ffece9c3c11f17d399",
            "d019e7057854e9834c4b6df1d9112fb1165683641bceedd70d7e7b00d8e277c0",
        ),
        (4.0, True, "vacuum"): (
            "b2d263bda26cb3e23d0d226f8dace9f0599dc4c72baa3ffb9a57aa10593dd039",
            "8e0f2d8de49599cdd300672cd7ca0e56fbe7bd7ccb371ca481c2cc3f53a33012",
            "01041663e514dfdecf6487fe3ab4e6a08462ed6ce5368797f4f698f10e4f3054",
        ),
        (4.0, False, "vacuum"): (
            "8b531b26c53ff1cf846bb2fb6d2d7a46fca6e39b113bcdf191a5122cf43ea30e",
            "30f886aa422b67560d369061d6462aa43bd5bcf531914f232911adffb3f22a84",
            "58cca6f72298ac66a5165c584bb3860452c86a5d16600ac52aa52c754135a2fb",
        ),
        (4.0, True, "solvent"): (
            "a805c7be4f74cc6ac270e174e99b8716f6a8cea7971897fbc53cf754168d3abb",
            "f48bed3327a8069bea7ab26cba1588f81ba0332406d6640b957b7861f9ebece6",
            "9ecefb6247be5037bf2bab762c44fdcc92d40b7380677a68f7d5625588a4d1b6",
        ),
        (4.0, False, "solvent"): (
            "0078b2e08d4376c99827bb8f254ec4613b55cc8156d1ce3b6bf4833e322fd98b",
            "8a83a55a72eecb790289d742ae8aa4c67726ec563ba4a8a1c93b0e5ecb4c32b5",
            "53c0d8384152fa78a25195b3ddd1e36ebf7ffd0b402e986a121cb656c5c01bfb",
        ),
        (4.0, True, "complex"): (
            "fe79276c5183d2d5909e15a199f91ffce4648a3fc41c57c8b8c87c607ca2807a",
            "36b94224cf0ea1ee97e3aa891b3217a7034bba492a30f0583434d77007801142",
            "856869dbca44d6f400cc7c8918434d3287577b946df575fad219895cbd151aef",
        ),
        (4.0, False, "complex"): (
            "d7decaa668748f979510b87fee3c94b9ce29401060165b99df0de659e49169cb",
            "77a274a2dcaa9bb36b91263f2157f23941ea5e5cf5bbb702f6acaaffc4849c6a",
            "47341009b92d6136dd1dbf80bca3574caaf474d9e750d5f6511c66a0f142dad8",
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
            "4b4f4aae7d9af242e9fd4c15269378db5d99da7a20ae2553e4db2e0b96bd16e5",
            "9b2d35221fe9889c441cd75e572dd4bf0c73c6e38062d9e302dd4c269bed63f8",
        ),
        (False, "solvent"): (
            "113fd0d787db8cec3b9d5975f9a7025684fe032bac8fe938f88c99d4e8cebc32",
            "1b325c7716222c9c9343792b8f1b6fcfb25ca34c789690177f751c81505a46ec",
        ),
        (False, "complex"): (
            "e196c48ef8ceb0ea94ef4511d3be45925e35b68b880ccf85319039007db9902c",
            "21627d316581526ad64d947850b834ff8ce1705d6fbb252cc72b7bd92e9f4d4f",
        ),
        (True, "vacuum"): (
            "46fa4e9e79cf8a13ce14c127e96da32282b20ee77673ed905d86c4b27d225766",
            "f7b575dd76564c2d4deb3ff3fd7390849100604ed9a6fc738ee334d5e7192317",
        ),
        (True, "solvent"): (
            "1a4348b8c833beb99bc65cb8961e9451baab09543f708de508473f9ba4ca0ef6",
            "c58e0fc198c616bda4b2405dedc65ec7b9e68a01ad81b9f77c13c869acd03c68",
        ),
        (True, "complex"): (
            "c236532df302a22eee9cfb8781dbbbd7379183dbaafdc552c4792a59ff414092",
            "d708f661a2cec3c99f27414aa9bfb8d23319d90cbdff5ea9756ce97e9e6e2917",
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
            "2537eee2845628b7f9592dddab51b57317e5b715c66e5e9ec32216935bfab15a",
            "22182946bf221193408844224c3529f66fdf96009a3e67de2fa344ad5cf97ae6",
            "dcf3ce4cc663c1136e7761da147a1283e8d7a691c73dfe8c7878ab9da90c628c",
        ),
        ("complex", 400, True): (
            "80768d2617f813e142bd49e289aedf66452be55fca1bc552f305027d83f1a548",
            "61bbb1695a7eb4a6fbadde5e68789a2376401091253fcfec1cdcc8adb438265f",
            "4de68ba1582be773a1991c4d3e62b98fe210e3211c6fed8e3e1b9f10f8248572",
        ),
        ("solvent", 390, True): (
            "6f23d25e837ebbd5403c46a7f9e925e731fbc26e0e2bfbacce221d5353ec2219",
            "56d42cfb100bbe57efe6ac0ade6b6c1c12a4f64c99e229b09c06cbb03e27a878",
            "a6f135fab7183728190e1753d491ba0899c821d37684d2b05ae5dac089bdf831",
        ),
        ("complex", 390, True): (
            "6bf0fb448e41d775c498e002a4b7bbf95679f0626b9d4f4c76cc969db9f92906",
            "0573d6fffbca394dd35413ebeb7151e382a11f1a49694923089ce99d4d2cb3d2",
            "113f166ad3bef41d604eb15e206c4f406050b9a8205cf3c8a72471b1fda08d6f",
        ),
        ("solvent", 400, False): (
            "b2597c3840f432ac8576bcfe9351609ece3f0d7659e80d18cbe0f190f79f8ca9",
            "1b62e684f0f7e076a53a49e4d8b4e41a01d27329f3051016eff7feca62f5dd0b",
            "181b628f62f40e772abc59d8de076352f0f9840c4e986a918b629f1b82737b57",
        ),
        ("complex", 400, False): (
            "306e592a9a8c4f960960afd29875d4d29c37ee2af29f4d720f43abe3ae8f274b",
            "10647744a61ccf5fad2d494b6aab606cb5445d4e1d84ed05bbd4fcbc82efd808",
            "9cf5651fb43bf7853d06f249034801d6e79f50bec121916f54c28688104880a8",
        ),
        ("solvent", 390, False): (
            "3add8910d1e7f2af41460745ddcb9cf7d43b9238b759ac7782d8bf1c83e60a1c",
            "a2669ff166aef7e810cd1e852e888f8efd29d6baf353eb1322b34567e1ccddff",
            "330ce51d6cdedb6a63288314c633a49ed566f2c269c494317e9e224dd3dad481",
        ),
        ("complex", 390, False): (
            "8b8ce900db185534f6973791a9c019663a1b819356acc0bb217a0bd98a010ed4",
            "685b734e5f337e4b48eed14038709c80ceb6d82e4479ca7098972fdd3446d88b",
            "b5b3c49aa9daef18581bad209c192f44e3b22ba25abb9e632ff9c5f156a898e8",
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
