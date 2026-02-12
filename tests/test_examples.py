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
from tmd.fe.utils import get_mol_experimental_value, read_sdf, read_sdf_mols_by_name
from tmd.ff import Forcefield

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


@pytest.mark.nocuda
@pytest.mark.parametrize("scoring_method", ["best", "jaccard", "dummy_atoms"])
def test_build_rbfe_graph(scoring_method):
    with resources.as_file(resources.files("tmd.testsystems.fep_benchmark.hif2a")) as hif2a_dir:
        base_args = [str(hif2a_dir / "ligands.sdf"), "--greedy_scoring", scoring_method]
        with NamedTemporaryFile(suffix=".json") as temp:
            # Build a graph
            proc = run_example("build_rbfe_graph.py", [*base_args, temp.name])
            assert proc.returncode == 0
            with open(temp.name) as ifs:
                ref_edges = json.load(ifs)
                # The number of edges changes based on the mapping
                assert 63 <= len(ref_edges) <= 64
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
    "parameters_to_adjust",
    [{"ring_matches_ring_only": True}, {"max_connected_components": 2}, {"enforce_core_core": False}],
)
def test_build_rbfe_graph_atom_mapping_parameters(parameters_to_adjust):
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
                assert 63 <= len(ref_edges) <= 66
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
    [("untargeted", "2417ba12aff134eef27762a71b226efbc276710acad36e8f4043be41529d8eb9")],
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
        ("targeted", "32a322c08e30b53f7f3f0b8d5593d44927317e94a339427cb9f98a1c0288bbc0"),
        ("untargeted", "f3dc7ab6f6c475651628799cf2e31e032d450f541bad389a9a1e12bb170a0b20"),
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
            "c907ce95d06fe5369b2f7890839c73c60672a5be43d9bcf004437571a1a54236",
            "75852ea851ad432d2541c0d55324d599e00d0f5cec1a26f64e5ca48e86a8f7e4",
            "fe7c9c16e1defee7a416fe8352a88266eec5dc12807f2216359c2e5cdb17ada7",
        ),
        (False, "solvent"): (
            "102a9cb62df09104e7ad18818af999a414121f02ceb87d5c1a39225480e60b33",
            "a839526092dce8883e246ba43bde84d644d12c04dddbde935e90c3c6cd906563",
            "e0ed87de3b6f4040999bbe2a19313b35e0afffc4a1addfba72ab61bdce80546c",
        ),
        (False, "complex"): (
            "e9a4de4eaaa48eaf7aa49b8eb41378df27ac91157747c9f6c4888cdeae263f6d",
            "b6634c8f3bb6f6b07bc8b3b69f7862138979619e340b2d2903298e230643081a",
            "a6b9ceb2387f6bb0fdbb1b2616c958a0066d89d3a63b685baec2c56ba4df1a94",
        ),
        (True, "vacuum"): (
            "60814da2840839494a3c846193fd917a0fb794fb8f018be7fca5f30c80e4b10b",
            "1d44b1cde9499e0b360ac3dd831f15ae0685f5b5ed9a0a15e38733053a30c7ff",
            "273f44224c1c88a81aa80280ccf842a5958583a92e514c80f328f238084493f3",
        ),
        (True, "solvent"): (
            "0f1d0b07a1dde22be1ab5afd4e98a5689bf1d9080d9f79f41512fdd36f1649bd",
            "127ae4b9d9cc671a1ec9966483597b3571e58f6f7001959bd613515e369d1229",
            "2c8702b72728146b14fb2788937e235162396736f5a8cc97bb8f6510d2d1e688",
        ),
        (True, "complex"): (
            "253ffee49ced6729aed6857cd9aa0b4e2191aa4d8951bee8dba008e91743f3d6",
            "70d6367422f7641d9070997bc3f2614b636f618813bb785bfc642fef29924172",
            "aaec1b7759b24fb7e684c5604feafc12db6a2c683fa0c089a94e79450a24899e",
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
            assert output_dir / "ddg_results.csv"

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
            assert 2 <= results["n_windows"] <= config["n_windows"]
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
            "e99f12af5c0a678cad659efafa75d0e89e9fecb4ec69f2d12a07f895daef32a4",
            "041c105260026bf390c184eb8bb974aab76703a6f6e1891ddc1ffbc66c758053",
            "d48741d0423055736f60e228debc3e6cfee1b9da5d67b9dde272dc3d8dd2447d",
        ),
        ("complex", 400, True): (
            "4b16f752c5dbbda334afda7a01616c449da206925b94a353e3ee77083629fbe3",
            "0a946faa7611c73f97dace407a9d11d87390cdf132ee4bf10536f36700c59f7b",
            "5164010350abcfab2e8b3158804ddcf13bac170a3dce4e9cebb731417f74b419",
        ),
        ("solvent", 390, True): (
            "f61fbf05cfc2c62c6c1cecde7767ce0ed7b2385729de72594c60db8d556b4795",
            "a2e31f659e240ce741ba2d7241845cd6d8538049acb3ec3096a0fbe79503d7ef",
            "b1e7cc4ce6951e2e1b22e0fdfff4a0ef03de30ad0b55981e33207879871118b2",
        ),
        ("complex", 390, True): (
            "1c403b13b07a01b3ba6df9d3c381e9fdda57a51ea1ef3055607ba7d368b6f512",
            "3834d190e132d068a940b4f3ee4011da242a62fee9b5495219ace1881ae5d3f3",
            "b71a3b8f5040788fc3018fae4b6dfc37d036705c5951d5ee1f72561523f4aeab",
        ),
        ("solvent", 400, False): (
            "e99f12af5c0a678cad659efafa75d0e89e9fecb4ec69f2d12a07f895daef32a4",
            "041c105260026bf390c184eb8bb974aab76703a6f6e1891ddc1ffbc66c758053",
            "d48741d0423055736f60e228debc3e6cfee1b9da5d67b9dde272dc3d8dd2447d",
        ),
        ("complex", 400, False): (
            "4c021e76ff69fbd1131dc90109aa289886932cccd4d8a9e5bc432c6981679aa9",
            "a07bccd5d85e44aaf694437674966c590480b7eeb010d19201bfcb3ca9b76173",
            "ccf5b9d8ed5ec7b5ff34d71074f631ee42250d713c91e0fe8c9e526eddec86f2",
        ),
        ("solvent", 390, False): (
            "13416594c844156eaff35b49b93d7ca78ce2a8b6cca86ea93108303977c9552d",
            "072b72fb884d0f00de3085ecc628df6d42916fb62d42b679bcd9683411a1779b",
            "6b3e0281f4d5ebe4abe7d69eddc27d7f64dcd97f0c2c3282f8e77f2b952515a8",
        ),
        ("complex", 390, False): (
            "6f5f2ef0d996e6195a45574b1b45bf52c6421896522515fade1cd4af2f9d5918",
            "9e1185b1111d125394fbcd32ee6a16a9da18542191fa7681ae9b615ffd5c90c4",
            "f7709fa4c691ca730a1200e5cfced5db2e275954ed97858928b6918affaf7d6a",
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
            assert output_dir / "ddg_results.csv"

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
            assert 2 <= results["n_windows"] <= config["n_windows"]
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
def test_dhfr_mps_benchmark(system):
    with temporary_working_dir() as temp_dir:
        proc = run_example("mps_benchmark.py", ["--processes", "1", "--local_md", "--system", system], cwd=temp_dir)
        assert proc.returncode == 0
