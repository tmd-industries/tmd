import json
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe.atom_mapping import get_cores_and_diagnostics, get_num_dummy_atoms
from tmd.fe.utils import get_mol_name, read_sdf

STAR_MAP = "star_map"
GREEDY = "greedy"
BEST = "best"

JACCARD = "jaccard"
DUMMY_ATOMS = "dummy_atoms"

CORE_FIELD = "core"
MOL_FIELD = "mol"


def atom_mapping_jaccard_distance(mol_a, mol_b, core) -> float:
    return 1 - (len(core) / (mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - len(core)))


def refine_mapping_wrapper(args):
    try:
        mol_a, mol_b, atom_mapping_kwargs = args
        assert atom_mapping_kwargs["initial_mapping"] is not None
        cores, _ = get_cores_and_diagnostics(mol_a, mol_b, **atom_mapping_kwargs)
        core = cores[0]
    except Exception:
        core = atom_mapping_kwargs["initial_mapping"]
    return core


def atom_mapping_wrapper(pair, atom_mapping_kwargs: dict[str, Any] | None = None):
    if atom_mapping_kwargs is None:
        atom_mapping_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    try:
        idx, mol_a, mol_b = pair
        cores, diag = get_cores_and_diagnostics(mol_a, mol_b, **atom_mapping_kwargs)
        core = cores[0]
    except Exception:
        core = None
        diag = None
    return idx, core, diag


def generate_nxn_atom_mappings(mols: list, atom_mapping_kwargs: dict[str, Any] | None = None) -> NDArray:
    if atom_mapping_kwargs is None:
        atom_mapping_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    all_pairs = [((i, j), mols[i], mols[j]) for i in range(len(mols)) for j in range(i + 1, len(mols))]
    core_matrix = np.empty((len(mols), len(mols)), dtype=object)
    with Pool() as pool:
        for res in pool.imap_unordered(
            partial(atom_mapping_wrapper, atom_mapping_kwargs=atom_mapping_kwargs), all_pairs
        ):
            (mol_a_idx, mol_b_idx), core, _ = res
            core_matrix[mol_a_idx, mol_b_idx] = core
    return core_matrix


def build_star_graph(hub_cmpd_name: str, mols: list, atom_mapping_kwargs: dict[str, Any] | None = None):
    mols_by_name = {get_mol_name(m): m for m in mols}
    assert hub_cmpd_name in mols_by_name
    hub_cmpd = mols_by_name[hub_cmpd_name]

    graph = nx.DiGraph()
    for mol in mols:
        graph.add_node(get_mol_name(mol), **{MOL_FIELD: mol})

    i = 0
    pairs = []
    for name, mol in mols_by_name.items():
        if name == hub_cmpd_name:
            continue
        pairs.append((i, hub_cmpd, mol))
        i += 1
    with Pool() as pool:
        for res in pool.imap_unordered(partial(atom_mapping_wrapper, atom_mapping_kwargs=atom_mapping_kwargs), pairs):
            i, core, _ = res
            graph.add_edge(hub_cmpd_name, get_mol_name(pairs[i][2]), **{CORE_FIELD: core})
    return graph


def build_greedy_graph(
    mols, scoring_methods: list[str], k_min_cut: int = 2, atom_mapping_kwargs: dict[str, Any] | None = None
) -> nx.DiGraph:
    """Build a densely connected graph using a greedy method

    Parameters
    ----------
    mols:
        List of Rdkit mols

    scoring_methods: list of string
        List of name of scoring methods to use. Will return the graph with the fewest number of dummy atoms

    k_min_cut: int
        Number of edges that can be cut before producing a disconnected graph. See networkx.k_edge_augmentation for more details

    Returns
    -------
        networkx.DiGraph
    """
    assert k_min_cut >= 1
    assert len(scoring_methods) >= 1
    mol_name_to_idx = {get_mol_name(m): i for i, m in enumerate(mols)}
    core_matrix = generate_nxn_atom_mappings(mols, atom_mapping_kwargs=atom_mapping_kwargs)

    def get_graph_mean_dummy_atoms(g):
        return np.mean([data[DUMMY_ATOMS] for _, _, data in g.edges(data=True)])

    best_graph = None
    # Try both scoring methods
    for score_method in scoring_methods:
        possible_edges = []
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                core = core_matrix[i, j]
                if core is None:
                    continue
                mol_a = mols[i]
                mol_b = mols[j]

                if score_method == JACCARD:
                    edge_score = atom_mapping_jaccard_distance(mol_a, mol_b, core)
                elif score_method == DUMMY_ATOMS:
                    edge_score = float(get_num_dummy_atoms(mol_a, mol_b, core))
                else:
                    assert 0, f"Invalid score_fn: {score_method}"
                possible_edges.append((get_mol_name(mol_a), get_mol_name(mol_b), edge_score))

        graph = nx.DiGraph()
        for mol in mols:
            graph.add_node(get_mol_name(mol), **{MOL_FIELD: mol})

        # If there are only two mols then, simply connect the two and exit
        if len(mols) == 2:
            i = 0
            j = 1
            core = core_matrix[i, j]
            dummy_atoms = float(get_num_dummy_atoms(mols[i], mols[j], core))
            graph.add_edge(get_mol_name(mols[i]), get_mol_name(mols[j]), **{CORE_FIELD: core, DUMMY_ATOMS: dummy_atoms})
            return graph

        for a, b in nx.k_edge_augmentation(graph.to_undirected(as_view=True), k_min_cut, possible_edges, partial=True):
            (i, j) = sorted([mol_name_to_idx[a], mol_name_to_idx[b]])
            core = core_matrix[i, j]
            src_mol = get_mol_name(mols[i])
            dst_mol = get_mol_name(mols[j])
            dummy_atoms = float(get_num_dummy_atoms(mols[i], mols[j], core))
            graph.add_edge(src_mol, dst_mol, **{CORE_FIELD: core, DUMMY_ATOMS: dummy_atoms})
        if best_graph is None:
            best_graph = graph
        elif get_graph_mean_dummy_atoms(best_graph) > get_graph_mean_dummy_atoms(graph):
            best_graph = graph
    assert best_graph is not None
    return best_graph


def refine_atom_mapping(nx_graph, cutoff: float, atom_mapping_kwargs: dict[str, Any] | None = None):
    if atom_mapping_kwargs is None:
        atom_mapping_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    edges = []
    for a, b, data in nx_graph.edges(data=True):
        if CORE_FIELD not in data:
            continue
        pair_kwargs = atom_mapping_kwargs.copy()
        pair_kwargs["initial_mapping"] = data[CORE_FIELD]
        pair_kwargs["ring_cutoff"] = cutoff
        pair_kwargs["chain_cutoff"] = cutoff
        edges.append((nx_graph.nodes[a][MOL_FIELD], nx_graph.nodes[b][MOL_FIELD], pair_kwargs))
    if len(edges) == 0:
        return nx_graph
    with Pool() as pool:
        refined_cores = pool.map(refine_mapping_wrapper, edges)
    refined_graph = nx_graph.copy()
    refined_graph.update(
        [(get_mol_name(a), get_mol_name(b), {CORE_FIELD: core}) for (a, b, _), core in zip(edges, refined_cores)]
    )
    return refined_graph


def main():
    parser = ArgumentParser()
    parser.add_argument("ligands_sdf")
    parser.add_argument("output_path", help="Json file to write out containing all of the edges")
    parser.add_argument(
        "--mode", default=GREEDY, choices=[STAR_MAP, GREEDY], help="Whether to generate a star map or a greedy map"
    )
    parser.add_argument("--hub_cmpd", help="If generating a star map, provide a hub compound")
    parser.add_argument(
        "--refine_cutoff",
        type=float,
        help="Refine the final graph with a new atom map cutoff, uses the initial atom mapping.",
    )
    parser.add_argument(
        "--greedy_scoring",
        choices=[BEST, JACCARD, DUMMY_ATOMS],
        default=BEST,
        help=f"How to score edges when generating greedy maps. The {BEST} option will try multiple scoring functions and return the mapping with the fewest dummy atoms",
    )

    parser.add_argument(
        "--greedy_k_min_cut",
        default=2,
        type=int,
        help="K min cut of graph to generate, only applicable if using greedy map generation",
    )
    parser.add_argument("--ligands", nargs="+", default=None, help="Name of ligands to consider")
    parser.add_argument(
        "--enable_charge_hops",
        action="store_true",
        help="Build graphs that allow for charge hops, else will generate separated graphs. TMD does not currently support charge hopping",
    )
    for arg, val in DEFAULT_ATOM_MAPPING_KWARGS.items():
        # No reason to provide an initial mapping
        if arg == "initial_mapping":
            continue
        help_str = f"Value for Atom Mapping argument {arg}"
        if isinstance(val, bool):
            parser.add_argument(f"--atom_map_{arg}", type=int, choices=[0, 1], default=1 if val else 0, help=help_str)
        elif isinstance(val, (int, float)):
            parser.add_argument(f"--atom_map_{arg}", type=type(val), default=val, help=help_str)
        else:
            raise ValueError(f"Unknown type for {arg}: {type(val)}")

    args = parser.parse_args()

    np.random.seed(2025)

    assert args.output_path.endswith(".json")

    ligand_path = Path(args.ligands_sdf).expanduser()

    mols = read_sdf(ligand_path)

    atom_mapping_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    atom_map_arg_prefix = "atom_map_"
    for key, val in vars(args).items():
        if not key.startswith(atom_map_arg_prefix):
            continue
        stripped_key = key[len(atom_map_arg_prefix) :]
        initial_type = type(atom_mapping_kwargs[stripped_key])
        atom_mapping_kwargs[stripped_key] = initial_type(val)

    if not args.enable_charge_hops:
        mols_by_charge = defaultdict(list)
        for mol in mols:
            mols_by_charge[Chem.GetFormalCharge(mol)].append(mol)
    else:
        # Force all of the ligands into a single charge object
        mols_by_charge = {0: mols}
    multiple_maps = len(mols_by_charge) > 1
    if multiple_maps:
        print(
            "Ligands contain provide contain different charges, generating multiple maps. Add --enable_charge_hops to force compounds into a single graph"
        )
    mols_by_name = {get_mol_name(m): m for m in mols}
    for charge, mol_subset in mols_by_charge.items():
        if args.ligands is not None and len(args.ligands):
            mol_subset = [mol for mol in mol_subset if get_mol_name(mol) in args.ligands]
        if len(mol_subset) <= 1:
            raise RuntimeError(f"Must provide at least 2 molecules to build graph, got {len(mol_subset)} mols")
        if args.mode == GREEDY:
            if args.greedy_scoring == BEST:
                scoring_methods = [JACCARD, DUMMY_ATOMS]
            else:
                scoring_methods = [args.greedy_scoring]
            nx_graph = build_greedy_graph(
                mol_subset, scoring_methods, k_min_cut=args.greedy_k_min_cut, atom_mapping_kwargs=atom_mapping_kwargs
            )
        else:
            assert args.hub_cmpd is not None
            nx_graph = build_star_graph(args.hub_cmpd, mol_subset, atom_mapping_kwargs=atom_mapping_kwargs)

        if args.refine_cutoff is not None:
            nx_graph = refine_atom_mapping(nx_graph, args.refine_cutoff, atom_mapping_kwargs=atom_mapping_kwargs)

        json_output = []
        # Sort the edges to ensure determinism
        for a, b, data in sorted(nx_graph.edges(data=True), key=lambda x: f"{x[0]}_{x[1]}"):
            edge = {"mol_a": a, "mol_b": b}
            if CORE_FIELD in data:
                edge[CORE_FIELD] = data[CORE_FIELD].tolist()
            json_output.append(edge)
        output_path = Path(args.output_path).expanduser()
        if multiple_maps:
            output_path = output_path.parent / f"{output_path.stem}_charge_{charge:d}{''.join(output_path.suffixes)}"
        print(f"Generated {args.mode} map with {len(json_output)} edges, writing to {output_path!s}")
        with open(output_path, "w") as ofs:
            json.dump(json_output, ofs, indent=1)
        dummy_atoms = []
        for a, b, data in nx_graph.edges(data=True):
            if CORE_FIELD in data:
                dummy_atoms.append(get_num_dummy_atoms(mols_by_name[a], mols_by_name[b], data[CORE_FIELD]))
                # Make this optional later
                # with open(f"atom_mapping_{a}_{b}.svg", "w") as ofs:
                #     ofs.write(plot_atom_mapping_grid(mols_by_name[a], mols_by_name[b], data[CORE_FIELD]))
        assert len(dummy_atoms) > 0
        dummy_atoms = [
            get_num_dummy_atoms(mols_by_name[a], mols_by_name[b], data[CORE_FIELD])
            for a, b, data in nx_graph.edges(data=True)
            if CORE_FIELD in data
        ]
        print("Graph Summary")
        print("-" * 20)
        print("Total Dummy Atoms", sum(dummy_atoms))
        print("Mean Dummy Atoms", np.round(np.mean(dummy_atoms), 2))
        print("Median Dummy Atoms", np.round(np.median(dummy_atoms), 2))
        print("Min Dummy Atoms", np.min(dummy_atoms))
        print("Max Dummy Atoms", np.max(dummy_atoms))
        print("Network Diameter", nx.diameter(nx_graph.to_undirected(as_view=True)))
        print("Node Connectivity", nx.node_connectivity(nx_graph.to_undirected(as_view=True)))


if __name__ == "__main__":
    main()
