import pytest

from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.ff import Forcefield
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology

# Plotting code should not depend on having a GPU
pytestmark = [pytest.mark.nogpu]


from tmd.fe.rest.plots import plot_rest_region


def test_plot_rest_region():
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

    st = SingleTopologyREST(mol_a, mol_b, core, ff, max_temperature_scale=3.0)

    svg = plot_rest_region(st)
    with open("rest_region.svg", "w") as ofs:
        ofs.write(svg)
