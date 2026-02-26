# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025, Forrest York
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

import copy
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe import atom_mapping
from tmd.fe.mcgregor import MaxVisitsWarning, NoMappingError
from tmd.fe.utils import (
    get_mol_name,
    get_romol_conf,
    plot_atom_mapping_grid,
    read_sdf,
    set_mol_name,
    set_romol_conf,
)

pytestmark = [pytest.mark.nocuda]

datasets = {
    "hif2a": "tmd/testsystems/fep_benchmark/hif2a/ligands.sdf",
    "eg5": "tmd/testsystems/fep_benchmark/eg5/ligands.sdf",
}


@pytest.fixture(scope="module")
def hif2a_ligands():
    return read_sdf(datasets["hif2a"])


def test_connected_core_with_large_numbers_of_cores():
    """The following tests that for two mols that have a large number of matching
    cores found prior to filtering out the molecules with disconnected cores and incomplete rings.

    Previously this pair could take about 45 minutes to generate a mapping, largely due to removing
    incomplete ring cores."""

    mol_a = Chem.MolFromMolBlock(
        """CHEMBL3645089
     RDKit          3D

 69 73  0  0  1  0  0  0  0  0999 V2000
   16.0520  -30.0883  -19.2730 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.8905  -27.8674  -19.9786 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.7990  -22.7096  -21.4994 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.1283  -24.6386  -20.8666 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.7580  -23.1584  -20.0406 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.3894  -22.0537  -21.7426 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.8126  -25.4606  -21.9537 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.5455  -26.7156  -19.6263 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.6125  -30.4066  -18.7837 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.4147  -22.6157  -22.1654 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.7818  -21.6172  -21.7628 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.4859  -25.7846  -19.8796 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.2735  -24.7054  -21.2276 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.1100  -20.7065  -20.4639 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.9745  -21.6761  -19.2968 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.8262  -23.6987  -18.3755 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2294  -24.0842  -18.9693 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.4341  -20.4540  -15.1211 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1173  -22.1766  -14.9112 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8225  -20.9925  -14.7205 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8100  -22.5875  -17.4137 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3214  -21.3352  -16.2847 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.7249  -20.8898  -17.9152 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1355  -22.0392  -19.2748 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.0251  -20.6007  -19.6181 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.7679  -25.7907  -17.7633 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.9836  -24.4749  -18.8632 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.4388  -23.7908  -16.4012 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.4874  -27.7302  -19.4019 H   0  0  0  0  0  0  0  0  0  0  0  0
   15.3585  -29.2668  -19.3889 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.8272  -28.0068  -19.8132 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.2976  -23.5767  -21.0948 C   0  0  0  0  0  0  0  0  0  0  0  0
   10.0630  -24.6960  -20.7200 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.8994  -23.6879  -20.4212 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.1253  -23.0424  -21.3981 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.6660  -24.9847  -21.4947 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0343  -25.8428  -20.0356 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.0264  -29.4807  -19.1627 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.9464  -26.9716  -20.0563 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.4093  -25.6407  -20.4989 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.8959  -23.5879  -20.9521 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.2602  -24.7322  -20.4223 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.4335  -25.8445  -20.1881 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.5397  -24.9781  -19.9502 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.0064  -23.7035  -21.9199 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.5492  -27.2205  -19.8930 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.4503  -27.3513  -19.7295 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.4202  -21.3519  -17.6391 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.0982  -22.3619  -21.3534 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.7555  -24.7953  -20.2531 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3340  -21.7347  -20.1776 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2167  -23.7404  -19.2804 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.7049  -21.2462  -15.3040 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0041  -21.5603  -17.1012 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.1114  -21.5772  -19.1419 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.2412  -24.9640  -18.2326 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.6782  -24.0232  -17.1438 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.4498  -26.4940  -20.0505 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.8127  -28.5638  -19.3450 N   0  0  0  0  0  0  0  0  0  0  0  0
   13.1579  -28.4895  -19.4464 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.0646  -22.4207  -19.9054 N   0  0  1  0  0  0  0  0  0  0  0  0
   10.0919  -27.0171  -19.7912 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.4139  -21.3934  -16.7341 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.5801  -21.1166  -17.3007 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.2382  -25.6368  -18.9689 O   0  0  0  0  0  0  0  0  0  0  0  0
   17.2582  -22.8587  -17.6347 F   0  0  0  0  0  0  0  0  0  0  0  0
   16.6960  -24.5926  -16.4580 F   0  0  0  0  0  0  0  0  0  0  0  0
   14.0385  -22.9925  -23.1208 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    4.6384  -22.5815  -20.7956 H   0  0  0  0  0  0  0  0  0  0  0  0
  1 30  1  0
  2 31  1  0
  3 32  1  0
  4 33  1  0
  5 34  1  0
  6 35  1  0
  7 36  1  0
  8 37  1  0
  9 38  1  0
 10 49  1  0
 11 49  1  0
 12 50  1  0
 13 50  1  0
 14 51  1  0
 15 51  1  0
 16 52  1  0
 17 52  1  0
 18 53  1  0
 19 53  1  0
 20 53  1  0
 21 54  1  0
 22 54  1  0
 23 54  1  0
 24 55  1  0
 25 55  1  0
 26 56  1  0
 27 56  1  0
 28 57  1  0
 29 62  1  0
 30 31  1  0
 30 38  2  0
 31 39  2  0
 32 33  2  0
 32 41  1  0
 33 43  1  0
 34 35  2  0
 34 44  1  0
 35 45  1  0
 36 40  1  0
 36 45  2  0
 37 42  1  0
 37 43  2  0
 38 60  1  0
 39 40  1  0
 39 46  1  0
 40 44  2  0
 41 42  2  0
 41 49  1  0
 42 50  1  0
 43 62  1  0
 44 65  1  0
 45 68  1  0
 46 58  2  0
 46 60  1  0
 47 58  1  0
 47 59  2  0
 47 62  1  0
 48 55  1  0
 48 63  1  0
 48 64  2  0
 49 51  1  0
 50 52  1  0
 51 61  1  0
 52 61  1  0
 53 63  1  0
 54 63  1  0
 55 61  1  0
 56 57  1  0
 56 65  1  0
 57 66  1  0
 57 67  1  0
 59 60  1  0
 61 69  1  6
M  CHG  1  61   1
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """CHEMBL3642347
     RDKit          3D

 71 75  0  0  1  0  0  0  0  0999 V2000
   16.0526  -22.0508  -16.7423 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.0564  -30.0836  -19.2752 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.0658  -24.3724  -17.5803 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.8973  -27.8858  -20.0025 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.5163  -26.5291  -19.6315 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.2546  -24.5771  -20.6469 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.3536  -24.4285  -20.0170 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.0923  -22.5184  -21.0223 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.7996  -20.2033  -18.2796 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.6181  -30.4092  -18.7935 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1238  -21.2003  -15.9837 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1146  -22.1989  -14.9229 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.6286  -24.0515  -16.2947 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.6481  -22.9968  -17.3069 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.2419  -20.2229  -16.4566 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.0956  -21.7403  -16.1166 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5668  -23.7121  -17.5780 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.3788  -23.4912  -18.8670 H   0  0  0  0  0  0  0  0  0  0  0  0
   19.0642  -24.0591  -21.0501 H   0  0  0  0  0  0  0  0  0  0  0  0
   18.7345  -24.1070  -22.7812 H   0  0  0  0  0  0  0  0  0  0  0  0
   19.5069  -22.6908  -22.0492 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.1404  -19.7412  -23.4521 H   0  0  0  0  0  0  0  0  0  0  0  0
   15.8464  -20.2097  -21.7749 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.5196  -19.9016  -22.3173 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.0679  -25.8484  -19.2429 H   0  0  0  0  0  0  0  0  0  0  0  0
   17.3759  -25.4132  -20.8886 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.8064  -20.5341  -18.4662 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.1380  -22.2042  -18.2292 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5751  -22.5140  -20.4533 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.9193  -20.8342  -20.6426 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.4868  -27.7026  -19.4198 H   0  0  0  0  0  0  0  0  0  0  0  0
   14.6444  -25.0100  -20.4847 H   0  0  0  0  0  0  0  0  0  0  0  0
   16.3541  -22.2550  -17.7568 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.3650  -29.2637  -19.3907 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.3563  -23.5680  -18.2307 C   0  0  0  0  0  0  0  0  0  0  0  0
   15.8379  -28.0067  -19.8080 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0911  -25.6611  -19.9310 C   0  0  0  0  0  0  0  0  0  0  0  0
   10.1933  -24.5937  -20.4868 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.4207  -24.4316  -20.1442 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.5224  -23.3810  -20.7194 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.7529  -21.2246  -18.6168 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.0351  -29.4815  -19.1646 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.7021  -23.8106  -19.5740 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.4921  -25.7616  -20.0849 C   0  0  0  0  0  0  0  0  0  0  0  0
   14.9547  -26.9862  -20.0659 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.1367  -23.2746  -20.5310 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.5594  -27.2360  -19.9195 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.0626  -22.7019  -20.3843 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.4624  -27.3574  -19.7574 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.9480  -21.9136  -15.9600 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6292  -23.1133  -16.8463 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2047  -21.3058  -16.5796 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.7262  -23.0987  -17.9101 C   0  0  0  0  0  0  0  0  0  0  0  0
   18.7475  -23.4579  -21.9048 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.5435  -20.2936  -22.6090 C   0  0  0  0  0  0  0  0  0  0  0  0
   16.6873  -25.2439  -20.0638 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.4066  -21.5307  -18.6553 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3014  -21.7503  -20.1814 C   0  0  0  0  0  0  0  0  0  0  0  0
   17.1107  -21.4450  -19.8948 N   0  0  0  0  0  0  0  0  0  0  0  0
   12.4698  -26.5158  -20.1118 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.8204  -28.5689  -19.3595 N   0  0  0  0  0  0  0  0  0  0  0  0
   13.1639  -28.4924  -19.4504 N   0  0  0  0  0  0  0  0  0  0  0  0
    5.1183  -21.6934  -17.9837 N   0  0  0  0  0  0  0  0  0  0  0  0
   10.1041  -26.9997  -19.8054 N   0  0  0  0  0  0  0  0  0  0  0  0
   15.3732  -25.7050  -20.4619 N   0  0  0  0  0  0  0  0  0  0  0  0
   17.4375  -22.8381  -21.7349 N   0  0  0  0  0  0  0  0  0  0  0  0
   17.6666  -22.1414  -24.2073 O   0  0  0  0  0  0  0  0  0  0  0  0
   15.3444  -22.6103  -23.2295 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.5840  -22.0380  -20.7412 O   0  0  0  0  0  0  0  0  0  0  0  0
   16.6964  -22.0504  -23.1015 S   0  0  0  0  0  0  0  0  0  0  0  0
    4.4131  -21.1592  -18.4499 H   0  0  0  0  0  0  0  0  0  0  0  0
  1 33  1  0
  2 34  1  0
  3 35  1  0
  4 36  1  0
  5 37  1  0
  6 38  1  0
  7 39  1  0
  8 40  1  0
  9 41  1  0
 10 42  1  0
 11 50  1  0
 12 50  1  0
 13 51  1  0
 14 51  1  0
 15 52  1  0
 16 52  1  0
 17 53  1  0
 18 53  1  0
 19 54  1  0
 20 54  1  0
 21 54  1  0
 22 55  1  0
 23 55  1  0
 24 55  1  0
 25 56  1  0
 26 56  1  0
 27 57  1  0
 28 57  1  0
 29 58  1  0
 30 58  1  0
 31 64  1  0
 32 65  1  0
 33 35  2  0
 33 41  1  0
 34 36  1  0
 34 42  2  0
 35 43  1  0
 36 45  2  0
 37 39  2  0
 37 44  1  0
 38 40  1  0
 38 44  2  0
 39 46  1  0
 40 46  2  0
 41 59  2  0
 42 62  1  0
 43 48  2  0
 43 56  1  0
 44 64  1  0
 45 47  1  0
 45 65  1  0
 46 69  1  0
 47 60  2  0
 47 62  1  0
 48 59  1  0
 48 66  1  0
 49 60  1  0
 49 61  2  0
 49 64  1  0
 50 51  1  0
 50 52  1  0
 51 53  1  0
 52 63  1  0
 53 63  1  0
 54 66  1  0
 55 70  1  0
 56 65  1  0
 57 58  1  0
 57 63  1  0
 58 69  1  0
 61 62  1  0
 66 70  1  0
 67 70  2  0
 68 70  2  0
 63 71  1  0
M  CHG  1  63   1
M  ENDG
$$$$""",
        removeHs=False,
    )

    start = time.time()
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e6,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )
    assert len(all_cores) > 0

    # If this takes more then a minute, something is off.
    assert time.time() - start < 60.0


# hif2a is easy
# eg5 is challenging
# notable outliers for eg5:
# CHEMBL1077227 -> CHEMBL1086410 has 81 cores of size 56
# CHEMBL1077227 -> CHEMBL1083836 has 18 cores of size 48
# CHEMBL1086410 -> CHEMBL1083836 has 15 cores of size 52
# CHEMBL1086410 -> CHEMBL1084935 has 27 cores of size 60
@pytest.mark.parametrize("dataset", datasets.items())
def test_all_pairs(dataset):
    key, filepath = dataset
    mols = read_sdf(filepath)
    cores_by_pair = dict()
    print("MOLS:", len(mols))
    for idx, mol_a in enumerate(mols):
        print(" testing", idx)
        mol_a_name = get_mol_name(mol_a)
        for idb, mol_b in enumerate(mols[idx + 1 :]):
            mol_b_name = get_mol_name(mol_b)
            # print("Processing", mol_a_name, "->", mol_b_name)
            start_time = time.time()
            all_cores, diagnostics = atom_mapping.get_cores_and_diagnostics(
                mol_a,
                mol_b,
                ring_cutoff=0.12,
                chain_cutoff=0.2,
                max_visits=1e7,  # 10 million max nodes to visit
                max_connected_components=1,
                min_connected_component_size=1,
                max_cores=1000,
                enforce_core_core=True,
                ring_matches_ring_only=False,
                heavy_matches_heavy_only=True,
                enforce_chiral=True,
                disallow_planar_torsion_flips=False,
                min_threshold=0,
                initial_mapping=None,
            )
            end_time = time.time()

            # # useful for visualization
            # for core_idx, core in enumerate(all_cores[:1]):
            #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
            #     with open(
            #         f"atom_mapping_{get_mol_name(mol_a)}_to_{get_mol_name(mol_b)}_core_{core_idx}.svg", "w"
            #     ) as fh:
            #         fh.write(res)

            # note that this is probably the bottleneck for hif2a
            for core in all_cores:
                # ensure at least half the atoms are mapped
                assert len(core) >= mol_a.GetNumAtoms() // 2

            print(
                f"{mol_a.GetProp('_Name')} -> {mol_b.GetProp('_Name')} has {len(all_cores)} cores of size {len(all_cores[0])} | total nodes visited: {diagnostics.total_nodes_visited} | wall clock time: {end_time - start_time:.3f}"
                )

            # convert arrays to lists to simplify comparison with reference
            cores = [core.tolist() for core in all_cores]
            cores_by_pair[mol_a_name, mol_b_name] = cores

    ref_pickle_path = Path(f"tests/data/all_pairs_atom_mappings_{key}.pkl")

    # Uncomment to update test reference
    # with Path(ref_pickle_path).open("wb") as fp:
    #     ref_cores_by_pair = pickle.dump(cores_by_pair, fp)

    with Path(ref_pickle_path).open("rb") as fp:
        ref_cores_by_pair = pickle.load(fp)

    for i, (new, ref) in enumerate(zip(cores_by_pair, ref_cores_by_pair)):
        assert new == ref
    #assert cores_by_pair == ref_cores_by_pair


def get_mol_by_name(mols, name):
    for m in mols:
        if get_mol_name(m) == name:
            return m

    assert 0, "Mol not found"


def tuples_to_set(arr):
    res = set()
    for a, b in arr:
        key = (a, b)
        assert key not in res
        res.add(key)
    return res


def assert_cores_are_equal(core_a, core_b):
    core_set_a = tuples_to_set(core_a)
    core_set_b = tuples_to_set(core_b)
    assert core_set_a == core_set_b


def get_all_cores_fzset(all_cores):
    all_cores_fzset = set()
    for core in all_cores:
        all_cores_fzset.add(frozenset(tuples_to_set(core)))
    return all_cores_fzset


def assert_core_sets_are_equal(core_set_a, core_set_b):
    fza = get_all_cores_fzset(core_set_a)
    fzb = get_all_cores_fzset(core_set_b)
    assert fza == fzb


# spot check
def test_linker_map():
    # test that we can map a linker size change when max_connected_components=None, and enforce_core_core=False
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 10 11  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  5  7  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )
    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2219 11232201352D

 11 12  0  0  0  0            999 V2000
  -12.2008    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4558    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2808    0.0842    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5357    0.8688    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8683    1.3538    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1785    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8460    2.8484    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5134    3.3333    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2585    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.4335    4.1179    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.2416    2.1140    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  6  7  1  0  0  0  0
  6 10  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  9 10  1  0  0  0  0
  7 11  1  0  0  0  0
 11  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        max_connected_components=None,
        min_connected_component_size=1,
        max_cores=1000000,
        enforce_core_core=False,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
        # NOTE: example missing explicit Hs, chiral detection will fail
        enforce_chiral=False,
    )

    assert len(all_cores) == 1

    assert_cores_are_equal(
        [[6, 6], [4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]], all_cores[0]
    )

    # now set max_connected_components=1 and enforce_core_core to True
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1000000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    # 2 possible matches, returned core ordering is fully determined
    # note that we return the larger of the two disconnected components here
    # (the 5 membered ring)
    assert len(all_cores) == 2

    expected_sets = ([[6, 6], [9, 9], [8, 8], [7, 7], [5, 5]], [[4, 4], [3, 3], [2, 2], [1, 1], [0, 0]])

    assert_core_sets_are_equal(expected_sets, all_cores)

    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.05,
        chain_cutoff=0.05,
        max_visits=1e7,  # 10 million max nodes to visit
        max_connected_components=None,
        min_connected_component_size=1,
        max_cores=1000000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    # 2 possible matches, if we do not require max_connected_components=1 but do
    # require core_core, we have a 9-atom disconnected map, one is a 5-membered ring
    # the other is 4-membered chain. There's 2 allowed maps due to the 2 fold symmetry.
    assert len(all_cores) == 2

    expected_sets = (
        [[6, 6], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
        [[4, 4], [9, 9], [8, 8], [7, 7], [5, 5], [3, 3], [2, 2], [1, 1], [0, 0]],
    )

    assert_core_sets_are_equal(expected_sets, all_cores)


def get_cyclohexanes_different_confs():
    """Two cyclohexane structures that differ enough in conformations to map poorly by MCS with threshold of 2.0"""
    mol_a = Chem.MolFromMolBlock(
        """
 cyclo_1

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7780    1.1695    0.1292 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3871   -0.1008    0.2959 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6896    1.3214   -0.2192 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5088    0.0613    0.0503 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7859   -1.2096   -0.4242 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6085   -1.3920    0.2133 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1105    2.1590    0.3356 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7549    1.5841   -1.2762 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6874   -0.0047    1.1175 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4858    0.1560   -0.4244 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6117   -1.0357   -1.4891 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1610   -2.0015   -0.5036 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.5422   -1.8809    1.1852 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4054   -2.0928   -0.2686 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3677    1.7499   -0.5802 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9940    1.7789    1.0067 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9567   -0.0955    1.2253 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.2449   -0.1670   -0.3734 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 15  1  0
  1 16  1  0
  2  6  1  0
  2 17  1  0
  2 18  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 11  1  0
  5 14  1  0
  6 12  1  0
  6 13  1  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
 cyclo_2

 18 18  0  0  1  0  0  0  0  0999 V2000
    0.7953    1.1614    0.0469 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3031   -0.0613    0.5362 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6118    1.1962   -0.5144 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9934   -0.1785   -1.1042 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6439   -1.3144   -0.1494 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4262   -1.2251    0.6719 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2949    1.4641    0.2937 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6968    1.9715   -1.2775 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0662   -0.1837   -1.3042 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4970   -0.3613   -2.0575 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6428   -1.9811    1.4121 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2637   -2.1987   -0.1345 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4850    1.5611   -0.6965 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8877    1.9212    0.8230 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8010    0.1189    1.4889 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1753   -0.3430   -0.0537 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2711   -0.8618    0.6186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1781   -0.6848    1.4006 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1 13  1  0
  1 14  1  0
  2  6  1  0
  2 15  1  0
  2 16  1  0
  3  4  1  0
  3  7  1  0
  3  8  1  0
  4  5  1  0
  4  9  1  0
  4 10  1  0
  5  6  1  0
  5 12  1  0
  5 17  1  0
  6 11  1  0
  6 18  1  0
M  END
$$$$""",
        removeHs=False,
    )
    return mol_a, mol_b


def test_hif2a_failure():
    # special failure with error message:
    # pred_sgg_a = a_cycles[a] == sg_a_cycles[a], KeyError: 18
    mols = Chem.SDMolSupplier(datasets["hif2a"], removeHs=False)
    mols = [m for m in mols]
    mol_a = get_mol_by_name(mols, "7a")
    mol_b = get_mol_by_name(mols, "224")
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.12,
        chain_cutoff=0.2,
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e6,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    expected_core = np.array(
        [
            [22, 15],
            [19, 12],
            [2, 2],
            [11, 11],
            [9, 9],
            [6, 6],
            [5, 5],
            [4, 4],
            [1, 1],
            [18, 24],
            [13, 22],
            [12, 21],
            [10, 10],
            [8, 8],
            [7, 7],
            [3, 3],
            [24, 20],
            [23, 19],
            [21, 14],
            [20, 13],
            [0, 0],
            [27, 27],
            [35, 33],
            [33, 32],
            [32, 31],
            [31, 30],
            [30, 29],
            [26, 26],
        ]
    )

    assert_cores_are_equal(all_cores[0], expected_core)
    # for core_idx, core in enumerate(all_cores[:1]):
    #     res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
    #     with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
    #         fh.write(res)


def test_cyclohexane_stereo():
    # test that cyclohexane in two different conformations has a core alignment that is stereo correct. Note that this needs a
    # larger than typical cutoff.
    mol_a, mol_b = get_cyclohexanes_different_confs()
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        ring_cutoff=0.15,
        chain_cutoff=0.30,
        max_visits=1e6,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=100000,
        enforce_core_core=True,
        ring_matches_ring_only=True,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    for core_idx, core in enumerate(all_cores[:1]):
        res = plot_atom_mapping_grid(mol_a, mol_b, core, num_rotations=5)
        with open(f"atom_mapping_0_to_1_core_{core_idx}.svg", "w") as fh:
            fh.write(res)

    # 1-indexed
    # With chiral enforcement and swap-based repair, all H pairs are preserved.
    # At centers where the Hungarian distance-based assignment conflicts with
    # chirality, the H assignment is swapped to the conflict-free permutation.
    expected_core = np.array(
        [
            [1, 1],  # C
            [2, 2],  # C
            [3, 3],  # C
            [4, 4],  # C
            [5, 5],  # C
            [6, 6],  # C
            [7, 7],  # C3H
            [8, 8],  # C3H
            [9, 9],  # C4H
            [10, 10],  # C4H
            [11, 12],  # C5H (swapped)
            [12, 11],  # C5H (swapped)
            [13, 18],  # C6H
            [14, 17],  # C6H (swapped)
            [15, 13],  # C1H
            [16, 14],  # C1H
            [17, 15],  # C2H
            [18, 16],  # C2H
        ]
    )

    # 0-indexed
    expected_core -= 1

    all_cores_fzset = get_all_cores_fzset(all_cores)
    assert tuples_to_set(expected_core) in all_cores_fzset

    assert len(all_cores) == 1


def test_chiral_atom_map():
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C"))

    AllChem.EmbedMolecule(mol_a, randomSeed=0)
    AllChem.EmbedMolecule(mol_b, randomSeed=0)

    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=np.inf,
        chain_cutoff=np.inf,
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e6,
        enforce_core_core=True,
        disallow_planar_torsion_flips=False,
        ring_matches_ring_only=True,
        heavy_matches_heavy_only=True,
        min_threshold=0,
        initial_mapping=None,
    )

    # With H-stripped MCS + Hungarian H augmentation, methane maps C->C then
    # optimally assigns all 4 Hs, producing exactly 1 core of size 5.
    chiral_aware_cores = get_cores(mol_a, mol_b, enforce_chiral=True)
    chiral_oblivious_cores = get_cores(mol_a, mol_b, enforce_chiral=False)

    # Both produce a single augmented core since Hs are assigned deterministically
    assert len(chiral_aware_cores) == 1
    assert len(chiral_oblivious_cores) == 1

    for key, val in chiral_aware_cores[0]:
        assert key == val, "expected first core to be identity map"
    assert len(chiral_aware_cores[0]) == 5


@pytest.mark.parametrize(
    "ring_matches_ring_only",
    [
        pytest.param(False),
        pytest.param(True),
    ],
)
def test_ring_matches_ring_only(ring_matches_ring_only):
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C(c1ccc1)"))
    AllChem.EmbedMolecule(mol_a, randomSeed=3)

    mol_b = Chem.AddHs(Chem.MolFromSmiles("C(c1ccccc1)"))
    AllChem.EmbedMolecule(mol_b, randomSeed=3)

    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=0.15,
        chain_cutoff=0.2,
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e6,
        enforce_core_core=False,
        enforce_chiral=False,
        disallow_planar_torsion_flips=False,
        heavy_matches_heavy_only=True,
        min_threshold=0,
        initial_mapping=None,
    )

    cores = get_cores(mol_a, mol_b, ring_matches_ring_only=ring_matches_ring_only)

    assert cores

    # should not map ring atoms to non-ring atoms and vice-versa
    for core in cores:
        for idx_a, idx_b in core:
            assert mol_a.GetAtomWithIdx(int(idx_a)).IsInRing() == mol_b.GetAtomWithIdx(int(idx_b)).IsInRing()

    # should map all ring atoms in mol_a
    assert all(len([() for idx_a in core[:, 0] if mol_a.GetAtomWithIdx(int(idx_a)).IsInRing()]) == 4 for core in cores)


def test_max_visits_error():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_connected_components=None,
        min_connected_component_size=1,
        max_cores=1000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )
    cores = get_cores(mol_a, mol_b, max_visits=10000)
    assert len(cores) > 0

    with pytest.raises(
        NoMappingError, match="Exceeded max number of visits/cores - no valid cores could be found: 1 nodes visited."
    ):
        get_cores(mol_a, mol_b, max_visits=1)


def test_empty_predicate_map_error():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    # Impossible to map the compounds if they are not aligned
    set_romol_conf(mol_b, get_romol_conf(mol_b) + 1000.0)

    with pytest.raises(NoMappingError, match="No possible mapping given the predicate matrix"):
        atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)


def test_max_cores_warning():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_connected_components=None,
        min_connected_component_size=1,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        max_visits=1e7,
        initial_mapping=None,
    )
    # Warning is triggered by reaching the max visits, but not the max_cores
    # With H-stripped MCS, cyclohexane only has 6 heavy atoms and needs ~12 visits,
    # so we use a lower max_visits to trigger the warning
    with pytest.warns(MaxVisitsWarning, match="Inexhaustive search: reached max number of visits"):
        all_cores = get_cores(mol_a, mol_b, max_cores=100, max_visits=10)
        assert len(all_cores) == 1


def test_min_threshold():
    mol_a, mol_b = get_cyclohexanes_different_confs()
    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_connected_components=None,
        min_connected_component_size=1,
        max_cores=1000,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=mol_a.GetNumAtoms(),
        initial_mapping=None,
    )

    with pytest.raises(NoMappingError, match="Unable to find mapping with at least 18 edges"):
        get_cores(mol_a, mol_b, max_visits=10000)


def test_get_cores_and_diagnostics():
    mols = read_sdf(datasets["hif2a"])
    n_pairs = 30
    random_pair_idxs = np.random.default_rng(2024).choice(len(mols), size=(n_pairs, 2))

    for i_a, i_b in random_pair_idxs:
        mol_a = mols[i_a]
        mol_b = mols[i_b]
        all_cores, diagnostics = atom_mapping.get_cores_and_diagnostics(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)

        assert len(all_cores) > 0
        assert all(len(core) == len(all_cores[0]) for core in all_cores)

        # diagnostics.core_size is the heavy-atom MCS core size;
        # augmented cores include H mappings and are thus >= the heavy core
        assert len(all_cores[0]) >= diagnostics.core_size
        assert diagnostics.num_cores >= len(all_cores)

        # must visit at least one node per atom pair in core
        assert diagnostics.total_nodes_visited >= diagnostics.core_size

        # must visit at least one leaf node per core
        assert diagnostics.num_cores <= diagnostics.total_leaves_visited

        assert diagnostics.total_leaves_visited <= diagnostics.total_nodes_visited


def polyphenylene_smiles(n):
    def go(k):
        return f"(c{k}ccc{go(k - 1)}cc{k})" if k > 0 else ""

    return go(n)[1:-1]


def make_polyphenylene(n, dihedral_deg):
    """Make a chain of n benzene rings with each ring rotated `dihedral_deg` degrees with respect to the previous ring"""
    mol = Chem.MolFromSmiles(polyphenylene_smiles(n))
    set_mol_name(mol, f"{n}_{dihedral_deg}")
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    for k in range(n - 1):  # n - 1 inter-ring bonds to rotate
        i = 2 + 4 * k
        AllChem.SetDihedralDeg(mol.GetConformer(0), i, i + 1, i + 2, i + 3, dihedral_deg)
    return mol


def get_core(mol_a, mol_b, **kwargs):
    cores = atom_mapping.get_cores(mol_a, mol_b, **{**DEFAULT_ATOM_MAPPING_KWARGS, **kwargs})
    return cores[0]


def test_max_connected_components():
    """Test mapping a pair of 5-phenyl mols; mol_a planar and mol_b with even rings rotated 90 degrees.

    For this example, setting max_connected_components=1 will only map 1 ring, max_connected_components=2 will map 2
    rings, etc.
    """

    mol_a = make_polyphenylene(5, 0.0)
    mol_b = make_polyphenylene(5, 90.0)

    # Align the molecules, otherwise the conformers are different enough to trigger test failures.
    AllChem.AlignMol(mol_a, mol_b)

    with pytest.raises(AssertionError, match="max_connected_components > 0"):
        get_core(mol_a, mol_b, max_connected_components=0)

    assert len(get_core(mol_a, mol_b, max_connected_components=1)) == 6 + 6  # maps 1 ring (6 C, 6 H)
    assert len(get_core(mol_a, mol_b, max_connected_components=2)) == 2 * (6 + 6)  # maps 2 rings

    # maps 3 rings
    core_3 = get_core(mol_a, mol_b, max_connected_components=3)
    assert len(core_3) == 3 * (6 + 6)

    np.testing.assert_array_equal(
        core_3, get_core(mol_a, mol_b, max_connected_components=None)
    )  # n=3 and n=None return same mapping


def test_min_connected_component_size():
    """Test mapping a pair of biphenyl mols; mol_a planar and mol_b with the second ring rotated 90 degrees.

    With the heavy-atoms-first MCS approach, connected component filtering operates on the heavy-atom
    graph. For this example, setting min_connected_component_size >= 2 removes isolated atom pairs in
    the second ring, reducing the core size.
    """
    mol_a = make_polyphenylene(2, 0.0)
    mol_b = make_polyphenylene(2, 90.0)

    # With min_connected_component_size=1, maps one ring plus opposite atoms of second ring
    # 8 heavy atoms + 6 hydrogens = 14
    core_1 = get_core(mol_a, mol_b, max_connected_components=None, min_connected_component_size=1)
    assert len(core_1) == 14

    # Any min_connected_component_size <= 1 is a no-op and returns the same result
    for min_connected_component_size in [-1, 0, 1]:
        np.testing.assert_array_equal(
            get_core(
                mol_a,
                mol_b,
                max_connected_components=None,
                min_connected_component_size=min_connected_component_size,
            ),
            core_1,
        )

    # With min_connected_component_size >= 2, can no longer map isolated atoms of second ring
    # 7 heavy atoms + 5 hydrogens = 12
    core_2 = get_core(mol_a, mol_b, max_connected_components=None, min_connected_component_size=2)
    assert len(core_2) == 12

    for min_connected_component_size in [3, 4, 5, 6, 7]:
        np.testing.assert_array_equal(
            get_core(mol_a, mol_b, min_connected_component_size=min_connected_component_size), core_2
        )

    # core can't be larger than one ring plus anchor; with heavy-first MCS, min_size=8 is too large
    with pytest.raises(NoMappingError):
        _ = get_core(mol_a, mol_b, min_connected_component_size=8)

    # Check that min_connected_component_size works as expected with max_connected_components
    core_1_1 = get_core(mol_a, mol_b, max_connected_components=1, min_connected_component_size=1)
    assert len(core_1_1) != len(core_1)
    np.testing.assert_array_equal(core_1_1, core_2)


def test_initial_mapping(hif2a_ligands):
    # Test that we can generate an equally good mapping if we specify
    # an initial mapping that is in the optimal mapping
    # Note: this adjusts bumps ring_cutoff and chain cutoff both to 0.4 (from 0.12, 0.2)
    mols = hif2a_ligands
    mol_a, mol_b = mols[0], mols[1]
    initial_mapping = np.array(
        [
            [17, 13],
            [16, 12],
            [14, 11],
            [13, 10],
            [12, 9],
            [19, 15],
            [11, 8],
            [10, 7],
            [9, 6],
            [8, 5],
            [7, 4],
            [6, 3],
            [5, 2],
        ]
    )

    # adjust for 1-indexing when reading off the atom-mapping
    initial_mapping = initial_mapping - 1

    get_cores_and_diagnostics = partial(
        atom_mapping.get_cores_and_diagnostics,
        # ring_cutoff=0.12,
        # chain_cutoff=0.2,
        ring_cutoff=0.4,  # bumped up to make the problem harder
        chain_cutoff=0.4,  # bumped up to make the problem harder
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e5,
        enforce_core_core=True,
        ring_matches_ring_only=True,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=True,
        min_threshold=0,
    )

    all_cores_test, diagnostics_test = get_cores_and_diagnostics(mol_a, mol_b, initial_mapping=initial_mapping)
    all_cores_ref, diagnostics_ref = get_cores_and_diagnostics(mol_a, mol_b, initial_mapping=None)

    assert len(all_cores_test[0]) == len(all_cores_ref[0])

    # should be something like "Test visited: 1480 Ref visited: 31796"
    print("Test visited:", diagnostics_test.total_nodes_visited, "Ref visited:", diagnostics_ref.total_nodes_visited)
    assert diagnostics_test.total_nodes_visited < diagnostics_ref.total_nodes_visited


def new_to_old_map_after_removing_hs(mol):
    # usually explicitHs are always placed at the end, but just to be safe
    # we explicitly compute the ordering
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    old_to_new_mapping = {}
    new_to_old_mapping = {}

    # generate old to new mapping
    for atom_idx, atom_num in enumerate(atomic_nums):
        if atom_num != 1:
            old_to_new_mapping[atom_idx] = len(old_to_new_mapping)

    # generate new to old mapping
    for old, new in old_to_new_mapping.items():
        new_to_old_mapping[new] = old

    return new_to_old_mapping


@pytest.mark.parametrize("pair", [(0, 1)])
def test_empty_initial_mapping_returns_identity(pair, hif2a_ligands):
    a, b = pair
    mol_a, mol_b = hif2a_ligands[a], hif2a_ligands[b]

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    cores = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    # Don't change the kwargs here, besides adding an empty core
    kwargs["initial_mapping"] = np.zeros((0, 2))

    cores_empty_initial_map = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    assert len(cores) >= 1
    assert len(cores_empty_initial_map) == len(cores)

    np.testing.assert_equal(cores, cores_empty_initial_map)


@pytest.mark.parametrize("pair", [(0, 1)])
def test_initial_mapping_returns_self_with_same_params(pair, hif2a_ligands):
    """If an initial map is provided for the same parameters, the values are identical"""
    a, b = pair
    mol_a, mol_b = hif2a_ligands[a], hif2a_ligands[b]

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    cores = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    kwargs["initial_mapping"] = cores[0]

    # Since the initial mapping is set and no parameters are changed, should only return a single core
    identity_core = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    assert len(cores) >= 1
    assert len(identity_core) == 1
    # Core ordering does get shuffled, but pairs should be identical
    assert_cores_are_equal(identity_core[0], cores[0])


@pytest.mark.parametrize("pair", [(0, 1)])
def test_initial_mapping_always_a_subset_of_cores(pair, hif2a_ligands):
    a, b = pair
    mol_a, mol_b = hif2a_ligands[a], hif2a_ligands[b]

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    cores = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    initial_map = cores[0]

    kwargs["initial_mapping"] = initial_map
    kwargs["ring_cutoff"] = kwargs["ring_cutoff"] * 2.0
    kwargs["chain_cutoff"] = kwargs["chain_cutoff"] * 2.0

    cores_with_map = atom_mapping.get_cores(mol_a, mol_b, **kwargs)

    initial_pairs = set(tuple(pair) for pair in initial_map)
    for core in cores_with_map:
        new_pairs = set(tuple(pair) for pair in core)
        assert new_pairs.issuperset(initial_pairs)


@pytest.mark.parametrize(
    "param_to_change,new_val,expect_exception",
    [
        ("max_connected_components", None, False),
        ("enforce_core_core", False, False),
        ("enforce_chiral", False, False),
        ("disallow_planar_torsion_flips", False, False),
    ],
)
def test_initial_mapping_ignores_filters(hif2a_ligands, param_to_change, new_val, expect_exception):
    mol_a = hif2a_ligands[0]
    mol_b = hif2a_ligands[1]
    cores = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)

    assert DEFAULT_ATOM_MAPPING_KWARGS[param_to_change] != new_val

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    kwargs[param_to_change] = new_val
    unfiltered_cores = atom_mapping.get_cores(mol_a, mol_b, **kwargs)
    # The core without the filter `param_to_change` is equal or larger
    assert len(cores[0]) <= len(unfiltered_cores[0])

    initial_map_kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    initial_map_kwargs["initial_mapping"] = unfiltered_cores[0]

    # If we remap with this core that is invalid under the mapping conditions, return the original core
    if expect_exception:
        # If there is no connected core that can be made from the disconnected core, expect NoMappingError
        with pytest.raises(NoMappingError):
            atom_mapping.get_cores(mol_a, mol_b, **initial_map_kwargs)
    else:
        new_cores = atom_mapping.get_cores(mol_a, mol_b, **initial_map_kwargs)
        assert len(new_cores) == 1


def test_hybrid_core_generation(hif2a_ligands):
    """
    Verify expectations around the generation of hybrid molecules given initial mappings generated fro molecules
    without hydrogens.

    The expectations are:
    * Cores generated with hydrogens are as large or larger than the hybrid. The reason the core can be smaller is
      if a terminal non-hydrogen atom is mapped to what is a terminal in the hydrogen-less molecule, but with hydrogens
      would be non-terminal.
    * Cores generated with the hybrid approach without hydrogens first and then with hydrogens will be strictly larger.
    """
    mols_with_hs = hif2a_ligands
    mols_without_hs = read_sdf(datasets["hif2a"], removeHs=True)

    n_mols = len(mols_with_hs)

    # with_h_visits = []
    # without_h_visits = []
    # hybrid_visits = []

    # with_h_core_sizes = []
    # without_h_core_sizes = []
    # hybrid_core_sizes = []

    TEST_ATOM_MAPPING_KWARGS = copy.deepcopy(DEFAULT_ATOM_MAPPING_KWARGS)

    # useful for testing larger cutoff settings
    # TEST_ATOM_MAPPING_KWARGS["ring_cutoff"] = 0.4
    # TEST_ATOM_MAPPING_KWARGS["chain_cutoff"] = 0.4

    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            mol_a_with_h, mol_b_with_h = mols_with_hs[i], mols_with_hs[j]
            mol_a_without_h, mol_b_without_h = mols_without_hs[i], mols_without_hs[j]

            cores_h, diagnostics_with_h = atom_mapping.get_cores_and_diagnostics(
                mol_a_with_h, mol_b_with_h, **TEST_ATOM_MAPPING_KWARGS
            )
            core_h = cores_h[0]

            cores_no_h, diagnostics_no_h = atom_mapping.get_cores_and_diagnostics(
                mol_a_without_h, mol_b_without_h, **TEST_ATOM_MAPPING_KWARGS
            )
            core_no_h = cores_no_h[0]

            # hybrid method, use core from without H atom-mapping
            ntom_mol_a = new_to_old_map_after_removing_hs(mol_a_with_h)
            ntom_mol_b = new_to_old_map_after_removing_hs(mol_b_with_h)

            core_a_initial = [ntom_mol_a[x] for x in core_no_h[:, 0]]
            core_b_initial = [ntom_mol_b[x] for x in core_no_h[:, 1]]

            initial_mapping = np.stack([core_a_initial, core_b_initial], axis=1)
            MAPPING_KWARGS_WITH_MAPPING = copy.deepcopy(TEST_ATOM_MAPPING_KWARGS)
            MAPPING_KWARGS_WITH_MAPPING["initial_mapping"] = initial_mapping

            cores_hybrid, diagnostics_hybrid = atom_mapping.get_cores_and_diagnostics(
                mol_a_with_h, mol_b_with_h, **MAPPING_KWARGS_WITH_MAPPING
            )

            core_hybrid = cores_hybrid[0]

            # useful printing for debugging
            # tnv_h = diagnostics_with_h.total_nodes_visited
            # tnv_hybrid = diagnostics_hybrid.total_nodes_visited
            # tnv_no_h = diagnostics_no_h.total_nodes_visited
            # print("TNV: all_hs, no_hs, hybrid", tnv_h, tnv_no_h, tnv_no_h + tnv_hybrid)
            # print("CORE SIZE: all_hs, no_hs, hybrid", len(core_h), len(core_no_h), len(core_hybrid))

            # with_h_visits.append(tnv_h)
            # without_h_visits.append(tnv_no_h)
            # hybrid_visits.append(tnv_no_h + tnv_hybrid)

            # with_h_core_sizes.append(len(core_h))
            # without_h_core_sizes.append(len(core_no_h))
            # hybrid_core_sizes.append(len(core_hybrid))

            assert len(core_no_h) < len(core_h), f"Mol {i} -> {j} failed to produce larger mapping by adding hydrogens"
            assert len(core_no_h) < len(core_hybrid), (
                f"Mol {i} -> {j} failed to produce larger mapping by running hybrid"
            )
            assert len(core_hybrid) <= len(core_h), (
                f"Mol {i} -> {j} failed to produce larger mapping by running with hydrogens than hybrid"
            )

    # useful diagnostics
    # import matplotlib.pyplot as plt
    # plt.subplot(231)
    # plt.title("with Hs visits")
    # plt.hist(with_h_visits, label=f"mean={np.mean(with_h_visits):.2f}", bins=20)
    # plt.xlabel("total visits")
    # plt.legend()

    # # plt.show()

    # plt.subplot(232)
    # plt.title("without Hs visits")
    # plt.hist(without_h_visits, label=f"mean={np.mean(without_h_visits):.2f}", bins=20)
    # plt.xlabel("total visits")
    # plt.legend()
    # # plt.show()

    # plt.subplot(233)
    # plt.title("hybrid visits")
    # plt.hist(hybrid_visits, label=f"mean={np.mean(hybrid_visits):.2f}", bins=20)
    # plt.xlabel("total visits")
    # plt.legend()

    # plt.subplot(234)
    # plt.title("with Hs core sizes")
    # plt.hist(with_h_core_sizes, label=f"mean={np.mean(with_h_core_sizes):.2f}", bins=20)
    # plt.xlabel("core_size")
    # plt.legend()

    # plt.subplot(235)
    # plt.title("without Hs core sizes")
    # plt.hist(without_h_core_sizes, label=f"mean={np.mean(without_h_core_sizes):.2f}", bins=20)
    # plt.xlabel("core_size")
    # plt.legend()

    # plt.subplot(236)
    # plt.title("hybrid core sizes")
    # plt.hist(hybrid_core_sizes, label=f"mean={np.mean(hybrid_core_sizes):.2f}", bins=20)
    # plt.xlabel("core_size")
    # plt.legend()

    # plt.show()


def test_heavy_matches_heavy_only():
    """Tests that only heavy atoms match heavy atoms in the atom mapping.

    Uses MCL1 ligand 49 and 54, where previously the hydrogens on a methyl would be mapped
    to ring atoms.

    A useful feature for applying constraints where the heavy atoms and hydrogens will be have to be demapped
    anyways.
    """
    mol_a = Chem.MolFromMolBlock(
        """49
     RDKit          3D

 41 43  0  0  1  0  0  0  0  0999 V2000
    3.2882  -48.3838    6.5890 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.7977  -49.5315    6.5856 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.4275  -50.6005    6.8061 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3311  -49.6554    6.3479 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5716  -50.8911    6.4074 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5121  -51.9101    6.5989 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8370  -51.6159    6.2843 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1985  -50.3836    5.7498 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2573  -49.3795    5.5112 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9064  -49.5984    5.8824 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3153  -48.8057    5.8681 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4768  -47.3560    5.4229 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4583  -46.3692    6.6107 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9047  -46.2168    7.3093 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8095  -45.5295    6.4458 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1313  -45.3765    6.8153 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.1136  -44.7343    6.0454 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.4404  -44.6590    6.4864 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.4575  -43.9587    5.5974 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.7906  -45.2331    7.7201 C   0  0  0  0  0  0  0  0  0  0  0  0
   -7.4178  -45.2246    8.3109 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -4.8092  -45.8417    8.5021 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4928  -45.9109    8.0501 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2277  -52.8746    6.9949 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6010  -52.3676    6.4443 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2321  -50.1889    5.4980 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7975  -47.9300    4.7462 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -0.2611  -47.0496    4.6791 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4420  -47.2460    4.9196 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7920  -45.3869    6.2700 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2002  -46.6940    7.3397 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7803  -45.6315    8.2225 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2793  -47.2014    7.5987 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.8537  -44.2885    5.0988 H   0  0  0  0  0  0  0  0  0  0  0  0
   -7.4741  -43.9976    5.9852 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.1834  -42.9109    5.4762 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.4643  -44.4144    4.6070 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7333  -46.3832    8.6540 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.0691  -46.2686    9.4578 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7885  -50.8885    6.6426 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.4242  -51.6087    6.9742 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  2  4  1  0
  4 11  2  0
  4 40  1  0
  5  6  2  0
  5 10  1  0
  5 40  1  0
  6  7  1  0
  6 24  1  0
  7  8  2  0
  7 25  1  0
  8  9  1  0
  8 26  1  0
  9 10  2  0
  9 27  1  0
 10 11  1  0
 11 12  1  0
 12 28  1  0
 12 29  1  0
 12 13  1  0
 13 14  1  0
 13 30  1  0
 13 31  1  0
 14 15  1  0
 14 32  1  0
 14 33  1  0
 15 16  1  0
 16 17  2  0
 16 23  1  0
 17 18  1  0
 17 34  1  0
 18 19  1  0
 18 20  2  0
 19 35  1  0
 19 36  1  0
 19 37  1  0
 20 21  1  0
 20 22  1  0
 22 23  2  0
 22 39  1  0
 23 38  1  0
 40 41  1  0
M  CHG  1   3  -1
M  END

$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """54
     RDKit          3D

 44 47  0  0  1  0  0  0  0  0999 V2000
    3.3478  -48.4173    6.5720 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.8388  -49.5557    6.5848 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.4483  -50.6256    6.8416 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3686  -49.6643    6.3411 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5625  -50.8698    6.4071 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5354  -51.8642    6.5694 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8499  -51.5261    6.2479 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1772  -50.2531    5.7893 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2084  -49.2604    5.6205 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8667  -49.5602    5.9220 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3637  -48.8113    5.8496 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4941  -47.3829    5.3462 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4921  -46.3634    6.5011 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8482  -46.2585    7.2540 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8570  -45.6667    6.4484 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1557  -45.4679    6.9012 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.6282  -45.2570    8.2561 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.5614  -45.9239    8.8390 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.3484  -46.0091    8.1742 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2986  -52.8520    6.9308 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.0966  -52.7140    6.4180 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -4.2139  -50.0339    5.5648 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4870  -48.2760    5.2766 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2946  -47.1322    4.6360 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4262  -47.2753    4.7827 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7813  -45.3778    6.1309 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2712  -46.6616    7.2034 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7092  -45.6540    8.1532 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1456  -47.2703    7.5314 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5202  -46.5088    8.6561 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.5008  -44.6692    6.9788 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2669  -44.7884    6.2518 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.2595  -44.2048    4.9425 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.3558  -43.5156    4.4300 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.5180  -43.3796    5.1734 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.6033  -43.9609    6.4324 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4282  -44.2576    4.2709 H   0  0  0  0  0  0  0  0  0  0  0  0
   -5.2899  -43.0835    3.4451 H   0  0  0  0  0  0  0  0  0  0  0  0
   -7.3549  -42.8301    4.7676 H   0  0  0  0  0  0  0  0  0  0  0  0
   -7.5170  -43.8657    7.0046 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.6664  -46.3641    9.8186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.5687  -45.1778    8.7789 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8015  -50.8924    6.6395 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.4200  -51.6277    6.9741 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  2  4  1  0
  4 11  2  0
  4 43  1  0
  5  6  2  0
  5 10  1  0
  5 43  1  0
  6  7  1  0
  6 20  1  0
  7  8  2  0
  7 21  1  0
  8  9  1  0
  8 22  1  0
  9 10  2  0
  9 23  1  0
 10 11  1  0
 11 12  1  0
 12 24  1  0
 12 25  1  0
 12 13  1  0
 13 14  1  0
 13 26  1  0
 13 27  1  0
 14 15  1  0
 14 28  1  0
 14 29  1  0
 15 16  1  0
 16 19  2  0
 16 32  1  0
 17 18  2  0
 17 31  1  0
 17 42  1  0
 18 19  1  0
 18 41  1  0
 19 30  1  0
 31 32  2  0
 31 36  1  0
 32 33  1  0
 33 34  2  0
 33 37  1  0
 34 35  1  0
 34 38  1  0
 35 36  2  0
 35 39  1  0
 36 40  1  0
 43 44  1  0
M  CHG  1   3  -1
M  END""",
        removeHs=False,
    )

    get_cores = partial(
        atom_mapping.get_cores,
        ring_cutoff=0.1,
        chain_cutoff=0.2,
        max_visits=1e7,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=1e6,
        enforce_core_core=True,
        ring_matches_ring_only=False,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    # heavy_matches_heavy_only=False is allowed; just verify it doesn't raise
    get_cores(mol_a, mol_b, heavy_matches_heavy_only=False)

    heavy_matches_heavy_core = get_cores(mol_a, mol_b, heavy_matches_heavy_only=True)[0]

    # Verify the invariant: every mapped pair is H↔H or heavy↔heavy
    for a, b in heavy_matches_heavy_core:
        atom_a = mol_a.GetAtomWithIdx(int(a))
        atom_b = mol_b.GetAtomWithIdx(int(b))

        assert (atom_a.GetAtomicNum() == 1 and atom_b.GetAtomicNum() == 1) or (
            atom_a.GetAtomicNum() > 1 and atom_b.GetAtomicNum() > 1
        )

    # Verify that mapped Hs have their parent heavy atoms also mapped
    core_set_a = set(int(a) for a, _ in heavy_matches_heavy_core)
    core_set_b = set(int(b) for _, b in heavy_matches_heavy_core)
    core_a_to_b = {int(a): int(b) for a, b in heavy_matches_heavy_core}
    for a, b in heavy_matches_heavy_core:
        a, b = int(a), int(b)
        if mol_a.GetAtomWithIdx(a).GetAtomicNum() == 1:
            # Find the heavy-atom parent of this H in mol_a
            parent_a = [nb.GetIdx() for nb in mol_a.GetAtomWithIdx(a).GetNeighbors() if nb.GetAtomicNum() != 1]
            parent_b = [nb.GetIdx() for nb in mol_b.GetAtomWithIdx(b).GetNeighbors() if nb.GetAtomicNum() != 1]
            assert len(parent_a) == 1, "H should have exactly one heavy-atom parent"
            assert len(parent_b) == 1, "H should have exactly one heavy-atom parent"
            assert parent_a[0] in core_set_a, "Parent heavy atom of mapped H must be in core"
            assert parent_b[0] in core_set_b, "Parent heavy atom of mapped H must be in core"
            assert core_a_to_b[parent_a[0]] == parent_b[0], "H parents must be mapped to each other"


def test_chiral_filtering_on_h_stripped_molecules():
    """Verify that chiral filtering on H-stripped molecules (via Chem.RemoveHs) produces
    results consistent with chiral filtering on full-H molecules, for heavy-atom-only cores.

    This validates the planned atom-mapper optimization where the MCS search runs on
    H-stripped molecules with chiral filters applied during the search (not post-hoc).

    Key properties tested:
    1. find_chiral_atoms finds the same chiral centers after RemoveHs (because [X4:1]
       counts total degree including implicit Hs).
    2. ChiralRestrIdxSet built from H-stripped molecules produces fewer but sufficient
       restraint tuples (e.g., C(3,3)=1 instead of C(4,3)=4 for a center with 1 implicit H).
    3. has_chiral_atom_flips on a heavy-atom core using H-stripped chiral sets gives the
       same answer as the full-H chiral sets for all heavy-atom-only mappings.
    4. Planar torsion flip detection works correctly on H-stripped molecules.
    """
    from tmd.fe.chiral_utils import (
        ChiralRestrIdxSet,
        find_chiral_atoms,
        has_chiral_atom_flips,
        setup_find_flipped_planar_torsions,
    )
    from tmd.fe.mcgregor import UNMAPPED, core_to_perm

    # --- Part 1: Chiral center detection is preserved after RemoveHs ---

    # FC(Cl)Br: carbon has 4 neighbors (F, Cl, Br, H) -> X4 with Hs, X4 after RemoveHs (3 explicit + 1 implicit)
    mol_with_h = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)Br"))
    AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
    mol_no_h = Chem.RemoveHs(mol_with_h)

    chiral_with_h = find_chiral_atoms(mol_with_h)
    chiral_no_h = find_chiral_atoms(mol_no_h)

    # The chiral carbon (index 1 in both) should be found in both cases
    assert len(chiral_with_h) > 0, "Should find chiral center with Hs"
    assert len(chiral_no_h) > 0, "Should find chiral center after RemoveHs"
    # The carbon index should be the same (it's heavy atom index 1 in both)
    assert 1 in chiral_with_h
    assert 1 in chiral_no_h

    # --- Part 2: Restraint tuples are fewer but sufficient ---

    conf_with_h = get_romol_conf(mol_with_h)
    conf_no_h = get_romol_conf(mol_no_h)

    chiral_set_with_h = ChiralRestrIdxSet.from_mol(mol_with_h, conf_with_h)
    chiral_set_no_h = ChiralRestrIdxSet.from_mol(mol_no_h, conf_no_h)

    # With Hs: carbon has 4 neighbors -> C(4,3) = 4 restraint tuples
    assert len(chiral_set_with_h.restr_idxs) == 4
    # Without Hs: carbon has 3 neighbors (F, Cl, Br) -> C(3,3) = 1 restraint tuple
    assert len(chiral_set_no_h.restr_idxs) == 1

    # The single H-stripped tuple should reference only heavy atoms
    center, i, j, k = chiral_set_no_h.restr_idxs[0]
    assert center == 1  # the chiral carbon
    assert all(mol_no_h.GetAtomWithIdx(idx).GetAtomicNum() != 1 for idx in [center, i, j, k])

    # --- Part 3: Chiral flip detection agrees for heavy-atom-only cores ---

    # Create two copies of FC(Cl)Br and test identity vs. chirality-flipping maps
    mol_a = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)Br"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("FC(Cl)Br"))
    AllChem.EmbedMolecule(mol_a, randomSeed=42)
    AllChem.EmbedMolecule(mol_b, randomSeed=42)

    mol_a_no_h = Chem.RemoveHs(mol_a)
    mol_b_no_h = Chem.RemoveHs(mol_b)

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    conf_a_no_h = get_romol_conf(mol_a_no_h)
    conf_b_no_h = get_romol_conf(mol_b_no_h)

    chiral_set_a_full = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
    chiral_set_b_full = ChiralRestrIdxSet.from_mol(mol_b, conf_b)
    chiral_set_a_no_h = ChiralRestrIdxSet.from_mol(mol_a_no_h, conf_a_no_h)
    chiral_set_b_no_h = ChiralRestrIdxSet.from_mol(mol_b_no_h, conf_b_no_h)

    n_a_full = mol_a.GetNumAtoms()
    n_a_no_h = mol_a_no_h.GetNumAtoms()

    # Identity core (heavy atoms only): F->F, C->C, Cl->Cl, Br->Br
    identity_core_heavy = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    # Chirality-flipping core: swap F and Cl (indices 0 and 2)
    flipped_core_heavy = np.array([[0, 2], [1, 1], [2, 0], [3, 3]])

    # Test with full-H chiral sets (using heavy-atom core embedded in full-H perm)
    identity_perm_full = core_to_perm(identity_core_heavy, n_a_full)
    flipped_perm_full = core_to_perm(flipped_core_heavy, n_a_full)

    identity_has_flip_full = has_chiral_atom_flips(identity_perm_full, chiral_set_a_full, chiral_set_b_full)
    flipped_has_flip_full = has_chiral_atom_flips(flipped_perm_full, chiral_set_a_full, chiral_set_b_full)

    # Test with H-stripped chiral sets
    identity_perm_no_h = core_to_perm(identity_core_heavy, n_a_no_h)
    flipped_perm_no_h = core_to_perm(flipped_core_heavy, n_a_no_h)

    identity_has_flip_no_h = has_chiral_atom_flips(identity_perm_no_h, chiral_set_a_no_h, chiral_set_b_no_h)
    flipped_has_flip_no_h = has_chiral_atom_flips(flipped_perm_no_h, chiral_set_a_no_h, chiral_set_b_no_h)

    # Both should agree: identity is not a flip, swapping F/Cl is a flip
    assert not identity_has_flip_full, "Identity map should not be a chiral flip (full-H)"
    assert not identity_has_flip_no_h, "Identity map should not be a chiral flip (no-H)"
    assert flipped_has_flip_full, "Swapping F/Cl should be a chiral flip (full-H)"
    assert flipped_has_flip_no_h, "Swapping F/Cl should be a chiral flip (no-H)"

    # --- Part 4: Test with a more complex molecule (ethanol: CC(O)F with chiral C) ---

    mol_a2 = Chem.AddHs(Chem.MolFromSmiles("CC(O)F"))
    mol_b2 = Chem.AddHs(Chem.MolFromSmiles("CC(O)F"))
    AllChem.EmbedMolecule(mol_a2, randomSeed=42)
    AllChem.EmbedMolecule(mol_b2, randomSeed=42)

    mol_a2_no_h = Chem.RemoveHs(mol_a2)
    mol_b2_no_h = Chem.RemoveHs(mol_b2)

    n_a2_full = mol_a2.GetNumAtoms()
    n_a2_no_h = mol_a2_no_h.GetNumAtoms()

    conf_a2 = get_romol_conf(mol_a2)
    conf_b2 = get_romol_conf(mol_b2)
    conf_a2_no_h = get_romol_conf(mol_a2_no_h)
    conf_b2_no_h = get_romol_conf(mol_b2_no_h)

    chiral_a2_full = ChiralRestrIdxSet.from_mol(mol_a2, conf_a2)
    chiral_b2_full = ChiralRestrIdxSet.from_mol(mol_b2, conf_b2)
    chiral_a2_no_h = ChiralRestrIdxSet.from_mol(mol_a2_no_h, conf_a2_no_h)
    chiral_b2_no_h = ChiralRestrIdxSet.from_mol(mol_b2_no_h, conf_b2_no_h)

    # CC(O)F: C0 has 4 neighbors (C1, H, H, H) -> X4; C1 has 4 neighbors (C0, O, F, H) -> X4
    # After RemoveHs: C0 has 1 heavy neighbor -> no C(1,3) restraints; C1 has 3 heavy neighbors -> C(3,3)=1
    chiral_atoms_full = find_chiral_atoms(mol_a2)
    chiral_atoms_no_h = find_chiral_atoms(mol_a2_no_h)
    assert 0 in chiral_atoms_full  # methyl C
    assert 1 in chiral_atoms_full  # chiral C
    assert 0 in chiral_atoms_no_h  # still X4 (1 explicit + 3 implicit)
    assert 1 in chiral_atoms_no_h  # still X4 (3 explicit + 1 implicit)

    # But only C1 (with 3 heavy neighbors) gets restraint tuples after H removal
    # C0 only has 1 heavy neighbor -> C(1,3) = 0 tuples
    assert len(chiral_a2_no_h.restr_idxs) == 1  # only from C1
    center, _, _, _ = chiral_a2_no_h.restr_idxs[0]
    assert center == 1  # the chiral carbon with 3 heavy neighbors

    # Identity heavy-atom core: C->C, C->C, O->O, F->F
    identity_heavy = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    # Flip: swap O and F (indices 2 and 3) on the chiral center
    flipped_heavy = np.array([[0, 0], [1, 1], [2, 3], [3, 2]])

    id_perm_full = core_to_perm(identity_heavy, n_a2_full)
    fl_perm_full = core_to_perm(flipped_heavy, n_a2_full)
    id_perm_no_h = core_to_perm(identity_heavy, n_a2_no_h)
    fl_perm_no_h = core_to_perm(flipped_heavy, n_a2_no_h)

    assert not has_chiral_atom_flips(id_perm_full, chiral_a2_full, chiral_b2_full)
    assert not has_chiral_atom_flips(id_perm_no_h, chiral_a2_no_h, chiral_b2_no_h)
    assert has_chiral_atom_flips(fl_perm_full, chiral_a2_full, chiral_b2_full)
    assert has_chiral_atom_flips(fl_perm_no_h, chiral_a2_no_h, chiral_b2_no_h)

    # --- Part 5: Planar torsion detection works on H-stripped molecules ---

    # Cl/C=N\F has a defined planar torsion around the C=N double bond
    mol_a3 = Chem.AddHs(Chem.MolFromSmiles(r"Cl/C=N\F"))
    mol_b3 = Chem.AddHs(Chem.MolFromSmiles(r"Cl/C=N/F"))  # opposite E/Z config
    AllChem.EmbedMolecule(mol_a3, randomSeed=42)
    AllChem.EmbedMolecule(mol_b3, randomSeed=42)

    mol_a3_no_h = Chem.RemoveHs(mol_a3)
    mol_b3_no_h = Chem.RemoveHs(mol_b3)

    # Full-H planar torsion check
    find_flipped_full = setup_find_flipped_planar_torsions(mol_a3, mol_b3)
    # H-stripped planar torsion check
    find_flipped_no_h = setup_find_flipped_planar_torsions(mol_a3_no_h, mol_b3_no_h)

    n_a3_full = mol_a3.GetNumAtoms()
    n_a3_no_h = mol_a3_no_h.GetNumAtoms()

    # Identity heavy-atom core: Cl->Cl, C->C, N->N, F->F
    identity_core_3 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    id_perm_full_3 = core_to_perm(identity_core_3, n_a3_full)
    id_perm_no_h_3 = core_to_perm(identity_core_3, n_a3_no_h)

    flips_full = list(find_flipped_full(id_perm_full_3))
    flips_no_h = list(find_flipped_no_h(id_perm_no_h_3))

    # Both should detect planar torsion flips between cis and trans isomers
    assert len(flips_full) > 0, "Should detect planar torsion flip with full Hs"
    assert len(flips_no_h) > 0, "Should detect planar torsion flip with H-stripped molecules"


def test_chiral_h_augmentation_ring_ch2():
    """Verify that chiral-aware H augmentation detects and removes conflicting
    H mappings at ring -CH₂- centers.

    Ring -CH₂- carbons have only 2 heavy neighbors, producing 0 restraint tuples
    on H-stripped molecules (C(2,3)=0).  After H augmentation restores 4 neighbors,
    C(4,3)=4 tuples exist.  If the Hungarian distance-based assignment picks the
    wrong H pairing, a chiral conflict is introduced.  The chiral repair should
    detect the conflict and unmap the offending H pairs.
    """
    from tmd.fe.atom_mapping import (
        _augment_core_with_hydrogens,
        _build_h_removal_index_maps,
        _get_removable_h_neighbors,
    )
    from tmd.fe.chiral_utils import ChiralRestrIdxSet, find_atom_map_chiral_conflicts

    # Build two cyclohexane molecules with DIFFERENT conformers so that
    # the Hungarian algorithm cross-maps some Hs, creating chiral conflicts.
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMolecule(mol_a, randomSeed=42)
    AllChem.EmbedMolecule(mol_b, randomSeed=123)

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    # Build chiral sets from the full-H molecules
    chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
    chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

    # Each ring carbon has 4 neighbors (2 heavy + 2 H) → 6 carbons × C(4,3) = 6×4 = 24 restraint tuples
    assert len(chiral_set_a.restr_idxs) == 24, f"Expected 24 restraint tuples, got {len(chiral_set_a.restr_idxs)}"

    # H-stripped versions should have 0 restraint tuples for ring carbons
    # because each carbon only has 2 heavy neighbors → C(2,3) = 0
    mol_a_no_h = Chem.RemoveHs(mol_a)
    conf_a_no_h = get_romol_conf(mol_a_no_h)
    chiral_set_no_h = ChiralRestrIdxSet.from_mol(mol_a_no_h, conf_a_no_h)
    assert len(chiral_set_no_h.restr_idxs) == 0, "H-stripped ring CH₂ should have 0 chiral tuples"

    # Identity heavy core: carbon 0→0, 1→1, ..., 5→5
    heavy_core = np.array([[i, i] for i in range(6)])

    _, _, removed_h_a = _build_h_removal_index_maps(mol_a, mol_a_no_h)
    _, _, removed_h_b = _build_h_removal_index_maps(mol_b, Chem.RemoveHs(mol_b))

    # Augment WITHOUT chiral enforcement — should produce conflicts
    augmented_no_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
    )
    conflicts_before = find_atom_map_chiral_conflicts(augmented_no_chiral, chiral_set_a, chiral_set_b)
    assert len(conflicts_before) > 0, "Different conformers should produce chiral conflicts without enforcement"

    # Augment WITH chiral enforcement — should resolve conflicts by swapping Hs
    augmented_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
        chiral_set_a=chiral_set_a, chiral_set_b=chiral_set_b, enforce_chiral=True,
    )

    # The chiral-enforced core should keep all H pairs (swapped, not removed)
    assert len(augmented_chiral) == len(augmented_no_chiral), (
        f"Expected chiral enforcement to resolve conflicts via swap (same size), but got "
        f"{len(augmented_chiral)} != {len(augmented_no_chiral)}"
    )

    # Verify no chiral conflicts remain in the augmented core
    conflicts = find_atom_map_chiral_conflicts(augmented_chiral, chiral_set_a, chiral_set_b)
    assert len(conflicts) == 0, f"Expected 0 chiral conflicts with enforcement, got {len(conflicts)}"

    # Verify the swap actually changed some H assignments (not just identity)
    # On different conformers, at least one parent center should have a different
    # b-side assignment compared to the non-enforced (Hungarian-only) core.
    changed = False
    for idx in range(len(heavy_core), len(augmented_chiral)):
        pair_enforced = tuple(augmented_chiral[idx])
        # Check if this pair appears in the non-enforced core
        found = any(
            tuple(augmented_no_chiral[j]) == pair_enforced
            for j in range(len(heavy_core), len(augmented_no_chiral))
        )
        if not found:
            changed = True
            break
    assert changed, "Expected chiral enforcement to swap some H pairs, but assignments are identical"


def test_chiral_h_augmentation_swap_preserves_all_pairs():
    """Verify that the swap-based chiral repair preserves ALL H pairs at a
    ring -CH₂- center when a valid alternative permutation exists, rather
    than removing them.

    This specifically tests that the fix resolves conflicts without shrinking
    the core: for a center with 2 Hs, there are exactly 2 permutations, and
    if one conflicts then the other must be conflict-free (the heavy-atom
    mapping already passed chiral filtering).
    """
    from tmd.fe.atom_mapping import (
        _augment_core_with_hydrogens,
        _build_h_removal_index_maps,
    )
    from tmd.fe.chiral_utils import ChiralRestrIdxSet, find_atom_map_chiral_conflicts

    # Cyclopentane: 5 ring -CH₂- centers, each with 2 Hs
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C1CCCC1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C1CCCC1"))
    AllChem.EmbedMolecule(mol_a, randomSeed=42)
    AllChem.EmbedMolecule(mol_b, randomSeed=999)  # different conformer

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
    chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

    mol_a_no_h = Chem.RemoveHs(mol_a)
    mol_b_no_h = Chem.RemoveHs(mol_b)
    _, _, removed_h_a = _build_h_removal_index_maps(mol_a, mol_a_no_h)
    _, _, removed_h_b = _build_h_removal_index_maps(mol_b, mol_b_no_h)

    heavy_core = np.array([[i, i] for i in range(5)])

    # Without enforcement
    augmented_no_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
    )
    n_before = len(augmented_no_chiral)

    # With enforcement
    augmented_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
        chiral_set_a=chiral_set_a, chiral_set_b=chiral_set_b, enforce_chiral=True,
    )

    # All H pairs should still be present (swap, not remove)
    assert len(augmented_chiral) == n_before, (
        f"Swap-based repair should preserve all H pairs, "
        f"but core shrank from {n_before} to {len(augmented_chiral)}"
    )

    # And no chiral conflicts should remain
    conflicts = find_atom_map_chiral_conflicts(augmented_chiral, chiral_set_a, chiral_set_b)
    assert len(conflicts) == 0, f"Expected 0 conflicts after swap repair, got {len(conflicts)}"


def test_chiral_h_augmentation_through_get_cores():
    """Integration test: verify that get_cores with enforce_chiral=True produces
    augmented cores with no chiral conflicts on the full-H molecules.

    This tests the complete pipeline: H-stripped MCS → H augmentation → chiral repair.
    """
    from tmd.fe.chiral_utils import ChiralRestrIdxSet, find_atom_map_chiral_conflicts

    # Use cyclohexane pair — the ring -CH₂- centers are the key case
    mol_a = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    AllChem.EmbedMolecule(mol_a, randomSeed=42)
    AllChem.EmbedMolecule(mol_b, randomSeed=42)

    cores = atom_mapping.get_cores(
        mol_a, mol_b,
        ring_cutoff=0.15,
        chain_cutoff=0.30,
        max_visits=1e6,
        max_connected_components=1,
        min_connected_component_size=1,
        max_cores=100,
        enforce_core_core=True,
        ring_matches_ring_only=True,
        heavy_matches_heavy_only=True,
        enforce_chiral=True,
        disallow_planar_torsion_flips=False,
        min_threshold=0,
        initial_mapping=None,
    )

    assert len(cores) > 0, "Should find at least one core"

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
    chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

    for i, core in enumerate(cores):
        conflicts = find_atom_map_chiral_conflicts(core, chiral_set_a, chiral_set_b)
        assert len(conflicts) == 0, f"Core {i} has {len(conflicts)} chiral conflicts, expected 0"


def test_chiral_h_augmentation_no_conflict_case():
    """Verify that when the Hungarian assignment does not introduce chiral
    conflicts, the augmented core is identical with and without enforcement.

    Uses methanol (C(H₃)-OH) with identical conformers — the identity
    assignment is chirally correct so no Hs should be unmapped.
    """
    from tmd.fe.atom_mapping import (
        _augment_core_with_hydrogens,
        _build_h_removal_index_maps,
    )
    from tmd.fe.chiral_utils import ChiralRestrIdxSet

    # Methanol: C(H₃)-OH — the methyl carbon has 3 removable Hs
    mol_a = Chem.AddHs(Chem.MolFromSmiles("CO"))
    mol_b = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol_a, randomSeed=42)
    AllChem.EmbedMolecule(mol_b, randomSeed=42)

    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)

    chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
    chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

    mol_a_no_h = Chem.RemoveHs(mol_a)
    mol_b_no_h = Chem.RemoveHs(mol_b)
    _, _, removed_h_a = _build_h_removal_index_maps(mol_a, mol_a_no_h)
    _, _, removed_h_b = _build_h_removal_index_maps(mol_b, mol_b_no_h)

    # Heavy core: C→C, O→O
    heavy_core = np.array([[0, 0], [1, 1]])

    # Augment without chiral enforcement (distance-only)
    augmented_no_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
    )

    # Augment with chiral enforcement
    augmented_chiral = _augment_core_with_hydrogens(
        mol_a, mol_b, heavy_core, conf_a, conf_b, removed_h_a, removed_h_b,
        chiral_set_a=chiral_set_a, chiral_set_b=chiral_set_b, enforce_chiral=True,
    )

    # No conflicts on identical molecules → both augmented cores should be identical
    np.testing.assert_array_equal(
        augmented_no_chiral, augmented_chiral,
        err_msg="No-conflict case: chiral enforcement should not change the core",
    )
