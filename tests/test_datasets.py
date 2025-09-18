# Copyright 2019-2025, Relay Therapeutics
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

import pytest

from tmd.testsystems import fetch_freesolv

pytestmark = [pytest.mark.nocuda]


def test_fetch_freesolv():
    """assert expected number of molecules loaded -- with unique names and expected property annotations"""
    mols = fetch_freesolv()

    # expected number of mols loaded
    assert len(mols) == 642

    # expected mol properties present, interpretable as floats
    for mol in mols:
        props = mol.GetPropsAsDict()
        _, _ = float(props["dG"]), float(props["dG_err"])

    # unique names
    names = [mol.GetProp("_Name") for mol in mols]
    assert len(set(names)) == len(names)

    # truncated list
    mols = fetch_freesolv(n_mols=5)
    assert len(mols) == 5

    # excluded mol
    exclude_name = mols[0].GetProp("_Name")
    mols = fetch_freesolv(exclude_mols=set([exclude_name]))
    assert len(mols) == 641
    names = [mol.GetProp("_Name") for mol in mols]
    assert exclude_name not in names
