# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2026, Forrest York
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

from tmd.constants import DEFAULT_NB_CUTOFF
from tmd.md.builders import load_pdb_system
from tmd.utils import path_to_internal_file


def setup_dhfr(nb_cutoff: float = DEFAULT_NB_CUTOFF):
    # Note that OpenMM uses a cutoff of 1.0 for its DHFR + Explicit-RF benchmarks (https://openmm.org/benchmarks)
    with path_to_internal_file("tmd.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
        host_config = load_pdb_system(str(pdb_path), "amber99sbildn", "tip3p", nb_cutoff=nb_cutoff)

    return host_config.host_system.get_U_fns(), host_config.masses, host_config.conf, host_config.box
