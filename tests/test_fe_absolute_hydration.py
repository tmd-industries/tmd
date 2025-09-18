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

import numpy as np

from tmd import testsystems
from tmd.fe import absolute_hydration
from tmd.fe.free_energy import MDParams
from tmd.ff import Forcefield


def test_run_solvent_absolute_hydration():
    seed = 2022
    n_frames = 10
    n_eq_steps = 100
    n_windows = 8
    steps_per_frame = 10
    mol, _ = testsystems.ligands.get_biphenyl()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    md_params = MDParams(seed=seed, n_eq_steps=n_eq_steps, n_frames=n_frames, steps_per_frame=steps_per_frame)
    res, host_config = absolute_hydration.run_solvent(mol, ff, None, md_params=md_params, n_windows=n_windows)

    assert res.plots.overlap_summary_png is not None
    assert np.linalg.norm(res.final_result.dG_errs) < 20.0
    assert len(res.frames) == n_windows
    assert len(res.boxes) == n_windows
    assert len(res.frames[0]) == n_frames
    assert len(res.frames[-1]) == n_frames
    assert len(res.boxes[0]) == n_frames
    assert len(res.boxes[-1]) == n_frames
    assert res.md_params == md_params
    assert host_config.omm_topology is not None
    # The number of waters in the system should stay constant
    assert host_config.num_water_atoms == 6282
    assert host_config.conf.shape == (res.frames[0][0].shape[0] - mol.GetNumAtoms(), 3)
    assert host_config.box.shape == (3, 3)
