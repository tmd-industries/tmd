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

{
    "HarmonicAngle": {"patterns": [("[*:1]~[*:2]~[*:3]", 100.0, 1.5707963267948966)]},
    "HarmonicBond": {"patterns": [("[*:1]~[*:2]", 100000.0, 0.1)]},
    "ImproperTorsion": {"patterns": [("[*:1]~[#6X3,#7X3:2](~[*:3])~[*:4]", 1.0, 3.141592653589793, 2.0)]},
    "LennardJones": {"patterns": [("[*:1]", 0.1, 1.0)]},
    "LennardJonesIntra": {"patterns": [("[*:1]", 0.1, 1.0)]},
    "LennardJonesSolvent": {"patterns": [("[*:1]", 0.1, 1.0)]},
    "ProperTorsion": {"patterns": [("[*:1]~[*:2]~[*:3]~[*:4]", [1.0, 0.0, 1.0])]},
    "ProteinForcefield": "amber99sbildn",
    "SimpleCharge": {"patterns": [("[*:1]", 0.0)]},
    "SimpleChargeIntra": {"patterns": [("[*:1]", 0.0)]},
    "SimpleChargeSolvent": {"patterns": [("[*:1]", 0.0)]},
    "WaterForcefield": "amber14/tip3p",
}
