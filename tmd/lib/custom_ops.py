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

# This file contains stubs for some classes defined in the C++
# extension module.
#
# The purpose is to allow importing modules with
# unavoidable top-level references to objects defined in the extension
# (e.g. modules containing subclasses of classes defined in the C++
# code).
#
# If the extension module .so file is present, the definitions
# in it will take precedence over the stubs defined here.


class Context_f32:
    pass


class Context_f64:
    pass


class Potential_f32:
    pass


class Potential_f64:
    pass


class BoundPotential_f32:
    pass


class BoundPotential_f64:
    pass


class FanoutSummedPotential_f32:
    pass


class FanoutSummedPotential_f64:
    pass


class SummedPotential_f32:
    pass


class SummedPotential_f64:
    pass


class TIBDExchangeMove_f32:
    pass


class TIBDExchangeMove_f64:
    pass


class MonteCarloBarostat_f32:
    pass


class MonteCarloBarostat_f64:
    pass


class AnisotropicMonteCarloBarostat_f32:
    pass


class AnisotropicMonteCarloBarostat_f64:
    pass
