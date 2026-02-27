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


__version__ = "0.3.0"


def _suppress_jax_no_gpu_warning():
    """Suppresses the following warning:

       WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

    See https://github.com/google/jax/issues/6805
    """
    import jax

    jax.config.update("jax_platform_name", "cpu")


_suppress_jax_no_gpu_warning()
