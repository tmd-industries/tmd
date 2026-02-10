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


__version__ = "0.2.0"


def _suppress_jax_no_gpu_warning():
    """Suppresses the JAX 'No GPU/TPU found' warning when no GPU is available.

    Only forces CPU mode if no GPU is actually available, allowing GPU usage
    when GPUs are present.

    See https://github.com/google/jax/issues/6805
    """
    import jax

    # Check if GPU is available before forcing CPU
    # This avoids suppressing GPU usage when GPUs are actually present
    try:
        devices = jax.devices()
        has_gpu = any(d.platform in ("gpu", "cuda") for d in devices)
        if not has_gpu:
            # Only force CPU if no GPU is available (suppresses the warning)
            jax.config.update("jax_platform_name", "cpu")
    except Exception:
        # If device detection fails, don't change platform settings
        pass


_suppress_jax_no_gpu_warning()
