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

import ast

from tmd import constants
from tmd.ff.handlers import bonded, nonbonded, serialization_format
from tmd.ff.handlers.suffix import _SUFFIX


def deserialize_handlers(obj):
    """
    Parameters
    ----------
    obj: bytes-like
        the binary we wish to deserialize.

    Returns
    -------
    a handler from either bonded or nonbonded

    """
    obj_dict = ast.literal_eval(obj)

    handlers = []

    protein_ff = obj_dict.pop(serialization_format.PROTEIN_FF_TAG, constants.DEFAULT_PROTEIN_FF)
    water_ff = obj_dict.pop(serialization_format.WATER_FF_TAG, constants.DEFAULT_WATER_FF)

    for k, v in obj_dict.items():
        cls_name = k + _SUFFIX

        ctor = None

        try:
            ctor = getattr(bonded, cls_name)
        except AttributeError:
            pass

        try:
            ctor = getattr(nonbonded, cls_name)
        except AttributeError:
            pass

        if ctor is None:
            raise Exception("Unknown handler:", k)

        patterns = v["patterns"]
        smirks = []
        params = []

        for elems in patterns:
            smirks.append(elems[0])
            if len(elems) == 2:
                params.append(elems[1])
            else:
                params.append(elems[1:])

        props = v.get("props")

        handlers.append(ctor(smirks, params, props))

    return handlers, protein_ff, water_ff
