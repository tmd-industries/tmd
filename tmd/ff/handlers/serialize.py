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

import io
import pprint

import numpy as np

from tmd.ff.handlers import serialization_format
from tmd.ff.handlers.suffix import _SUFFIX


def serialize_handlers(all_handlers, protein_ff, water_ff):
    final_ff = {}
    final_ff[serialization_format.PROTEIN_FF_TAG] = protein_ff
    final_ff[serialization_format.WATER_FF_TAG] = water_ff

    for handler in all_handlers:
        if handler is None:  # optional handler not specified
            continue
        ff_obj = handler.serialize()

        for k in ff_obj.keys():
            assert k not in final_ff, f"Handler {k} already exists"

        final_ff.update(ff_obj)

    return bin_to_str(final_ff)


def bin_to_str(binary):
    buf = io.StringIO()
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=buf)
    pp._sorted = lambda x: x
    pp.pprint(binary)
    return buf.getvalue()


class SerializableMixIn:
    def serialize(self):
        """

        Returns
        -------
        result : dict
        """
        handler = self
        key = type(handler).__name__[: -len(_SUFFIX)]
        patterns = []
        for smi, p in zip(handler.smirks, handler.params):
            if isinstance(p, (list, tuple)):
                patterns.append((smi, *p))
            elif isinstance(p, np.ndarray):
                patterns.append((smi, *p.tolist()))
            else:
                # SimpleCharges only have one parameter
                patterns.append((smi, float(p)))

        body = {"patterns": patterns}
        if handler.props is not None:
            body["props"] = handler.props

        result = {key: body}

        return result
