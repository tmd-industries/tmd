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

# converts smirnoff xmls into python dictionaries.
import ast
import operator as op
import pprint
from argparse import ArgumentParser
from typing import Any
from xml.dom import minidom

import numpy as np
from openmm import unit

from tmd.ff.charges import AM1BCC_CHARGES, AM1CCC_CHARGES, AMBER_AM1_CCC_CHARGES, PRECOMPUTED_CHARGES, SIMPLE_CHARGES

CHARGE_TYPE_AMBER_AM1BCC = "AMBER-BCC"
CHARGE_TYPE_AMBER_AM1CCC = "AMBER-CCC"
CHARGE_TYPE_AM1BCC = "BCC"
CHARGE_TYPE_AM1CCC = "CCC"
CHARGE_TYPE_SIMPLE = "SC"
CHARGE_TYPE_PRECOMPUTED = "PRECOMPUTED"

BOND_TAG = "Bond"
ANGLE_TAG = "Angle"
PROPER_TAG = "Proper"
IMPROPER_TAG = "Improper"
VDW_TAG = "Atom"

tags = [BOND_TAG, ANGLE_TAG, PROPER_TAG, IMPROPER_TAG, VDW_TAG]


# (ytz): lol i think i wrote this originally
def _ast_eval(node):
    """
    Performs an abstract syntax tree evaluation of a unit.
    Parameters
    ----------
    node : An ast parsing tree node
    """

    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.BitXor: op.xor,
        ast.USub: op.neg,
    }

    if isinstance(node, ast.Constant):  # <number>
        return node.value
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](_ast_eval(node.left), _ast_eval(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](_ast_eval(node.operand))
    elif isinstance(node, ast.Name):
        # see if this is a openmm unit
        b = getattr(unit, node.id)
        return b
    # TODO: This was a quick hack that surprisingly worked. We should validate this further.
    elif isinstance(node, ast.List):
        return ast.literal_eval(node)
    else:
        raise TypeError(node)


def to_md_units(q):
    return q.value_in_unit_system(unit.md_unit_system)


def string_to_unit(unit_string):
    """
    Deserializes a openmm.unit.Quantity from a string representation, for
    example: "kilocalories_per_mole / angstrom ** 2"

    Parameters
    ----------
    unit_string : dict
        Serialized representation of a openmm.unit.Quantity.

    Returns
    -------
    output_unit: openmm.unit.Quantity
        The deserialized unit from the string

    """
    output_unit = _ast_eval(ast.parse(unit_string, mode="eval").body)
    return output_unit


def parse_quantity(number_string):
    """
    Parse a quantity into MD units.
    """
    pos = number_string.find("*")

    number = float(number_string[:pos])
    item = number_string[pos + 2 :]
    quantity = number * string_to_unit(item)
    return to_md_units(quantity)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert an openforcefield XML FF to a tmd FF")
    parser.add_argument("input_path", help="Path to XML ff")
    parser.add_argument(
        "--charge_type",
        default=CHARGE_TYPE_AMBER_AM1CCC,
        choices=[
            CHARGE_TYPE_AMBER_AM1BCC,
            CHARGE_TYPE_AMBER_AM1CCC,
            CHARGE_TYPE_SIMPLE,
            CHARGE_TYPE_AM1CCC,
            CHARGE_TYPE_AM1BCC,
            CHARGE_TYPE_PRECOMPUTED,
        ],
    )
    parser.add_argument("--output_path", help="Path to write FF file", default=None)
    args = parser.parse_args()

    if args.charge_type == CHARGE_TYPE_SIMPLE:
        print("Warning: Simple charges should only be used for testing")

    xmldoc = minidom.parse(args.input_path)
    forcefield: dict[str, Any] = {}

    for tag in tags:
        itemlist = xmldoc.getElementsByTagName(tag)
        if tag == BOND_TAG:
            params = []
            for s in itemlist:
                patt = s.attributes["smirks"].value
                b0 = parse_quantity(s.attributes["length"].value)
                kb = parse_quantity(s.attributes["k"].value)
                params.append([patt, kb, b0])
            bonds = {
                "patterns": params,
            }
            forcefield["HarmonicBond"] = bonds

        elif tag == ANGLE_TAG:
            params = []
            for s in itemlist:
                patt = s.attributes["smirks"].value
                a0 = parse_quantity(s.attributes["angle"].value)
                ka = parse_quantity(s.attributes["k"].value)
                params.append([patt, ka, a0])
            angles = {
                "patterns": params,
            }
            forcefield["HarmonicAngle"] = angles
        elif tag == PROPER_TAG:
            params = []
            for s in itemlist:
                patt = s.attributes["smirks"].value
                counter = 1
                components = []
                while True:
                    try:
                        k = parse_quantity(s.attributes["k" + str(counter)].value)
                        phase = parse_quantity(s.attributes["phase" + str(counter)].value)
                        period = float(s.attributes["periodicity" + str(counter)].value)
                        idivf = float(s.attributes["idivf" + str(counter)].value)
                        k = k / idivf
                        components.append([k, phase, period])
                        counter += 1
                    except KeyError:
                        break
                params.append([patt, components])
            torsions = {
                "patterns": params,
            }
            forcefield["ProperTorsion"] = torsions
        elif tag == IMPROPER_TAG:
            params = []
            for s in itemlist:
                patt = s.attributes["smirks"].value
                impdivf = 3
                k = parse_quantity(s.attributes["k1"].value) / impdivf
                phase = parse_quantity(s.attributes["phase1"].value)
                period = float(s.attributes["periodicity1"].value)
                params.append([patt, k, phase, period])
            impropers = {"patterns": params}
            forcefield["ImproperTorsion"] = impropers
        elif tag == VDW_TAG:
            params = []
            for s in itemlist:
                patt = s.attributes["smirks"].value
                epsilon = parse_quantity(s.attributes["epsilon"].value)
                if "rmin_half" in s.attributes:
                    rmin_half = parse_quantity(s.attributes["rmin_half"].value)
                    sigma = 2.0 * rmin_half / (2.0 ** (1.0 / 6.0))
                else:
                    sigma = parse_quantity(s.attributes["sigma"].value)
                # Take sqrt of epsilon to avoid singularity in backprop
                params.append([patt, sigma, float(np.sqrt(epsilon))])
            props: dict[str, Any] = {}
            for key, val in xmldoc.getElementsByTagName("vdW")[0].attributes.items():
                if key == "cutoff":
                    # we don't do cutoffs.
                    continue
                elif "scale" in key:
                    val = float(val)
                elif key == "switch_width":
                    continue
                if key == "version":
                    continue
                props[key] = val
            forcefield["LennardJones"] = {"patterns": params, "props": props}
    if args.charge_type == CHARGE_TYPE_AM1CCC:
        forcefield["AM1CCC"] = AM1CCC_CHARGES
    elif args.charge_type == CHARGE_TYPE_SIMPLE:
        forcefield["SimpleCharge"] = SIMPLE_CHARGES
    elif args.charge_type == CHARGE_TYPE_AM1BCC:
        forcefield["AM1BCC"] = AM1BCC_CHARGES
    elif args.charge_type == CHARGE_TYPE_AMBER_AM1CCC:
        forcefield["AmberAM1CCC"] = AMBER_AM1_CCC_CHARGES
    elif args.charge_type == CHARGE_TYPE_AMBER_AM1BCC:
        forcefield["AmberAM1BCC"] = AM1BCC_CHARGES
    elif args.charge_type == CHARGE_TYPE_PRECOMPUTED:
        forcefield["PrecomputedCharge"] = PRECOMPUTED_CHARGES
    else:
        assert False, f"Unknown charge type: {args.charge_type}"

    stream = None
    if args.output_path is not None:
        stream = open(args.output_path, "w")
    pp = pprint.PrettyPrinter(width=500, compact=False, stream=stream, indent=2)
    pp.pprint(forcefield)
