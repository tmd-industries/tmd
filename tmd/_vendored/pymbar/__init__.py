##############################################################################
# Changes made for TMD
#
# * No longer imports timeseries, FES and testsytems module to avoid warning triggered by import
##############################################################################

##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2017 University of Colorado Boulder, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

"""The pymbar package contains the pymbar suite of tools for the analysis of
simulated and experimental data with the multistate Bennett acceptance
ratio (MBAR) estimator.

"""

__author__ = "Michael R. Shirts and John D. Chodera"
__license__ = "MIT"
__maintainer__ = "Levi N. Naden, Jaime Rodríguez-Guerra, Michael R. Shirts and John D. Chodera"
__email__ = "levi.naden@choderalab.org,jaime.rodriguez-guerra@choderalab.org,michael.shirts@colorado.edu,john.chodera@choderalab.org"

from importlib.metadata import version, PackageNotFoundError

from . import confidenceintervals
from .mbar import MBAR
from .other_estimators import bar, bar_overlap, bar_zero, exp, exp_gauss


__all__ = [
    "exp",
    "exp_gauss",
    "bar",
    "bar_overlap",
    "bar_zero",
    "MBAR",
    "confidenceintervals",
]

try:
    __version__ = version("pymbar")
except PackageNotFoundError:
    # package is not installed
    pass
