# -*- coding: utf-8 -*-
"""Tools for celestial mechanics."""

import sympy # sympy autodetects python 3 here, but errors when importing from other .py files in package
# Find suffix
import sysconfig
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# Import shared library
import os
import warnings
pymodulepath = os.path.dirname(__file__)
from ctypes import cdll, c_char_p
clibcelmech = cdll.LoadLibrary(pymodulepath+"/../libcelmech"+suffix)

# Version
__version__ = c_char_p.in_dll(clibcelmech, "celmech_version_str").value.decode('ascii')

# Build
__build__ = c_char_p.in_dll(clibcelmech, "celmech_build_str").value.decode('ascii')

# Githash
__githash__ = c_char_p.in_dll(clibcelmech, "celmech_githash_str").value.decode('ascii')

# Check for version
try:
    moduleversion = pkg_resources.require("celmech")[0].version
    libcelmechversion = __version__
    if moduleversion != libcelmechversion:
        warnings.warn("WARNING: python module and libcelmech have different version numbers: '%s' vs '%s'.\n" %(moduleversion, libcelmechversion), ImportWarning)
except:
    # Might fails on python3 versions, but not important
    pass

sympy.init_printing(use_latex='mathjax')
from .poincare import PoincareParticle, Poincare, PoincareHamiltonian
from .hamiltonian import Hamiltonian,PhaseSpaceState
from .canonical_transformations import CanonicalTransformation
from .miscellaneous import get_symbol, get_symbol0
__all__ = ["Hamiltonian","PhaseSpaceState","Andoyer", "AndoyerHamiltonian", "PoincareParticle", "Poincare",
           "PoincareHamiltonian","CanonicalTransformation","LaplaceLagrangeSystem", "get_symbol", "get_symbol0"]
