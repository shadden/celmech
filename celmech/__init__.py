# -*- coding: utf-8 -*-
"""Tools for celestial mechanics."""

import sympy # sympy autodetects python 3 here, but errors when importing from other .py files in package

# Make changes for python 2 and 3 compatibility
try:
    import builtins      # if this succeeds it's python 3.x
    builtins.xrange = range
    builtins.basestring = (str,bytes)
except ImportError:
    pass                 # python 2.x

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

from .andoyer import Andoyer, AndoyerHamiltonian
from .poincare import PoincareParticle, Poincare, PoincareHamiltonian

__all__ = ["Andoyer", "AndoyerHamiltonian", "PoincareParticle", "Poincare", "PoincareHamiltonian"]
