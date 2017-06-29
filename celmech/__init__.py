# -*- coding: utf-8 -*-
"""Tools for celestial mechanics."""
# Make changes for python 2 and 3 compatibility
import sympy # sympy autodetects python 3 here, but errors when importing from other .py files in package
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
