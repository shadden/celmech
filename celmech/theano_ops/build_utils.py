# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import pkg_resources
import os
from ctypes import cdll, c_char_p
import sysconfig

suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

pymodulepath = os.path.dirname(__file__)
clibcelmech = cdll.LoadLibrary(pymodulepath+"/../../libcelmech"+suffix)

# Version
__version__ = c_char_p.in_dll(clibcelmech, "celmech_version_str").value.decode('ascii')

def get_compile_args(compiler):
    opts = ["-std=c++11", "-O2", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    return opts


def get_cache_version():
    if "dev" in __version__:
        return ()
    return tuple(map(int, __version__.split(".")))


def get_header_dirs():
    dirs = [pkg_resources.resource_filename(__name__, "lib/include")]
    return dirs

__all__ = ["get_compile_args", "get_cache_version", "get_header_dirs"]

