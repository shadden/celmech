try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from codecs import open
import os
import sys

import sysconfig
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"

# Try to get git hash
try:
    import subprocess
    ghash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii")
    ghash_arg = "-DCELMECHGITHASH="+ghash.strip()
except:
    ghash_arg = "-DCELMECHGITHASH=b48d1d9fb4e0e7a82402ea61d84f9dff899f048d" #GITHASHAUTOUPDATE

extra_link_args=[]
if sys.platform == 'darwin':
    from distutils import sysconfig
    vars = sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-shared')
    extra_link_args=['-Wl,-install_name,@rpath/libcelmech'+suffix]
    
libcelmechmodule = Extension('libcelmech',
                    sources = [ 
                        'src/disturbing_function.c',
                        'src/poisson_series.c',
                        'src/fmft.c',
                        'src/fmftPy.c',
                        'src/nrutil.c'
                    ],
                    include_dirs = ['src'],
                    define_macros=[ ('LIBCELMECH', None) ],
                    # Removed '-march=native' for now.
                    extra_compile_args=['-fstrict-aliasing', '-O3','-std=c99','-Wno-unknown-pragmas', ghash_arg, '-DLIBCELMECH', '-D_GNU_SOURCE', '-fPIC'],
                    extra_link_args=extra_link_args,
                    )

if not os.getenv('READTHEDOCS'):
    packages = ['exoplanet-core>=0.3.0','pytensor>=2.18' ,'sympy>=1.1.1', 'numpy', 'scipy>=1.2.0', 'reboundx>=4.0.0', 'rebound>=4.0.1', 'mpmath>=1.0.0']
    try:
        install_requires += packages
    except:
        install_requires = packages

setup(name='celmech',
    version='1.5.2',
    description='Open source tools for celestial mechanics',
    url='http://github.com/shadden/celmech',
    author='Dan Tamayo, Sam Hadden',
    author_email='tamayo.daniel@gmail.com, shadden1107@gmail.com',
    license='GPL',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='astronomy astrophysics celestial-mechanics orbits orbital-mechanics',
    packages=['celmech'],
    install_requires=['exoplanet-core>=0.3.0rc2','pytensor>=2.18', 'mpmath>=1.0.0', 'sympy>=1.1.1', 'rebound>=4.0.1', 'reboundx>=4.0.0', 'numpy', 'scipy>=1.2.0'],
    test_suite="celmech.test",
    ext_modules = [libcelmechmodule],
    zip_safe=False)
