"""Standard Python setup.py for building Cython modules.
"""
from __future__ import division

from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy

_INCLUDE_GSL_DIR = '/usr/local/include/'
_LIB_GSL_DIR = '/usr/local/lib/'


def MakeExtension(name, src=None):
  if not src:
    src = [name]
  src_arg = ['%s.pyx' % name for name in src]
  return Extension(name,
                   src_arg,
                   include_dirs=[numpy.get_include(), _INCLUDE_GSL_DIR],
                   library_dirs=[_LIB_GSL_DIR],
                   libraries=['gsl', 'gslcblas'],
                   language='c++')

ext_modules = [
    MakeExtension('cCollect'),
    MakeExtension('cDists'),
    MakeExtension('cPosteriors')
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
