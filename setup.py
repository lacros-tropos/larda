#!/usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pyLARDA/peakTree_fastbuilder.pyx")
)


