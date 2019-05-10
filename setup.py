#!/usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='larda',
    version='0.1',
    packages=['pyLARDA',],
    license='MIT License',
    ext_modules = cythonize("pyLARDA/peakTree_fastbuilder.pyx")
)


