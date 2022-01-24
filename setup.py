#!/usr/bin/python3

import re
from setuptools import setup
from Cython.Build import cythonize

with open('README.md') as f:
    readme = f.read()


#VERSIONFILE="pyLARDA/_version.py"
#verstrline = open(VERSIONFILE, "rt").read()
#VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
#mo = re.search(VSRE, verstrline, re.M)
#if mo:
#    verstr = mo.group(1)
#else:
#    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
#from pyLARDA import __version__ as verstr
#from pyLARDA import __author__ as authorstr
meta = {}
with open("pyLARDA/_meta.py") as fp:
    exec(fp.read(), meta)

setup(
    name='pyLARDA',
    version=meta['__version__'],
    description='Data cube for handling atmospheric observations of profiling remote sensing instruments.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author=meta['__author__'],
    author_email='radenz@tropos.de',
    url='https://github.com/lacros-tropos/larda',
    download_url='https://github.com/lacros-tropos/larda/archive/refs/tags/v3.3.tar.gz',
    license='MIT License',
    packages=['pyLARDA'],
    include_package_data=True,
    python_requires='>=3.8',
    # automatic installation of the dependencies did not work with the test.pypi
    # below the try to fix it
    setup_requires=['wheel', 'numpy==1.21', 'scipy>=1.6', 'netCDF4>=1.4.2', 'msgpack', 'cython>=0.29.13', 'xarray',
                      'matplotlib>=3.0.2', 'requests>=2.21', 'toml>=0.10.0', 'tqdm>=4.36.1', 'numba>=0.45.1'],
    install_requires=['numpy==1.21', 'scipy>=1.6', 'netCDF4>=1.4.2', 'msgpack', 'cython>=0.29.13', 'xarray',
                      'matplotlib>=3.0.2', 'requests>=2.21', 'toml>=0.10.0', 'tqdm>=4.36.1', 'numba>=0.45.1'],
    ext_modules=cythonize("pyLARDA/peakTree_fastbuilder.pyx"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
