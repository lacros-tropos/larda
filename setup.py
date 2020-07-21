#!/usr/bin/python3
from setuptools import setup
from Cython.Build import cythonize

with open('README.md') as f:
    readme = f.read()

setup(
    name='larda',
    version='0.1.0',
    description='Python package for prediction of lidar backscatter and depolarization using cloud radar Doppler spectra.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Johannes Bühl, Martin Radenz, Willi Schimmel, Teresa Vogl',
    author_email='martin.radenz@tropos.de',
    url='https://github.com/lacros-tropos/larda',
    license='MIT License',
    packages=['pyLARDA'],
    #package_dir={'pyLARDA': __file__[:-8] + 'pyLARDA/'},
    include_package_data=True,
    python_requires='>=3.6',
    setup_requires=['wheel', 'cython']
    install_requires=['numpy>=1.19', 'scipy>=1.2', 'netCDF4>=1.4.2', 'msgpack==0.6.1', 'cython>=0.29.13',
                      'matplotlib>=3.0.2', 'requests>=2.21', 'toml>=0.10.0', 'tqdm>=4.36.1', 'numba>=0.45.1'],
    ext_modules=cythonize("pyLARDA/peakTree_fastbuilder.pyx"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
