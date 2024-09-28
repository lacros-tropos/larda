
####################################
PyPi package
####################################

Preparation
^^^^^^^^^^^^^^

#. Adapt the version number in _meta


.. note::
    The Cython module for ``peakTree_fastreader.pyx`` is left out from the pypi package to simplify the build process.


Build the .tar and .whl (new)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python3.11 -m build
    python3.11 -m twine upload -r testpypi --verbose dist/pyLARDA-3.3.7-*
    python3.11 -m twine upload -r testpypi --verbose dist/pylarda-3.3.7*



Legacy build process
^^^^^^^^^^^^^^^^^^^^^^^^


Build the .tar package
------------------------------

.. code-block:: bash

    python3 setup.py sdist
    # check build files
    python3 -m twine check dist/*
    tar tzf dist/pyLARDA-3.3.tar.gz
    # upload to test server
    python3 -m twine upload --repository testpypi dist/* 
    #
    python3 -m twine upload dist/*


Build the windows binaries
------------------------------

 Using the respective anaconda version:

.. code-block:: bash

    python3 setup.py bdist_wheel


.. code-block:: bash

    python3 setup.py bdist_wheel

    set CONDA_FORCE_32BIT=1
    conda create -n py38_32bit python=3.8


Build the manylinux binaries
------------------------------

pypi only accepts the manylinux wheels.
Most conveniently they are packaged with the manylinux docker images (I only got the manylinux2014 working).


.. code-block:: bash

    docker run -it -v $(pwd):/io 90ac8ec # or whatever id your manylinux image has

    # might be necessary to update some dependencies
    python3.9 -m pip install Cython
    python3.9 -m pip install numba -U

    python3.8 setup.py bdist_wheel
    # and/or
    python3.9 setup.py bdist_wheel
    
    auditwheel repair <name-of-the-produced-wheel>.whl


