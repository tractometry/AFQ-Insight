############
Installation
############

*AFQ-Insight* requires Python >= 3.7.

Installing the release version
------------------------------

The recommended way to install *AFQ-Insight* is from PyPI,

.. code-block:: console

    $ pip install afqinsight

This will install *AFQ-Insight* and all of its dependencies.

Installing the development version
----------------------------------

The development version is less stable but may include new features.
You can install the development version using ``pip``:

.. code-block:: console

    pip install git+https://github.com/tractometry/AFQ-Insight.git

Alternatively, you can clone the source code from the `github repository
<https://github.com/tractometry/AFQ-Insight>`_:

.. code-block:: console

    $ git clone git@github.com:tractometry/AFQ-Insight.git
    $ cd AFQ-Insight
    $ pip install .

If you would like to contribute to *AFQ-Insight*, see the `contributing guidelines
<contributing.html>`_.

Next, go to the `user guide <user_guide.html>`_ or see the `example gallery
<auto_examples/index.html>`_ for further information on how to use *AFQ-Insight*.

Installing on M1 / M2 Macs
~~~~~~~~~~~~~~~~~~~~~~~~~~
Due to the complexity of installing support of HDF5 on newer macs, we recommend
using the following steps to install AFQ-Insight on M1 / M2 Macs and relying
on the `homebrew <https://brew.sh/>`_ package manager to install the required
dependencies.

.. code-block:: console

    $ pip install cython
    $ brew install hdf5
    $ brew install c-blosc
    $ export HDF5_DIR=/opt/homebrew/opt/hdf5
    $ export BLOSC_DIR=/opt/homebrew/opt/c-blosc
    $ pip install tables
