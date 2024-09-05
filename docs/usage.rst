Usage
=====

.. _installation:

Clone the repository to your local machine or HPC cluster:

.. code-block:: console

   $ git clone https://github.com/antolonappan/echolens.git
   $ cd echolens

Conda Environment
-----------------

To set the conda environment run the following command:

.. code-block:: console

   $ chmod u+x conda.sh
   $ ./conda.sh

If this doesn't work, you can create the conda environment manually.
The `conda` direcotry contains the environment yml files with and without packages build. To create a new conda environment with all the required packages, run the following command:

.. code-block:: console

   $ conda env create -f conda/environment_< version >.yml


Development mode
-----------------
Currently the package is in development mode. To install the package in development mode, run the following command:

.. code-block:: console

   $ python setup.py develop


Importing the package
---------------------
To import the package, run the following command in python:

.. code-block:: python

   >>> import echolens
   >>> print(echolens.__version__)
   >>> echolens.CMB_bharat().get_frequency()

Now you are ready to use the package and try the examples in the notebooks.


