|ci| |docs|

.. |ci| image::
    https://img.shields.io/github/actions/workflow/status
    /openego/eTraGo/ci.yaml?branch=dev&event=push&label=ci
    :alt: Continuous Integration Workflow Status
    :target: https://github.com/openego/eTraGo/actions/workflows/ci.yaml

.. |docs| image::
    https://readthedocs.org/projects/etrago/badge/?version=latest
    :alt: Documentation Status
    :target: http://etrago.readthedocs.io/en/latest/?badge=latest

.. end-header

eTraGo
======

Optimization of flexibility options for transmission grids based on PyPSA

A speciality in this context is that transmission grids are described by the
380, 220 and 110 kV in Germany. Conventionally, the 110kV grid is part of the
distribution grid. The integration of the transmission and 'upper' distribution
grid is part of eTraGo.

The focus of optimization are flexibility options with a special focus on
energy storages. Grid expansion measures are not part of this tool and will be
instead part of 'eGo' https://github.com/openego/eGo

eTraGo is documented on `readthedocs <https://etrago.readthedocs.io>`_.


Input data
==========
The grid model data for eTraGo was created with the open source tool 
`eGon-data <https://github.com/openego/eGon-data.>`_. The resulting data will 
be pubished on the `OpenEnergyPlatform <https://openenergy-platform.org/.>`_.
As long as the data is not published there, a local database is needed. 
We published a backup of the required tables and instruction on how to use it 
on zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8376714.svg
   :target: https://doi.org/10.5281/zenodo.8376714


Installation
============
eTraGo is designed as a Python package therefore it is mandatory to have
`Python 3 <https://www.python.org/downloads/.>`_ installed. If you have a
working Python3 environment, use pypi to install the latest eTraGo version.
We highly recommend you to use a virtual environment. Use following pip
command in order to install eTraGo..

.. code-block:: bash

  $ pip3 install eTraGo

Installation for Developers
===========================

Clone the source code from github:

.. code-block::

   $ git clone https://github.com/openego/eTraGo

You can checkout to the dev branch and create new feature branches.
For the correct work-flow, please mind the 
`Dreissen Branching Model <https://nvie.com/posts/a-successful-git-branching-model/>`_

Use the pip -e to install eTraGo directly from the cloned repository:

.. code-block::

   $ pip3 install -e /path/to/eTraGo/ 

When you want to draw geographical features in the background of network plots,
please install cartopy:

.. code-block::

   $ pip3 install cartopy

If you run into problems when using cartopy, try to install shapely without binaries:

.. code-block::

   $ pip3 install shapely --no-binary shapely

Using a virtual environment
===========================

Before installing eTraGo, 
you create a virtual environment (where you like it) and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.10
   $ source venv/bin/activate
   $ cd venv

Inside your activated virtual environment you can 
install eTraGo with the pip command, as previously explained.


Copyleft
=========================

Code licensed under "GNU Affero General Public License Version 3 (AGPL-3.0)"
It is a collaborative work with several copyright owners:
Cite as "eTraGo" © Flensburg University of Applied Sciences, Centre for
Sustainable Energy Systems © Europa-Universität Flensburg, Centre for
Sustainable Energy Systems © DLR Institute for Networked Energy Systems"
