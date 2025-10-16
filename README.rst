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

*eTraGo* is a cross-sectoral grid planning tool focusing on the extra-high and high-voltage power grid level. In addition to the electricity sector, the gas (methane and hydrogen), heat and mobility sectors are considered. *eTraGo* optimizes grid and storage expansion as well as power plant deployment, taking into account various flexibility options. These include electrical flexibilities such as demand-side management or dynamic line rating, as well as flexibilities arising from sector coupling, such as heat stores, gas stores or shifting potentials from charging electric vehicles. *eTraGo* is an open-source Python package primarily based on `PyPSA <https://pypsa.readthedocs.io/en/latest/>`_.

*eTraGo* is documented on `readthedocs <https://etrago.readthedocs.io>`_.

Key Features
------------

* open, cross-sectoral grid planning tool on extra-high and high-voltage level
* includes a variety of functionalities for scenario variations and reduction of complexity in spatial and temporal dimension
* market optimization as well as linear-optimal power flow available
* analyses and plots the results


Input Data
==========

The grid model data for *eTraGo* was created using the open-source tool
`eGon-data <https://github.com/openego/eGon-data>`_.
The resulting data is currently available in a test environment of the
`OpenEnergyPlatform <https://openenergy-platform.org/>`_.
In the future, it will be published on the main
`OpenEnergyPlatform <https://openenergy-platform.org/>`_.

As an alternative, a local database can be used.
We have published the scenarios as backups of the required tables,
along with instructions on how to use them, on Zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8376714.svg
   :target: https://doi.org/10.5281/zenodo.8376714
   
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.13143968.svg
   :target: https://doi.org/10.5281/zenodo.13143968
   

Installation
============

*eTraGo* is implemented as a Python package; therefore, Python 3 
must be installed on your system. The current version of *eTraGo* is 
compatible with Python 3.10 or higher.

If you already have a working Python3 environment, use *pypi* to install the latest
*eTraGo* version. We highly recommend using a virtual environment.
Use the following command in order to install *eTraGo*.

.. code-block:: bash

  $ pip3 install eTraGo

Installation for Developers
===========================

Clone the source code from *GitHub*:

.. code-block::

   $ git clone https://github.com/openego/eTraGo

You can checkout to the dev branch and create new feature branches.
For the correct work-flow, please mind the  `guidelines <https://etrago.readthedocs.io/contributing.html>`_.

Use the following code to install *eTraGo* directly from the cloned repository:

.. code-block::

   $ pip3 install -e /path/to/eTraGo/ 

If you want to display geographical features in the background of network plots, please install *cartopy*:

.. code-block::

   $ pip3 install cartopy

If you run into problems when using cartopy, try to install *shapely* without binaries:

.. code-block::

   $ pip3 install shapely --no-binary shapely

Using a Virtual Environment
===========================

Before installing *eTraGo*, 
create a virtual environment in your preferred location and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.10
   $ source venv/bin/activate
   $ cd venv

Inside your activated virtual environment you can 
install *eTraGo* with the pip command, as previously explained.


Copyleft
=========================

Code licensed under "GNU Affero General Public License Version 3 (AGPL-3.0)"
It is a collaborative work with several copyright owners:
Cite as "eTraGo" © Flensburg University of Applied Sciences, Centre for
Sustainable Energy Systems © Europa-Universität Flensburg, Centre for
Sustainable Energy Systems © DLR Institute for Networked Energy Systems"
