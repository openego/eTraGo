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

*eTraGo* (**e**\lectric **Tra**\nsmission **G**\rid **o**\ptimization) is a cross-sectoral grid planning tool focusing on the extra-high and high-voltage power grid level. In addition to the electricity sector, the gas (methane and hydrogen), heat and mobility sectors are considered. *eTraGo* optimizes grid and storage expansion as well as power plant deployment, taking into account various flexibility options. These include electrical flexibilities such as demand-side management or dynamic line rating, as well as flexibilities arising from sector coupling, such as heat stores, gas stores or shifting potentials from charging electric vehicles. *eTraGo* is an open-source Python package primarily based on `PyPSA <https://pypsa.readthedocs.io/en/latest/>`_.

*eTraGo* has been developed as part of a comprehensive toolchain. 
Explore related models and tools, detailed guidance on installation, an overview of the functionalities, and references to projects and studies where it has been used, all on `readthedocs <https://etrago.readthedocs.io>`_.

Key Features
------------

* open, cross-sectoral grid planning tool on extra-high and high-voltage level
* includes a variety of functionalities for scenario variations and reduction of complexity in spatial and temporal dimension
* market optimization as well as linear-optimal power flow available
* analyses and plots the results


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
  
If you want to display geographical features in the background of network plots, please install *cartopy*:

.. code-block::

   $ pip3 install cartopy
  
For a more detailed guide on the installation, please refer to the readthedocs section `Installation <https://etrago.readthedocs.io/en/latest/installation.html>`_.

Input Data
==========

The grid model data for *eTraGo* was created using the open-source tool
`eGon-data <https://github.com/openego/eGon-data>`_.
The resulting data is available on the `OpenEnergyPlatform <https://openenergy-platform.org/>`_.

As an alternative, a local database can be used.
We have published the scenarios as backups of the required tables,
along with instructions on how to use them, on Zenodo. 
For more information on available scenarios and instructions on the usage of a local database, 
please refer to the readthedocs sections `Installation <https://etrago.readthedocs.io/en/latest/installation.html>`_ 
and `Functionalities <https://etrago.readthedocs.io/en/latest/theoretical_background.html>`_.


Copyleft
==========

Code licensed under "GNU Affero General Public License Version 3 (AGPL-3.0)"
It is a collaborative work with several copyright owners:
Cite as "eTraGo" © Flensburg University of Applied Sciences, Centre for
Sustainable Energy Systems © Europa-Universität Flensburg, Centre for
Sustainable Energy Systems © DLR Institute for Networked Energy Systems
