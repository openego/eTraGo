============
About eTraGo
============

*eTraGo* (**e**\lectric **Tra**\nsmission **G**\rid **o**\ptimization) has been developed as part of a comprehensive toolchain. 
The following sections introduce key projects, data models and software tools within this context.
More detailed information can be found in the publications listed below or the following reports: [openeGo_report]_, [eGon_report]_


Research Projects and Publications
==================================

*eTraGo* was initially developed within the research projects
`open_eGo <https://openegoproject.wordpress.com>`_ and `eGon <https://ego-n.org/>`_.
It is constantly further developed in different reserach projects, 
e.g. `PoWerD <https://www.h2-powerd.de/>`_. 

Publications:

* *The impact of redispatch on grid and storage expansion planning in the German energy system* [Buettner20242]_
* *Avoiding False Inter-Zonal Meshing in the Clustering of a Large-Scale German Power Grid* [Esterl2024]_
* *Influence of flexibility options on the German transmission grid — A sector-coupled mid-term scenario* [Buettner2024]_
* *Integrated Techno-Economic Power System Planning of Transmission and Distribution Grids* [Mueller2019]_
* *The eGo grid model: An open source approach towards a model of German high and extra-high voltage power grids* [Mueller20181]_
* *The Role of the High Voltage Power Level in Future Power Systems and Their Modelling* [Mueller20182]_


eTraGo as part of the eGo-Toolchain
===================================

.. figure:: images/ego_tools.svg
   :align: center
   :scale: 75%

   
The tools illustrated in the graph above have been developed for cross-grid level optimization of the sector-coupled energy system in Germany as part of the aforementioned research projects. The following section describes the individual elements of this toolchain. 

Open Energy Platform
--------------------

The `Open Energy Platform <https://openenergy-platform.org/>`_ has been developed within the *open_eGo* project.
It addresses students and researchers as well as interested people in the field of energy modelling and analytics and provides tools to make energy system modelling process transparent.

All data of the *open_eGo* and *eGon* project are stored at the *Open Energy Platform*. *eTraGo* retrieves the input data from the *Open Energ Platform* and enables to store back the results.
For access to the coresponding data models, registration and login are necessary. Learn more about the database access `here  <https://oep-data-interface.readthedocs.io>`_.

Data Model Creation
-------------------

The corresponding data model is created using the Python tool `eGon-data <https://github.com/openego/eGon-data>`, which represents a further development of the `data processing <https://github.com/openego/data_processing>` tool. The model covers the coupling of electricity grid models on different voltage levels with a gas grid model, demands and flexibilities from the mobility, heat and hydrogen sectors as well as the integration of other electrical flexibilities such as demand-side management and dynamic line rating. It is characterised by a high spatial resolution within Germany, while other countries are considered in an aggregated form. Several future scenarios have been developed, each covering one year in hourly resolution and differing in terms of generation, demand and availability of some technologies. 

The developed data model provides data consistent on different aggregation levels and serves as input for the tools `eTraGo <https://github.com/openego/eTraGo>`_, `eDisGo <https://github.com/openego/eDisGo>`_ and `ding0 <https://github.com/openego/ding0>`_. The outputs of *eGon-data* are published under open source and open data licenses.

Distribution Grid Optimization
------------------------------

`Ding0 <https://dingo.readthedocs.io/>`_ (**Di**\stribution **N**\etwork **G**\enerat**0**\r) is a tool to generate synthetic 
medium and low voltage distribution grids based on open data.

`eDisGo <https://dingo.readthedocs.io/>`_ (**e**\lectric **Dis**\tributon **G**\rid **o**\ptimization) provides a toolbox for optimization and analysis
of medium and low voltage distribution grids.

Cross-Grid Level Optimization
-----------------------------

`eGo <http://openego.readthedocs.io/>`_ combines the tools *eTraGo* and *eDisGo* for cross-grid level optimization of the sector-coupled energy system in Germany.


License
=======

© Copyright 2015-2024
Flensburg University of Applied Sciences,
Europa-Universität Flensburg,
Centre for Sustainable Energy Systems and
DLR-Institute for Networked Energy Systems

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
more details.

You should have received a copy of the GNU General Public License along
with this program.
If not, see `www.gnu.org/licenses <https://www.gnu.org/licenses/>`_.
