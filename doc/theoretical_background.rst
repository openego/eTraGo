======================
Theoretical Background
======================



Definitions and Units
=====================

eTraGo executes the Open Source software PyPSA to perform power flow 
simulations and uses its definitions and
`units <https://pypsa.org/doc/conventions.html#unit-conventions>`_.



Assumptions on Data
===================

eTraGo fetches its necessary input data from the OpenEnergy Platform including
load, generation, grid and scenario-related data. More details can be found in
the `Data-Processing <https://data-processing.readthedocs.io/en/latest>`_.

As overview, the Open Source grid structure is developed by processing data 
from `OpenStreetMap (OSM) <https://www.openstreetmap.de>`_ to obtain 
geo-referenced locations of substations and links equal or above the 110 kV
voltage level. OSM also provides information about residential, retail,
industrial and agricultural areas which is used with standardized profiles to
obtain load data. Generation data of solar and wind rely on weather data from
[coastdat-2]_. Both, load and generation data, match the annual amount for the
year 2011. eTraGo enables the investigation of three scenarios - Status Quo,
NEP 2035 and eGo100. Status Quo corresponds to the actual grid, NEP2035 
follows assumptions for the year 2035 by [NEP2015]_ and eGo100 assumes to 
operate the future energy system completely by renewables [ehighway2050]_.


Methods
===========

PyPSA
-----
The power flow simulations are performed by the Open Source tool
`PyPSA <https://pypsa.org/doc/index.html>`_ with a linear approximation for the
optimization of power flows in general. Expecting that eTraGo fulfills the 
assumptions to perfom a LOPF (small voltage angle differences, branch 
resistances negligible to their reactances, voltage magnitudes can be kept at
nominal values) since it focuses on the extra-high and high voltage levels. As
objective value of the optimization, the overall system costs are considered.


Clustering approaches
---------------

EHV-Clustering
^^^^^^^^^^^^^^

This method maps an input network to an output network with the nodes of the
extra-high voltage level. All nodes with a voltage level below the extra-high
voltage level are mapped to their nearest neighboring node in the extra-high
voltage level with the
`dijkstra algorithm <https://en.wikipedia.org/wiki/Dijkstra's_algorithm>`_
(110 kV ---> 220,380 kV).

K-Means Clustering
^^^^^^^^^^^^^^^^^^

This `method <https://en.wikipedia.org/wiki/K-means_clustering>`_ maps an input
network to a new output network with an adjustable number of nodes and new
coordinates. The algorithm sets these coordinates randomly and minimizes a
certain parameter like for example the distances between old coordinates and
their nearest neighbor in the set of new coordinates. The method was
implemented by `Hoersch et al. <https://arxiv.org/pdf/1705.07617.pdf>`_ within
PyPSA.

Snapshot skipping
^^^^^^^^^^^^^^^^^
This method simplifies the simulation temporally by considering every n-th
snapshot of a given time series. The regarded snapshots are weighted by the
number of neglected snapshots to ensure a comparable calculation of costs.
This method assumes the chosen snapshots to be represenative for the next n-th
snapshots in the time series.

Snapshot-Clustering
^^^^^^^^^^^^^^^^^^^
This method aggregate given time series for various time intervals like i.e.
days using the `tsam <https://pypi.org/project/tsam>`_ package. Contrary to
snapshot skipping, this approach averages a certain period of snapshots
instead of choosing a representative snapshot.


Storage expansion
-----------------
To evaluate the amount of storage units in future energy systems, the possible
installation of new storage units at every node in the network is allowed. The
size and operation of these storages are part of the optimization problem.

Two types of storage technologies are considered - batteries and hydrogen in
underground caverns. Li-Ion battery storages as representatives for short-term
(several hours) technologies, which can be installed at every node.
Underground hydrogen storages represent long-term or seasonal (weeks)
technologies and can be build at every node with appropriate salt formations
in the underground. The storage parameters for both types are reached by
[Acatech2015]_, the information about salt formations are given by [BGR]_.


Grid expansion
--------------
The grid expansion is realized by extending the capacities of existing
lines and substations. These capacities are regarded as part of the
optimization problem, whereby the possible extension is unlimited. With respect
to the different voltage levels and lengths MVA-specific costs are considered
in the linear optimization of the power flow. Besides, several planned grid
expansion scenarios from the German grid development plan can be considered as
possible additional power lines.


Miscellaneous Features
--------
Several features were developed to enhance the functionality of eTraGo. As
appropriate computer setting, the 'solver_options' and a 'generator_noise' are
possible arguments. The latter adds a reproducible small random noise to the
marginal costs of each generator in order to prevent an optima plateau. The
specific solver options depend on the applied solver like for example Gurobi,
CPLEX or GLPK. Considering reproducibility, the 'load_cluster' argument
enables to load a former calculated clustered network. Besides,
'line_grouping' provides a grouping of lines which connect the same buses and
the 'branch_capacity_factor' adds a factor to adapt all line capacities. The
'load_shedding' argument is used for debugging complex grids in order to avoid
infeasibilities. It introduces a very expensive generator at each bus to meet 
the demand. When optimizing storage units and grid expansion without limiting
constraints, the need for load shedding should not be existent. The
'minimize_loading' argument forces to minimize the loading of the lines next
to the costs. 'Parallelization' provides the opportunity to devide the 
optimization problem into a given number of sub-problems. For a group of
snapshots the problem will be solved separately. This functionality can 
only be used for problems which do not have dependencies from one snapshot 
to another. Therefore this option can not be used
with the optimization of storage units due to their state of charge.


References
==========

.. [NEP2015] Übertragungsnetzbetreiber Deutschland. (2015).:
    *Netzentwicklungsplan Strom 2025*, Version 2015, 1. Entwurf, 2015.
    (`<https://www.netzentwicklungsplan.de/sites/default/files
    /paragraphs-files/NEP_2025_1_Entwurf_Teil1_0_0.pdf>`_)

.. [coastdat-2] coastDat-2 (2017).:
    Hindcast model `<http://www.coastdat.de/data/index.php.en>`_

.. [ehighway2050] e-Highway2050. (2015).:
    e-HIGHWAY 2050 Modular Development Plan of the Pan-European Transmission
    System 2050 - database per country.  Retrieved from 
    (`<http://www.e-highway2050.eu/fileadmin/documents/Results
    /e-Highway_database_per_country-08022016.xlsx>`_)

.. [Acatech2015] 'Flexibilitätskonzepte für die Stromversorgung 2050 
    `<www.acatech.de/fileadmin/user_upload/Baumstruktur_nach_Website/Acatech>`_'

.. [BGR] 'Salzstruktur in Norddeutschland <>'_. 2015.:
         Data provided by the Federal Institute for Geosciences and Natural
         Resources (Bundesanstalt für Geowissenschaften und Rohstoffe, BGR)
