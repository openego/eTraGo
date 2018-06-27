======================
Theoretical Background
======================



Definitions and Units
=====================

eTraGo executes the Open Source software PyPSA to perform power flow 
simulations and uses their definitions and
`units<https://pypsa.org/doc/conventions.html#unit-conventions>`.



Assumptions on Data
===================

eTraGo fetches its necessary input data from the OpenEnergy Platform including
load, generation, grid and scenario-related data. More details can be found in
the `Data-Processing <https://data-processing.readthedocs.io/en/latest>`.

As overview, the Open Source grid structure is developed by processing data 
from `OpenStreetMap <https://www.openstreetmap.de>` (OSM) to obtain 
geo-referenced locations of substations and links equal or above the 110 kV
voltage level. OSM also provides information about residential, retail,
industrial and agricultural areas which is used with standardized profiles to
obtain load data. Generation data of solar and wind rely on weather data from
[coastdat-2]_. Both, load and generation data, match the annual amount for the
year 2011. eTraGo enables the investigation of three scenarios - Status Quo,
NEP 2035 and eGo100. Status Quo corresponds to the actual grid, NEP2035 
follows assumptions for the year 2035 by [NEP2015]_ and eGo100 assumes to 
operate the future energy system completely by renewables [ehighway2050]_.



Methodology
===========


PyPSA
-----
The power flow simulations are performed by the Open Source tool
`PyPSA <https://pypsa.org/doc/index.html>` with a linear approximation for the
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
`dijkstra algorithm <https://en.wikipedia.org/wiki/Dijkstra's_algorithm>`
(110 kV ---> 220,380 kV).

K-Means Clustering
^^^^^^^^^^^^^^^^^^

This `method<https://en.wikipedia.org/wiki/K-means_clustering>` maps an input
network to a new output network with an adjustable number of nodes and new
coordinates. The algorithm sets these coordinates randomly and minimizes a
certain parameter like for example the distances between old coordinates and
their nearest neighbor in the set of new coordinates. The method was
implemented by `Hoersch et al. <https://arxiv.org/pdf/1705.07617.pdf>` within
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
days using the `tsam <https://pypi.org/project/tsam>` package. Contrary to
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
tbd line_grouping? branch_capacity factor?


Features
--------
tbd (solver options, generator_noise, load_cluster, line_grouping, branch_capacity factor, load shedding)



References
==========

.. [NEP2015] Übertragungsnetzbetreiber Deutschland. (2015).:
    *Netzentwicklungsplan Strom 2025*, Version 2015, 1. Entwurf, 2015. (https://
    www.netzentwicklungsplan.de/sites/default/files/paragraphs-files/NEP_2025_
    1_Entwurf_Teil1_0_0.pdf)

.. [coastdat-2] coastDat-2 (2017).:
     Hindcast model http://www.coastdat.de/data/index.php.en

.. [FlEnS] Bunke, Wolf-Dieter, Martin Söthe, Marion Wingenbach, and Cord Kaldemeyer. 2018.:
    *“(Fl)ensburg (En)ergy (S)cenarios - open_eGo Scenarios for 2014/2035/2050.”* Open Science Framework. June 13. doi:10.17605/OSF.IO/BPF36.

.. [opsd-conv] `Open Power System Data <http://open-power-system-data.org/>`_. 2016.:
    Data provided by Open Power System Data - Data Package Conventional power plants, version 2016-10-27. Primary data from BNetzA Kraftwerksliste,
    Umweltbundesamt Datenbank Kraftwerke in Deutschland.

.. [opsd-res] `Open Power System Data <http://open-power-system-data.org/>`_. 2017.:
    Data provided by Open Power System Data - Data Package Renewable power plants, early version 2016-02-10. Primary data from BNetzA, BNetzA_PV, TransnetBW, TenneT, Amprion, 50Hertz, Netztransparenz.de, Postleitzahlen Deutschland, Energinet.dk, Energistyrelsen, GeoNames, French Ministery of the Environment, Energy and the Sea, OpenDataSoft, Urzad Regulacji Energetyki (URE)

.. [ehighway2050] e-Highway2050. (2015).:
     e-HIGHWAY 2050 Modular Development Plan of the Pan-European Transmission System 2050 - database per country.  Retrieved from http://www.e-highway2050.eu/fileadmin/documents/Results/e-Highway_database_per_country-08022016.xlsx

.. [christ2017] Christ, M. (2017).:
     Integration sozial-ökologischer Faktoren in die Energiesystemmodellierung am Beispiel von Entwicklungspfaden für den Windenergieausbau in Deutschland (PhD Thesis). Europa-Universität Flensburg.

.. [BMWi]  tbd

.. [Acatech2015] tbd

.. [BGR] tbd
