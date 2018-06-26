======================
Theoretical Background
======================




Definitions and Units
=====================

* import csv



Data Sources
============

* Input and -/Output data
* Scenario definition


Methodology
===========



PyPSA
------



Data Clustering
---------------

EHV-Clustering
^^^^^^^^^^^^^^

This method maps an input network to an output network with the nodes of
the extra-high voltage level. All nodes with a voltage level below the
extra-high voltage level are mapped to their nearest neighboring node in
the extra-high voltage level with the
<dijkstra algorithm>[https://en.wikipedia.org/wiki/Dijkstra's_algorithm]
(110 kV ---> 220,380 kV).

K-Means Clustering
^^^^^^^^^^^^^^^^^^

This <method>[https://en.wikipedia.org/wiki/K-means_clustering] maps an
input network to a new output network with an adjustable number of nodes
and new coordinates. The algorithm sets these coordinates randomly and
minimizes a certain parameter like for example the distances between old
coordinates and their nearest neighbor in the set of new coordinates.
The method was implemented by Hoersch et al.
(https://arxiv.org/pdf/1705.07617.pdf) within
<PyPSA>[https://pypsa.org/doc/index.html].


Storage expansion
-----------------


Grid expansion
--------------


Features and plots
------------------


Test make a footnote test-a_



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

.. [BMWi]  text ...



Footnotes
---------

.. [#test-a] this is a footnote
