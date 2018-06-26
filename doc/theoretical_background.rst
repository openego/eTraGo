======================
Theoretical Background
======================




Definitions and Units
=====================


.. toctree::
   :maxdepth: 2
   :titlesonly:

   Definitions and Units <definitions_units>



Data Sources
-------------

* Input and -/Output data
* Scenario definition


Methodology
-----------

* PyPSA
* Cluster
  * EHV-Clustering
    This method maps an input network to an output network with the nodes of
    the extra-high voltage level. All nodes with a voltage level below the
    extra-high voltage level are mapped to their nearest neighboring node in
    the extra-high voltage level with the
    <dijkstra algorithm>[https://en.wikipedia.org/wiki/Dijkstra's_algorithm] 
    (110 kV ---> 220,380 kV).
  * K-Means Clustering
    This <method>[https://en.wikipedia.org/wiki/K-means_clustering] maps an 
    input network to a new output network with an adjustable number of nodes
    and new coordinates. The algorithm sets these coordinates randomly and
    minimizes a certain parameter like for example the distances between old
    coordinates and their nearest neighbor in the set of new coordinates.
    The method was implemented by Hoersch et al.
    (https://arxiv.org/pdf/1705.07617.pdf) within
    <PyPSA>[https://pypsa.org/doc/index.html].
* Plots
* Features (check it)

References
----------
