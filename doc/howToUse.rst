.. _HowToUse:
==================
How to use eTraGo?
==================

After you installed eTraGo you would typically start optimization runs by
executing the ‘appl.py’ wich is situated in 
``./eTrago/etrago/`` (e.g by ``python3 appl.py``).

The ‘appl.py’ is used as a simple user interface. Here
parameters, calculation methods and scenario settings are set in a python
dictionary called 'args'. It is crucial to understand these parameters.
For example some of them contradict the usage of others. 
You find the documentation of all defined parameters from the 'args' here:
:meth:`etrago.appl.etrago`.

The appl.py contains the etrago(args) function which uses the
defined 'args' dictionary to start the desired calculation.

Afterwards a PyPSA network will contain all results. You can use
several plotting functions from the :meth:`etrago.tools.plot` in order
to visualize the results. For example 
the :meth:`etrago.tools.plot.plot_line_loading` plots
the relative line loading in % of all AC lines and DC links of the network.

To save the results you can use an interface to the oedb or write them
simply to csv files. These functionalites can be specified 
also in :meth:`etrago.appl.etrago`.


.. _Examples:
Examples and tutorial notebooks
===============================



.. toctree::
   :maxdepth: 7

   OpenMod  <https://github.com/openego/eGo/blob/master/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>
