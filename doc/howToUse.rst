.. _HowToUse:
==================
How to use eTraGo?
==================

After you installed eTraGo you can run eTraGo via terminal with
``$ cd../<your path>/eTrago/etrago/`` and ``python3 appl.py``.
The program will execute following functions:

.. code-block:: python

   # execute etrago function
    network = etrago(args)
    # plots
    # make a line loading plot
    plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    storage_distribution(network)


Overview of setting arguments
=============================

The tool eTraGo is using a main python script ‘appl.py’ in which your
parameters, calculation methods and scenario settings are set in a python
dictionary called args. The documentation of the program settings can you
find here: :meth:`etrago.appl.etrago`.


.. _Examples:
Examples and tutorial notebooks
===============================



.. toctree::
   :maxdepth: 7

   OpenMod  <https://github.com/openego/eGo/blob/features/tutorial/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>
