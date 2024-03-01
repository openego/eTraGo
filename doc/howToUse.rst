.. _HowToUse:
==================
How to use eTraGo?
==================

After you installed eTraGo you would typically start optimization runs by
executing the ‘appl.py’ which is situated in 
``./eTrago/etrago/`` (e.g by ``python3 appl.py`` from the terminal).

eTraGo doesn't have a graphical user interface, 
the ‘appl.py’ is used as a simple user interface which can be edited with 
the preferred python-editor.
Here parameters, calculation methods and scenario settings are set in a python
dictionary called 'args'. 
To run the desired calculation, it is crucial to understand these parameters. 
In addition, some of them contradict the usage of others.
You find the documentation of all defined parameters from the 'args' here:
:func:`etrago.appl.run_etrago`.

Alternatively, the 'args' dictionary can be edited in a json-file.
Then the path to the json-file has to be set in the initilization of the 
Etrago-object (:class:`etrago.tools.network.Etrago`). Once a path is given
the 'args' dictionary within the 'appl.py' is ignored
and replaced by the 'args' of the json-file.

The appl.py contains the :func:`etrago.appl.run_etrago` function which uses the
defined 'args' dictionary to start the desired calculation.

To improve the performance of the optimization of the selected solver, 
you might want to use solver options (part of 'args'). For gurobi
the most used ones are described 
`here <https://github.com/openego/eTraGo/issues/213>`_.

For more specific or extensive changes you are highly invited
to write code and add new functionalities.

Once the calculation has finished the PyPSA network of the Etrago-object will
contain all results. Some main results (e.g. anuual system costs) are calculated
by :meth:`etrago.calc_results` and can be accesed via 'etrago.results'.
You can use several plotting functions from the :meth:`etrago.tools.plot` in order
to visualize the results. For example 
the :meth:`etrago.tools.plot.plot_grid` can be used to plot relative line loading
in % or the optimized expansion of all AC lines and DC links of the network.

To save the results you can write them to csv files. These functionalites can be
specified also in the 'args' dictionary.


.. _Examples:
Examples and tutorial notebooks
===============================



**eTraGo version 0.5.1:**
`etrago_OpenMod_Zuerich18 <https://github.com/openego/eGo/blob/master/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>`_.

**eTraGo version 0.9:**
`eTraGo_eGon_final_workshop <https://github.com/openego/eTraGo/blob/master/doc/eTraGo_tutorial_release0.9.ipynb>`_.
