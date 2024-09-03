.. _HowToUse:
==================
How to use eTraGo
==================

Once the *eTraGo* application has been installed, optimization runs can be initiated by executing the ``appl.py`` script, which is located in the ``./eTrago/etrago/`` directory. This may be done, for example, by entering the command ``python3 appl.py`` in the terminal.

The ``appl.py`` file presents the user interface and may be edited with the preferred Python editor. Within the ``appl.py`` module, the scenario settings, parameters and calculation methods are defined within a Python dictionary, referred to as ``args``. It is important to comprehend the parameters in order to execute the desired calculation. It should be noted that some parameters are mutually exclusive, and thus, their usage must be carefully considered. Further information can be found in the subsequent section (Section `Functionalities`_), or in the documentation of all defined parameters from the ``args`` dictionary, accessible in the function :func:`etrago.appl.run_etrago<etrago.appl.run_etrago>`.

As an alternative approach, the ``args`` dictionary can be modified through the use of a JSON-file.
The path to the JSON-file must be specified during the initialization of the ``Etrago object`` (of the class ``Etrago.network.Etrago``). Once a path is provided, the ``args`` dictionary in the ``appl.py`` is disregarded and substituted with the dictionary from the JSON-file.

The ``appl.py`` contains the function :func:`etrago.appl.run_etrago<etrago.appl.run_etrago>` which uses the
defined ``args`` dictionary to start the desired calculation.

In order to enhance the efficacy of the optimization process using the selected solver, it may be beneficial to consider the utilisation of solver options (which form part of the ``args`` dictionary). 
For Gurobi, the some preferrable settings are outlined in Section `Functionalities`_.

For more specific or extensive changes you are kindly invited
to write code and add new functionalities. Please see Section `Contributing`_.

Once the calculation has finished, the ``Etrago-object`` will contain all of the resulting data. Some principal results (e.g. annual system costs) are calculated by the function :meth:`etrago.calc_results<etrago.analyze.calc_results.calc_etrago_results>` and can be accessed via ``etrago.results``. Additionally, several plotting functions are available (:meth:`etrago..analyze.plot<etrago.analyze.plot>`). 
To save the results, you can write them to csv files. You can specify this export within the ``args``.


.. _Examples:
Examples and Tutorial Notebooks
===============================

The following links provide examples and tutorial notebooks:

**eTraGo version 0.5.1:**
`etrago_OpenMod_Zuerich18 <https://github.com/openego/eGo/blob/master/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>`_.

**eTraGo version 0.9:**
`eTraGo_eGon_final_workshop <https://github.com/openego/eTraGo/blob/master/doc/eTraGo_tutorial_release0.9.ipynb>`_.
