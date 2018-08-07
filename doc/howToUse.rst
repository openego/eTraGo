.. _HowToUse:
==================
How to use eTraGo?
==================

After you installed eTraGo you would typically start optimization runs by
executing the ‘appl.py’ wich is situated in 
``/<your path>/eTrago/etrago/`` (e.g by ``python3 appl.py``).

The ‘appl.py’ is used as a simple user interface. Here
parameters, calculation methods and scenario settings are set in a python
dictionary called 'args'. It is crucial to understand these parameters.
For example some of them contradict the usage of others. 
The documentation of all parameters being defined within the 'args' you
find here: :meth:`etrago.appl.etrago`.


.. _Examples:
Examples and tutorial notebooks
===============================



.. toctree::
   :maxdepth: 7

   OpenMod  <https://github.com/openego/eGo/blob/features/tutorial/ego/examples/tutorials/etrago_OpenMod_Zuerich18.ipynb>
