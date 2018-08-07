
===============
Developer notes
===============


Installation for Developers
===========================


.. note::
      Installation is primarly tested on (Ubuntu like) linux OS.

1. If you like, create a virtual environment (where you like it) and activate it (if you do not use venv start with 2.):

.. code-block:: bash

   $ virtualenv --clear -p python3.5  etrago``
   $ cd etrago/
   $ source bin/activate

2. Clone the source code from github

.. code-block::

   $ git clone https://github.com/openego/eTraGo
   $ git checkout dev

With your activated environment `cd` to the cloned directory and run:

.. code-block::

   $ pip3 install -e eTraGo/ --process-dependency-links

This will install all needed packages into your environment.
Now you should be ready to go.

