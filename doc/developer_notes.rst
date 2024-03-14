
===============
Developer notes
===============


Installation for Developers
===========================


.. note::
      Installation is primarly tested on (Ubuntu like) linux OS.

1. If you like, create a virtual environment (where you like it) and activate it (if you do not use venv start with 2.):

.. code-block:: bash

   $ virtualenv --clear -p python3.10  etrago``
   $ cd etrago/
   $ source bin/activate
   
2. Clone the source code from github:

.. code-block:: bash

   $ git clone https://github.com/openego/eTraGo

You can checkout to the dev branch and create new feature branches.
For the correct work-flow, please mind the 
`Dreissen Branching Model <https://nvie.com/posts/a-successful-git-branching-model/>`_

3. Use the pip -e to install eTraGo directly from the cloned repository:

.. code-block:: bash

   $ pip3 install -e /path/to/eTraGo/

