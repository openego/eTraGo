============
Installation
============

*eTraGo* is implemented as a Python package; therefore, Python 3 
must be installed on your system (see the 
`Python 3 Download Page <https://www.python.org/downloads/.>`_). 
The current version of *eTraGo* is compatible with Python 3.10 or higher.

If you already have a working Python3 environment, use *pypi* to install the latest
*eTraGo* version. We highly recommend using a virtual environment.
Use the following command in order to install *eTraGo*.

.. code-block:: bash

  $ pip3 install eTraGo


Using a Virtual Environment
===========================

Before installing *eTraGo*, create a virtual environment in your preferred location and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.10
   $ source venv/bin/activate
   $ cd venv

Inside your activated virtual environment you can 
install *eTraGo* with the pip command, as previously explained.


Linux and Ubuntu
================

The package *eTraGo* is tested with Ubuntu 20.04 and 22.04 inside the virtual
environments of `virtualenv <https://virtualenv.pypa.io/en/stable/>`_.
The installation is shown above.


Windows or Mac OSX
==================

For Windows and macOS users, we highly recommend installing and using *Anaconda*
for your Python 3 environment. 

First, download and install *Conda* (including Python 3.10 or higher) 
from the `Anaconda download page <https://www.anaconda.com/download/>`_.
Then, open an *Anaconda Prompt* as an administrator and run the following command:

.. code-block:: bash

  $ conda config --add channels conda-forge
  $ conda create -n etrago_env python=3.10
  $ conda activate etrago_env
  $ pip install eTraGo

The full installation documentation can be found
`on this page <https://docs.anaconda.com/anaconda/install/>`_.
We use *Anaconda* with a dedicated environment to minimize issues related to
package dependencies and version conflicts on the system.
Learn more about *Anaconda environments* in the
`official Conda user guide <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.


Setup Database Connection
=========================

The *eTraGo* module
`db <https://github.com/openego/eTraGo/blob/dev/etrago/tools/db.py>`_
provides Python SQLAlchemy representations of the
`OpenEnergy-Database (oedb) <https://openenergy-platform.org/dataedit/>`_
and allows access to it via
`oedialect <https://github.com/openego/oedialect>`_, a SQLAlchemy binding
Python package for the REST API used by the
`Open Energy Platform (OEP) <https://openenergy-platform.org/>`_.

In the future, the data will be available on the main *OEP,* and the access settings will be updated accordingly.  
For now, the data for the scenarios `eGon2035` and `eGon2035_lowflex` is stored on a *test OEP*.  
To access this data, you can select `'test-oep'` in the
`db <https://github.com/openego/eTraGo/blob/7e6719b8f71c78ea8f8312d74558e59c96c1b3a2/etrago/appl.py#L51>`_
argument of the `'args'` parameter of the :func:`etrago.appl.etrago` function.  
No account or additional information is required.

Local Database Connection
-----------------------------

As an alternative, you can use a local database.  
You may assign any name you like to the
`'db' <https://github.com/openego/eTraGo/blob/7e6719b8f71c78ea8f8312d74558e59c96c1b3a2/etrago/appl.py#L51>`_
argument of the `'args'` parameter of the :func:`etrago.appl.etrago` function.  

When :func:`etrago.appl.etrago` is executed, you will be prompted to specify how to connect to the desired database.
Your API access and login credentials will be stored in the folder ``.etrago_database`` in the file ``config.ini``.  
You can also edit the ``config.ini`` to modify existing connection parameters or add new ones.  
The example below shows the structure of ``config.ini`` when using a local PostgreSQL database.

.. code-block:: desktop

   [local]
   username = YourLocalUserName
   database = YourLocalDatabaseName
   host = localhost or 127.0.0.1
   port = YourDatabasePort
   pw = YourLocalPassword
   
Once a connection has been created (and saved in ``config.ini``), you will no longer need to re-enter the connection parameters.  
The software will automatically use the parameters corresponding to the entry specified in the
`'db' <https://github.com/openego/eTraGo/blob/7e6719b8f71c78ea8f8312d74558e59c96c1b3a2/etrago/appl.py#L51>`_ argument.
  
Installation for Developers
===========================

.. note::
      Installation has been primarily tested on Linux-based operating systems (e.g., Ubuntu).

1. If you wish, create a virtual environment in your preferred location and activate it. 
    If you are not using ``venv``, start with step 2.

.. code-block:: bash

   $ virtualenv --clear -p python3.10  etrago``
   $ cd etrago/
   $ source bin/activate
   
2. Clone the source code from *GitHub*:

.. code-block:: bash

   $ git clone https://github.com/openego/eTraGo

You can checkout to the dev branch and create new feature branches.
For the correct work-flow, please mind the 
`Dreissen Branching Model <https://nvie.com/posts/a-successful-git-branching-model/>`_

3. Use the following code to install *eTraGo* directly from the cloned repository:

.. code-block:: bash

   $ pip3 install -e /path/to/eTraGo/
