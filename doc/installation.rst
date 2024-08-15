============
Installation
============

eTraGo is designed as a Python package therefore it is mandatory to have
`Python 3 <https://www.python.org/downloads/.>`_ installed. If you have a
working Python3 environment, use pypi to install the latest eTraGo version.
We highly recommend you to use a virtual environment. Use following pip
command in order to install eTraGo.

.. code-block:: bash

  $ pip3 install eTraGo


Using a Virtual Environment
===========================

Before installing eTraGo, 
you create a virtual environment (where you like it) and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.10
   $ source venv/bin/activate
   $ cd venv

Inside your activated virtual environment you can 
install eTraGo with the pip command, as previously explained.


Linux and Ubuntu
================

The Package eTraGo is tested with Ubuntu 16.04, 18.04, 20.04 and 22.04 inside the virtual
environments of `virtualenv <https://virtualenv.pypa.io/en/stable/>`_.
The installation is shown above.



Windows or Mac OSX
==================

For Windows and/or Mac OSX user we highly recommend to install and use Anaconda
for your Python3 installation. First install Conda including python 3.10 or
higher version from https://www.anaconda.com/download/ and open an anaconda
prompt as administrator and run:

.. code-block:: bash

  $ conda config --add channels conda-forge
  $ conda create -n etrago_env python=3.10
  $ conda activate etrago_env
  $ pip install eTraGo

The full Documentation can be found
`on this page <https://docs.anaconda.com/anaconda/install/>`_ . We use Anaconda
with an own environment in order to reduze problems with Packages and different
versions on our system. Learn more about
`Anaconda <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
environments.


Setup Database Connection
=========================
The eTraGo module `db <https://github.com/openego/eTraGo/blob/dev/etrago/tools/db.py>`_ 
gives you a python SQL-Alchemy representations of
the `OpenEnergy-Database(oedb) <https://openenergy-platform.org/dataedit/>`_ 
and access to it by using the
`oedialect <https://github.com/openego/oedialect>`_, which is a SQL-Alchemy binding
Python package for the REST-API used by the OpenEnergy Platform (OEP).

In order to connect eTraGo via the oedialect with the oedb you
have to create an account at
`openenergy-platform.org/login <http://openenergy-platform.org/login/>`_. 
You can name the `'db' <https://github.com/openego/eTraGo/blob/7e6719b8f71c78ea8f8312d74558e59c96c1b3a2/etrago/appl.py#L51>`_
argument of the 'args' of the :func:`etrago.appl.etrago`
as you wish. Once the :func:`etrago.appl.etrago` is executed you will be asked 
to enter how you want to connect to which database. If you want to use 
the oedialect enter the following connection parameter. For <username> and
<token> you have to take your credentials which you obtained by registering
at `openenergy-platform.org/login <http://openenergy-platform.org/login/>`_.

Your API access / login data will be saved in the folder ``.etrago_database`` in the file
``config.ini``. Consequently, in the config.ini you can also change 
your connection parameters or add new ones.
In the following you can see how the config.ini looks like when you use the
oedialect, a local postgresql database or the old psycopg2 developer connection.

Once you have created a connection (which is saved in the config.ini) you do not have
to enter the connection parameter again. The software will take the connection parameter
which corresponds to the entry at the `'db' <https://github.com/openego/eTraGo/blob/7e6719b8f71c78ea8f8312d74558e59c96c1b3a2/etrago/appl.py#L51>`_ argument.

oedialect Connection
--------------------

.. code-block:: desktop

  [oedb]
  dialect  = oedialect
  username = <username>
  database = oedb
  host     = openenergy-platform.org
  port     = 80
  password = <token>

Local Database Connection
-------------------------

.. code-block:: desktop

   [local]
   username = YourOEDBUserName
   database = YourLocalDatabaseName
   host = localhost or 127.0.0.1
   port = 5433
   pw = YourLocalPassword
  
  
Installation for Developers
===========================

.. note::
      Installation is primarly tested on (Ubuntu like) linux OS.

1. If you like, create a virtual environment (where you like it) and activate it (if you do not use venv start with 2.):

.. code-block:: bash

   $ virtualenv --clear -p python3.8  etrago``
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
