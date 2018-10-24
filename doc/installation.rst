============
Installation
============
eTraGo is designed as a Python package therefore it is mandatory to have
`Python 3 <https://www.python.org/downloads/.>`_ installed. If you have a
working Python3 environment, use pypi to install the latest eTraGo version.
We highly recommend you to use a virtual environment. Use following pip
command in order to install eTraGo:

.. code-block:: bash

  $ pip3 install eTraGo --process-dependency-links



Using a virtual environment
===========================


Before installing eTraGo, 
you create a virtual environment (where you like it) and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.5
   $ source venv/bin/activate
   $ cd venv

Inside your activated virtual environment you can 
install eTraGo with the pip command, as previously explained.

Linux and Ubuntu
================

The Package eTraGo is tested with Ubuntu 16.04 and 18.04 inside the virtual
environments of `virtualenv <https://virtualenv.pypa.io/en/stable/>`_.
The installation is shown above.



Windows or Mac OSX users
========================

For Windows and/or Mac OSX user we highly recommend to install and use Anaconda
for you Python3 installation. First install anaconda inclusing python 3.5 or
higher version from https://www.anaconda.com/download/ and open an anaconda
prompt as administrator and run:

.. code-block:: bash

  $ conda install pip
  $ conda config --add channels conda-forge
  $ conda install shapely
  $ pip3 install eTraGo --process-dependency-links

The full Documentation can be found
`on this page <https://docs.anaconda.com/anaconda/install/>`_ . We use Anaconda
with an own environment in order to reduze problems with Packages and different
versions on our system. Learn more about
`Anacona <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
environments.



Setup database connection
=========================
The package `ego.io <https://github.com/openego/ego.io>`_ will be installed
automatically when eTraGo is installed. The ``egoio``
gives you python SQL-Alchemy representations of
the `OpenEnergy-Database(oedb) <https://openenergy-platform.org/dataedit/>`_ 
and access to it by using the
`oedialect <https://github.com/openego/oedialect>`_, which is a SQL-Alchemy binding
Python package for the REST-API used by the OpenEnergy Platform (OEP).

In order to connect eTraGo via the oedialect with the oedb you
have to create an account at
`openenergy-platform.org/login <http://openenergy-platform.org/login/>`_. 
You can name the `'db' <https://github.com/openego/eTraGo/blob/42a8629e966c866af2d228d49fb7234a6cc38cfe/etrago/appl.py#L105>`_
argument of the 'args' of the :func:`etrago.appl.etrago`
as you wish. Once the :func:`etrago.appl.etrago` is executed you will be asked 
to enter how you want to connect to which database. If you want to use 
the oedialect enter the following connection parameter. For <username> and
<token> you have to take your credentials which you obtained by registering
at `openenergy-platform.org/login <http://openenergy-platform.org/login/>`_.

Your API access / login data will be saved in the folder ``.egoio`` in the file
``config.ini``. Consequently, in the config.ini you can also change 
your connection parameters or add new ones.
In the following you can see how the config.ini looks like when you use the
oedialect, a local postgresql database or the old psycopg2 developer connection.


oedialect connection
--------------------

.. code-block:: desktop

  [oedb]
  dialect  = oedialect
  username = <username>
  database = oedb
  host     = openenergy-platform.org
  port     = 80
  password = <token>


Local database connection
-------------------------

.. code-block:: desktop

   [local]
   username = YourOEDBUserName
   database = YourLocalDatabaseName
   host = localhost or 127.0.0.1
   port = 5433
   pw = YourLocalPassword



Old developer connection
-------------------------

.. code-block:: desktop

  [oedb]
  username = YourOEDBUserName
  database = oedb
  host = oe2.iws.cs.ovgu.de
  port = 5432
  pw = YourOEDBPassword
