============
Installation
============
If you have a working Python3 environment, use pypi to install the latest 
eTraGo version. We highly recommend you to use a virtual environment. 
Use following pip command in order to install eTraGo:

.. code-block:: bash

  $ pip3 install eTraGo --process-dependency-links




Using virtual environment
=========================

Firstly, you create a virtual environment (where you like it) and activate it:

.. code-block:: bash

   $ virtualenv venv --clear -p python3.5
   $ source venv/bin/activate
   $ cd venv

Inside your virtual environment you can install eTraGo with the pip command.

Linux and Ubuntu
================

The Package eTraGo is tested with Ubuntu 16.04 and 18.04 inside the virtual
environments of *virtualenv*. The installation is shown above.



Windows or Mac OSX users
========================

For Windows and/or Mac OSX user we highly recommend to install and use Anaconda
for you Python3 installation. First install anaconda inclusing python 3.x
version from https://www.anaconda.com/download/ and open an anaconda prompt as
administrator and run:

.. code-block:: bash

  $ conda install pip
  $ conda config --add channels conda-forge
  $ conda install shapely

The full Documentation can be found
`on this page.<https://docs.anaconda.com/anaconda/install/>`_. We use Anaconda
with an own environment in order to reduze problems with Packages and different
versions on our system. Learn more about  (`Anacona environments
 <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_).





Setup ego.io
=============



 [oedb]

 username = YourOEDBUserName

 database = oedb

 host = oe2.iws.cs.ovgu.de

 port = 5432

 pw = YourOEDBPassword

 [local]

 username = YourLocalUserName

 database = YourLocalDatabaseName

 host = 127.0.0.1

 port = 5432

 pw = YourLocalPassword



when you just calculate local or on the oedb you just need this section
