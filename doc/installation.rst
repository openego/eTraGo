============
Installation
============

We highly recommend you to use a virtual environment and pip for your
installation.

Step 1) Create a virtualenvironment (where you like it) and activate it:

.. code-block:: bash

   $ virtualenv venv_etrago --clear -p python3.5
   $ source venv_etrago/bin/activate
   $ cd venv_etrago


Step 2) Use pip in order to install eTraGo

.. code-block::

   $ pip3 install eTraGo --process-dependency-links



Installation for Developers
---------------------------

Step 1) see above

Step 2) Clone the source code from github

.. code-block::

   $ git clone https://github.com/openego/eTraGo
   $ git checkout dev

With your activated environment `cd` to the cloned directory and run:

.. code-block::

   $ pip3 install -e eTraGo/ --process-dependency-links

This will install all needed packages into your environment.
Now you should be ready to go.


On Ubuntu
=========

The Package eTraGo is tested with Ubuntu 16.04 and 18.04 in the virtual
environments of *virtualenv* and *Anaconda*. See full installation is shown
above.



Windows or Mac OSX users
========================

For Windows and/or Mac OSX user we highly recommend to install and use Anaconda
as your...




- install anaconda inclusing python 3 (https://www.anaconda.com/download/)

- open an anaconda prompt as administrator and run:

   ```
   conda install pip
   ```

   ```
   conda config --add channels conda-forge
   ```

   ```
   conda install shapely
   ```

- download and install github (https://desktop.github.com)

- open GitHubDesktop and clone eTraGo from open_eGo

- open an anaconda prompt as administrator and run:

   ```
   pip install -e path/to/Github/Folder/eTraGo
   ```

   ```
   pip install pandas == 0.20.3 (version 0.21 is not working!)
   ```

- to check if everything is installed run:

   ```
   pip freeze
   ```

Using the ego.io
================


For using the session maker in eTraGo you need oemof.db:

- open an anaconda prompt as administrator and run:

   ```
   pip install oemof.db
   ```

   ```
   cd C:/Users/YourUserName
   ```

   ```
   md .oemof
   ```

- open the new folder .oemof and use the editor to create a file „config.ini“
  and insert the following lines, when you just calculate local or on the oedb
  you just need this section:

::

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
