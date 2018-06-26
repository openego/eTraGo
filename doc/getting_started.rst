Installation
============


Run:

    ```
    git clone https://github.com/openego/eTraGo
    ```

Create a virtualenvironment (where you like it) and activate it:

   ```
   virtualenv -p python3 venv
   source venv/bin/activate
   ```

With your activated environment `cd` to the cloned directory and run:

    ```
    pip install -e eTraGo
    ```

This will install all needed packages into your environment. Now you should be ready to go.

Installation for windows users
-----------------------------------

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

- open the new folder .oemof and use the editor to create a file „config.ini“ and insert the following lines, when you just calculate local or on the oedb you just need this section:

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
