
===============
Developer notes
===============


Installation for Developers
===========================

The best way is to use a virtual environment. see:

Step 2) Clone the source code from github

.. code-block::

   $ git clone https://github.com/openego/eTraGo
   $ git checkout dev

With your activated environment `cd` to the cloned directory and run:

.. code-block::

   $ pip3 install -e eTraGo/ --process-dependency-links

This will install all needed packages into your environment.
Now you should be ready to go.


Windows or Mac OSX users
************************



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
