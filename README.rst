eTraGo
======

Optimization of flexibility options for transmission grids based on PyPSA

A speciality in this context is that transmission grids are described by the 380, 220 and 110 kV in Germany. Conventionally the 110kV grid is part of the distribution grid. The integration of the transmission and 'upper' distribution grid is part of eTraGo.

The focus of optimization are flexibility options with a special focus on energy storages. Grid expansion measures are not part of this tool and will be instead part of 'eGo' https://github.com/openego/eGo


Setup dev branch
=========================


Run:

    ```
    git clone -b dev https://github.com/openego/eTraGo
    ```

To get the repository with the cluster code. 

Create a virtualenvironment (where you like it) and activate it: 

   ```
   virtualenv -p python3 venv
   source venv/bin/activate 
   ```

With you activate environment `cd` to the cloned directory and run: 

    ```
    pip install -r requirements.txt
    ```

This will install all need packages into your environment. Now you should be 
ready to go. 

Copyleft
=========================

Code licensed under "GNU Affero General Public License Version 3 (AGPL-3.0)"
It is a collaborative work with several copyright owner:
Cite as "eTraGo" © Flensburg University of Applied Sciences, Centre for Sustainable Energy Systems © Europa-Universität Flensburg, Centre for Sustainable Energy Systems © DLR Institute for Networked Energy Systems"
