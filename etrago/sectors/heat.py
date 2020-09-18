# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
import numpy as np
import pandas as pd

def _add_heat_sector(etrago):
    
    # Import heat buses, loads, links and generators from db and add to electrical network 
    pass

def _try_add_heat_sector(etrago):
    
    for i in etrago.network.buses.index:
        etrago.network.add("Bus",
                           "heat bus {}".format(i),
                           carrier="heat",
                           x = etrago.network.buses.x[i]+0.5,
                           y = etrago.network.buses.y[i]+0.5)

        etrago.network.add("Bus",
                           "heat store bus {}".format(i),
                           carrier="heat",
                           x = etrago.network.buses.x[i]+0.5,
                           y = etrago.network.buses.y[i]+0.5)

        etrago.network.add("Link",
                           "heat pump {}".format(i),
                           bus0 = i,
                           bus1 = "heat bus {}".format(i),
                           efficiency = 3,
                           p_nom = 4)
        
        etrago.network.add("Load",
                           "heat load {}".format(i),
                           bus = "heat bus {}".format(i),
                           p_set = np.random.rand(len(etrago.network.snapshots))*10)
        

        etrago.network.add("Generator", 
                           "solar thermal {}".format(i),
                           bus = "heat bus {}".format(i),
                           p_max_pu = np.random.rand(len(etrago.network.snapshots)),
                           p_nom = 2)
        
        etrago.network.add("Link",
                           "heat store charger {}".format(i),
                           bus0 = "heat bus {}".format(i),
                           bus1 = "heat store bus {}".format(i),
                           efficiency = 1,
                           p_nom = 4)

        etrago.network.add("Link",
                           "heat store discharger {}".format(i),
                           bus0 = "heat store bus {}".format(i),
                           bus1 = "heat bus {}".format(i),
                           efficiency = 1,
                           p_nom = 4) 

        etrago.network.add("Store",
                           "heat store {}".format(i),
                           bus ="heat store bus {}".format(i),
                           e_nom = 40)