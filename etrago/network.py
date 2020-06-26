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

from pypsa.components import Network
from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import json
from etrago.tools.io import NetworkScenario

from etrago.tools.utilities import (set_branch_capacity, convert_capital_costs, 
add_missing_components, set_random_noise, geolocation_buses, check_args, load_shedding, 
set_q_foreign_loads)

from etrago.tools.plot import add_coordinates, plot_grid

from etrago.tools.extendable import (
            extendable,
            print_expansion_costs)

import logging
logger = logging.getLogger(__name__)

class Etrago():
    
    def __init__(self,
                 args={},
                 csv_folder_name=None,
                 name="",
                 ignore_standard_types=False,
                 **kwargs):
        
        if args != {}:
            self.args = args
        elif csv_folder_name!=None:
            with open(csv_folder_name + '/args.json') as f:
                self.args = json.load(f)

        else:
            logger.error('Give args or csv_folder')
        
        self.network = Network(
                csv_folder_name, name, ignore_standard_types)

        if self.args['disaggregation']!=None:
            self.disaggregated_network = Network(
                (csv_folder_name + '/disaggegated_network'
                 if csv_folder_name!=None
                 else csv_folder_name),
                name, ignore_standard_types)

        self.__renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar',
                                     'biomass', 'run_of_river', 'reservoir']

        
        if csv_folder_name==None: 
            conn = db.connection(section=args['db'])
            Session = sessionmaker(bind=conn)
            self.session = Session()
            check_args(self.args)
            self._build_network_from_db()
            self._adjust_network()


    def _build_network_from_db(self):
        self.scenario = NetworkScenario(self.session,
                                   version=self.args['gridversion'],
                                   prefix=('EgoGridPfHv' if 
                                           self.args['gridversion'] is None
                                           else 'EgoPfHv'),
                                   method=self.args['method'],
                                   start_snapshot=self.args['start_snapshot'],
                                   end_snapshot=self.args['end_snapshot'],
                                   scn_name=self.args['scn_name'])

        self.network = self.scenario.build_network()
        
        logger.info('Imported network from db')
        
    def _adjust_network(self):
        add_coordinates(self.network)
        geolocation_buses(self.network, self.session)
        add_missing_components(self.network)
        load_shedding(self.network)
        set_random_noise(self, 0.01)
        self.network.lines['v_nom'] = self.network.lines.bus0.map(
                self.network.buses.v_nom)
        self.network.links['v_nom'] = self.network.links.bus0.map(
                self.network.buses.v_nom)
        set_q_foreign_loads(self.network, cos_phi=1)
        set_branch_capacity(self) # will be replaced when using new pypsa version

        if self.args['extendable'] != []:
            extendable(self, line_max=4)
            convert_capital_costs(self)
        
        if 'extendable_storage' in self.network.storage_units.carrier.unique():
            self.network.storage_units.carrier[
                    (self.network.storage_units.carrier=='extendable_storage')&
                    (self.network.storage_units.max_hours==6)] =\
                    'extendable_batterry_storage'
            self.network.storage_units.carrier[
                    (self.network.storage_units.carrier=='extendable_storage')&
                    (self.network.storage_units.max_hours==168)] = \
                    'extendable_hydrogen_storage'
    
    def _ts_weighted(self, timeseries):
        return timeseries.mul(self.snapshot_weightings, axis = 0)
    plot_grid = plot_grid
    

        
        
    
