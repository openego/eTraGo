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
"""
Define class Etrago
"""

import logging
import json
import pandas as pd
from pypsa.components import Network
from egoio.tools import db
from sqlalchemy.orm import sessionmaker
from etrago import __version__
from etrago.tools.io import NetworkScenario, extension, decommissioning
from etrago.tools.utilities import (set_branch_capacity,
                                    convert_capital_costs,
                                    add_missing_components,
                                    set_random_noise,
                                    geolocation_buses,
                                    check_args,
                                    load_shedding,
                                    set_q_foreign_loads,
                                    foreign_links,
                                    crossborder_capacity)
from etrago.tools.plot import add_coordinates, plot_grid
from etrago.tools.extendable import extendable


logger = logging.getLogger(__name__)

class Etrago():
    """
    Object containing pypsa.Network including the transmission grid,
    input parameters and optimization results.

    Parameters
    ----------
    args : dict
        Dictionary including all inpu parameters.
    csv_folder_name : string
        Name of folder from which to import CSVs of network data.
    name : string, default ""
        Network name.
    ignore_standard_types : boolean, default False
        If True, do not read in PyPSA standard types into standard types
        DataFrames.
    kwargs
        Any remaining attributes to set

    Returns
    -------
    None

    Examples
    --------
    """
    def __init__(self,
                 args={},
                 csv_folder_name=None,
                 name="",
                 ignore_standard_types=False,
                 **kwargs):

        self.tool_version = __version__

        self.results = pd.DataFrame()

        if args != {}:
            self.args = args
        elif csv_folder_name is None:
            with open(csv_folder_name + '/args.json') as json_file:
                self.args = json.load(json_file)
        else:
            logger.error('Set args or csv_folder_name')

        self.network = Network(csv_folder_name, name, ignore_standard_types)

        if self.args['disaggregation'] is not None:
            self.disaggregated_network = Network(
                (csv_folder_name + '/disaggegated_network'
                 if csv_folder_name is not None
                 else csv_folder_name),
                name, ignore_standard_types)

        self.__re_carriers = ['wind_onshore', 'wind_offshore', 'solar',
                              'biomass', 'run_of_river', 'reservoir']
        self.__vre_carriers = ['wind_onshore', 'wind_offshore', 'solar']

        # Create network
        if csv_folder_name is None:
            conn = db.connection(section=args['db'])
            session = sessionmaker(bind=conn)
            self.session = session()
            check_args(self)
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
        """
        Function that adjusts the network imported from the database according
        to given input-parameters.

        Returns
        -------
        None.

        """
        add_coordinates(self.network)
        geolocation_buses(self)

        if self.args['scn_extension'] is not None:
            for i in range(len(self.args['scn_extension'])):
                extension(
                    self,
                    scn_extension=self.args['scn_extension'][i])
                geolocation_buses(self)

        add_missing_components(self.network)

        if self.args['scn_decommissioning'] is not None:
            decommissioning(self)

        if self.args['load_shedding']:
            load_shedding(self.network)
        set_random_noise(self, 0.01)
        self.network.lines['v_nom'] = self.network.lines.bus0.map(
            self.network.buses.v_nom)
        self.network.links['v_nom'] = self.network.links.bus0.map(
            self.network.buses.v_nom)
        set_q_foreign_loads(self.network, cos_phi=1)

        if self.args['foreign_lines']['carrier'] == 'DC':
            foreign_links(self.network)
            geolocation_buses(self)

        if self.args['foreign_lines']['capacity'] != 'osmTGmod':
            crossborder_capacity(
                self.network, self.args['foreign_lines']['capacity'])

        set_branch_capacity(self)

        if self.args['extendable'] != []:
            extendable(self, line_max=4)
            convert_capital_costs(self)

        if 'extendable_storage' in self.network.storage_units.carrier.unique():
            self.network.storage_units.carrier[
                (self.network.storage_units.carrier == 'extendable_storage')&
                (self.network.storage_units.max_hours == 6)] =\
                    'extendable_batterry_storage'
            self.network.storage_units.carrier[
                (self.network.storage_units.carrier == 'extendable_storage')&
                (self.network.storage_units.max_hours == 168)] = \
                    'extendable_hydrogen_storage'

    def _ts_weighted(self, timeseries):
        return timeseries.mul(self.network.snapshot_weightings, axis=0)

    # Add functions
    plot_grid = plot_grid

    def _calc_storage_expansion(self):
        return (self.network.storage_units.p_nom_opt -
                self.network.storage_units.p_nom_min
                )[self.network.storage_units.p_nom_extendable]\
                    .groupby(self.network.storage_units.carrier).sum()

    def calc_investment_cost(self):
        """
        Function tht calulates overall annualized investment costs .

        Returns
        -------
        network_costs : float
            Investments in line expansion
        storage_costs : float
            Investments in storage expansion

        """
        network = self.network
        ext_storage = network.storage_units[network.storage_units.p_nom_extendable]
        ext_lines = network.lines[network.lines.s_nom_extendable]
        ext_links = network.links[network.links.p_nom_extendable]
        ext_trafos = network.transformers[network.transformers.s_nom_extendable]
        storage_costs = 0
        network_costs = [0, 0]
        if not ext_storage.empty:
            storage_costs = (ext_storage.p_nom_opt*
                             ext_storage.capital_cost).sum()

        if not ext_lines.empty:
            network_costs[0] = ((ext_lines.s_nom_opt-ext_lines.s_nom_min
                                 )*ext_lines.capital_cost).sum()

        if not ext_links.empty:
            network_costs[1] = ((ext_links.p_nom_opt-ext_links.p_nom_min
                                 )*ext_links.capital_cost).sum()

        if not ext_trafos.empty:
            network_costs[0] = network_costs[0]+((
                ext_trafos.s_nom_opt-ext_trafos.s_nom
                )*ext_trafos.capital_cost).sum()

        return  network_costs, storage_costs

    def calc_marginal_cost(self):
        """
        Function that caluclates and returns marginal costs, considering
        generation and storage dispatch costs

        Returns
        -------
        marginal_cost : float
            Annual marginal cost in EUR

        """
        network = self.network
        gen = network.generators_t.p.mul(
            network.snapshot_weightings, axis=0).sum(axis=0).mul(
                network.generators.marginal_cost).sum()
        stor = network.storage_units_t.p.mul(
            network.snapshot_weightings, axis=0).sum(axis=0).mul(
                network.storage_units.marginal_cost).sum()
        marginal_cost = gen + stor
        return marginal_cost

    def calc_etrago_results(self):
        """
        Function that calculates main results and adds them to Etrago object.

        Returns
        -------
        None.

        """
        self.results = pd.DataFrame(columns=['unit', 'value'],
                                    index=['annual system costs',
                                           'annual_investment_costs',
                                           'annual_marginal_costs',
                                           'annual_grid_investment_costs',
                                           'ac_annual_grid_investment_costs',
                                           'dc_annual_grid_investment_costs',
                                           'annual_storage_investment_costs',
                                           'storage_expansion',
                                           'battery_storage_expansion',
                                           'hydrogen_storage_expansion',
                                           'abs_network_expansion',
                                           'rel_network_expansion'])

        self.results.unit[self.results.index.str.contains('cost')] = 'EUR/a'
        self.results.unit[self.results.index.isin([
            'storage_expansion', 'abs_network_expansion',
            'battery_storage_expansion', 'hydrogen_storage_expansion'])] = 'MW'
        self.results.unit['abs_network_expansion'] = 'MW'
        self.results.unit['rel_network_expansion'] = 'p.u.'



        self.results.value['ac_annual_grid_investment_costs'] = self.calc_investment_cost()[0][0]
        self.results.value['dc_annual_grid_investment_costs'] = self.calc_investment_cost()[0][1]
        self.results.value['annual_grid_investment_costs'] = sum(self.calc_investment_cost()[0])

        self.results.value['annual_storage_investment_costs'] = self.calc_investment_cost()[1]

        self.results.value['annual_investment_costs'] = \
            self.calc_investment_cost()[1] + sum(self.calc_investment_cost()[0])
        self.results.value['annual_marginal_costs'] = self. calc_marginal_cost()

        self.results.value['annual system costs'] = \
            self.results.value['annual_investment_costs'] + self.calc_marginal_cost()

        if 'storage' in self.args['extendable']:
            self.results.value['storage_expansion'] = \
                self._calc_storage_expansion().sum()
            self.results.value['battery_storage_expansion'] = \
                self._calc_storage_expansion()['extendable_batterry_storage']
            self.results.value['hydrogen_storage_expansion'] = \
                self._calc_storage_expansion()['extendable_hydrogen_storage']

        if 'network' in self.args['extendable']:
            self.results.value['abs_network_expansion']
