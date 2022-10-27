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
calc_results.py defines methods to calculate results of eTraGo
"""
import os
if 'READTHEDOCS' not in os.environ:
    import time
    import logging

    import pandas as pd
    import numpy as np

    logger = logging.getLogger(__name__)

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"

def _calc_network_expansion(self): ###
        """ Function that calulates electrical network expansion in MW
        Returns
        -------
        float
            network expansion (AC lines and DC links) in MW
        """

        network = self.network

        lines = (network.lines.s_nom_opt -
                network.lines.s_nom_min
                )[network.lines.s_nom_extendable]

        ext_links = network.links[network.links.p_nom_extendable]
        ext_dc_lines = ext_links[ext_links.carrier=='DC']

        dc_links = (ext_dc_lines.p_nom_opt -
                ext_dc_lines.p_nom_min)

        return lines, dc_links

def _calc_storage_expansion(self):
        """ Function that calulates storage expansion in MW


        Returns
        -------
        float
            storage expansion in MW

        """
        return (self.network.storage_units.p_nom_opt -
                self.network.storage_units.p_nom_min
                )[self.network.storage_units.p_nom_extendable]\
                    .groupby(self.network.storage_units.carrier).sum()


def calc_investment_cost(self):
        """ Function that calulates overall annualized investment costs.

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
            network.snapshot_weightings.objective, axis=0).sum(axis=0).mul(
                network.generators.marginal_cost).sum()
        stor = network.storage_units_t.p.mul(
            network.snapshot_weightings.objective, axis=0).sum(axis=0).mul(
                network.storage_units.marginal_cost).sum()
        marginal_cost = gen + stor
        return marginal_cost

def calc_etrago_results(self):
        """ Function that calculates main results of grid optimization
        and adds them to Etrago object.

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
                                           'abs_network_expansion',                                         'abs. electrical grid expansion',
                                           'abs. electrical ac grid expansion',
                                           'abs. electrical dc grid expansion',
                                           'rel. electrical ac grid expansion',
                                           'rel. electrical dc grid expansion'])

        self.results.unit[self.results.index.str.contains('cost')] = 'EUR/a'
        self.results.unit[self.results.index.str.contains('expansion')] = 'MW'
        self.results.unit[self.results.index.str.contains('rel.')] = 'p.u.'

        self.results.value['ac_annual_grid_investment_costs'] = calc_investment_cost(self)[0][0]
        self.results.value['dc_annual_grid_investment_costs'] = calc_investment_cost(self)[0][1]
        self.results.value['annual_grid_investment_costs'] = sum(calc_investment_cost(self)[0])

        self.results.value['annual_storage_investment_costs'] = calc_investment_cost(self)[1]

        self.results.value['annual_investment_costs'] = \
            calc_investment_cost(self)[1] + sum(calc_investment_cost(self)[0])
        self.results.value['annual_marginal_costs'] = calc_marginal_cost(self)

        self.results.value['annual system costs'] = \
            self.results.value['annual_investment_costs'] + calc_marginal_cost(self)

        '''if 'storage' in self.args['extendable']:
            self.results.value['storage_expansion'] = \
                _calc_storage_expansion(self).sum()
            self.results.value['battery_storage_expansion'] = \
                _calc_storage_expansion(self)['extendable_battery_storage']
            self.results.value['hydrogen_storage_expansion'] = \
                _calc_storage_expansion(self)['extendable_hydrogen_storage']'''

        if not self.network.lines[self.network.lines.s_nom_extendable].empty:

            self.results.value['abs. electrical ac grid expansion'] = _calc_network_expansion(self)[0].sum()
            self.results.value['abs. electrical dc grid expansion'] = _calc_network_expansion(self)[1].sum()
            self.results.value['abs. electrical grid expansion'] = self.results.value['abs. electrical ac grid expansion'] + self.results.value['abs. electrical dc grid expansion']

            ext_lines = self.network.lines[self.network.lines.s_nom_extendable]
            ext_links = self.network.links[self.network.links.p_nom_extendable]
            ext_dc_lines = ext_links[ext_links.carrier=='DC']

            self.results.value['rel. electrical ac grid expansion'] = (_calc_network_expansion(self)[0].sum() / ext_lines.s_nom.sum()) * 100
            self.results.value['rel. electrical dc grid expansion'] = (_calc_network_expansion(self)[1].sum() / ext_dc_lines.p_nom.sum()) * 100