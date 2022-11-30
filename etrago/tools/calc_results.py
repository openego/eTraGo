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

def _calc_store_expansion(self): ###
        """ Function that calulates store expansion in MW

        Returns
        -------
        float
            store expansion in MW

        """
        return (self.network.stores.e_nom_opt -
                self.network.stores.e_nom_min
                )[self.network.stores.e_nom_extendable]

def _calc_sectorcoupling_link_expansion(self):
        """ Function that calulates expansion of sectorcoupling links in MW

        Returns
        -------
        float
            link expansion in MW (differentiating between technologies)

        """
        ext_links = self.network.links[self.network.links.p_nom_extendable]

        links = [0, 0, 0, 0]

        l1 = ext_links[ext_links.carrier=='H2_to_power']
        l2 = ext_links[ext_links.carrier=='power_to_H2']
        l3 = ext_links[ext_links.carrier=='H2_to_CH4']
        l4 = ext_links[ext_links.carrier=='CH4_to_H2']

        links[0] = (l1.p_nom_opt-l1.p_nom_min).sum()
        links[1] = (l2.p_nom_opt-l2.p_nom_min).sum()
        links[2] = (l3.p_nom_opt-l3.p_nom_min).sum()
        links[3] = (l4.p_nom_opt-l4.p_nom_min).sum()

        return links

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

def calc_investment_cost(self):
        """ Function that calulates overall annualized investment costs.

        Returns
        -------
        network_costs : float
            Investments in line expansion (AC+DC)
        link_costs : float
            Investments in sectorcoupling link expansion
        stor_costs : float
            Investments in storage and store expansion

        """
        network = self.network

        # electrical grid: AC lines, DC lines

        network_costs = [0, 0]

        ext_lines = network.lines[network.lines.s_nom_extendable]
        ext_trafos = network.transformers[network.transformers.s_nom_extendable]
        ext_links = network.links[network.links.p_nom_extendable]
        ext_dc_lines = ext_links[ext_links.carrier=='DC']

        if not ext_lines.empty:
            network_costs[0] = ((ext_lines.s_nom_opt-ext_lines.s_nom_min
                                 )*ext_lines.capital_cost).sum()

        if not ext_trafos.empty:
            network_costs[0] = network_costs[0]+((
                ext_trafos.s_nom_opt-ext_trafos.s_nom
                )*ext_trafos.capital_cost).sum()

        if not ext_dc_lines.empty:
            network_costs[1] = ((ext_dc_lines.p_nom_opt-ext_dc_lines.p_nom_min
                                 )*ext_dc_lines.capital_cost).sum()

        # links in other sectors / coupling different sectors

        link_costs = 0

        ext_links = ext_links[ext_links.carrier!='DC']

        if not ext_links.empty:
            link_costs = ((ext_links.p_nom_opt-ext_links.p_nom_min
                                 )*ext_links.capital_cost).sum()

        # storage and store costs

        sto_costs = [0, 0]

        ext_storage = network.storage_units[network.storage_units.p_nom_extendable]
        ext_store = network.stores[network.stores.e_nom_extendable]

        if not ext_storage.empty:
            sto_costs[0] = (ext_storage.p_nom_opt*
                             ext_storage.capital_cost).sum()

        if not ext_store.empty:
            sto_costs[1] = (ext_store.e_nom_opt*
                             ext_store.capital_cost).sum()

        return  network_costs, link_costs, sto_costs

def calc_marginal_cost(self):
        """
        Function that caluclates and returns marginal costs, considering
        generation and link and storage dispatch costs

        Returns
        -------
        marginal_cost : float
            Annual marginal cost in EUR

        """
        network = self.network
        gen = network.generators_t.p.mul(
            network.snapshot_weightings, axis=0).sum(axis=0).mul(
                network.generators.marginal_cost).sum()
        link = abs(network.links_t.p0).mul(
            network.snapshot_weightings, axis=0).sum(axis=0).mul(
                network.links.marginal_cost).sum()
        stor = network.storage_units_t.p.mul(
            network.snapshot_weightings, axis=0).sum(axis=0).mul(
                network.storage_units.marginal_cost).sum()
        marginal_cost = gen + link + stor
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
                                           'annual investment costs',
                                           'annual marginal costs',
                                           'annual electrical grid investment costs',
                                           'annual ac grid investment costs',
                                           'annual dc grid investment costs',
                                           'annual links investment costs',
                                           'annual storage+store investment costs',
                                           'annual electrical storage investment costs',
                                           'annual store investment costs',
                                           'battery storage expansion',
                                           'store expansion',
                                           'H2 store expansion',
                                           'CH4 store expansion',
                                           'heat store expansion',
                                           'storage+store expansion',
                                           'fuel cell links expansion',
                                           'electrolyzer links expansion',
                                           'methanisation links expansion',
                                           'Steam Methane Reformation links expansion',
                                           'abs. electrical grid expansion',
                                           'abs. electrical ac grid expansion',
                                           'abs. electrical dc grid expansion',
                                           'rel. electrical ac grid expansion',
                                           'rel. electrical dc grid expansion'])

        self.results.unit[self.results.index.str.contains('cost')] = 'EUR/a'
        self.results.unit[self.results.index.str.contains('expansion')] = 'MW'
        self.results.unit[self.results.index.str.contains('rel.')] = 'p.u.'

        # system costs

        self.results.value['annual ac grid investment costs'] = calc_investment_cost(self)[0][0]
        self.results.value['annual dc grid investment costs'] = calc_investment_cost(self)[0][1]
        self.results.value['annual electrical grid investment costs'] = sum(calc_investment_cost(self)[0])

        self.results.value['annual links investment costs'] = calc_investment_cost(self)[1]

        self.results.value['annual electrical storage investment costs'] = calc_investment_cost(self)[2][0]
        self.results.value['annual store investment costs'] = calc_investment_cost(self)[2][1]
        self.results.value['annual storage+store investment costs'] = sum(calc_investment_cost(self)[2])


        self.results.value['annual investment costs'] = \
            sum(calc_investment_cost(self)[0]) + calc_investment_cost(self)[1] + sum(calc_investment_cost(self)[2])
        self.results.value['annual marginal costs'] = calc_marginal_cost(self)

        self.results.value['annual system costs'] = \
            self.results.value['annual investment costs'] + self.results.value['annual marginal costs']

        # storage and store expansion

        network = self.network

        if not network.storage_units[network.storage_units.p_nom_extendable].empty:

            self.results.value['battery storage expansion'] = \
                _calc_storage_expansion(self).sum()

            store = _calc_store_expansion(self)
            self.results.value['store expansion'] = store.sum()
            self.results.value['H2 store expansion'] = \
                store[store.index.str.contains('H2')].sum()
            self.results.value['CH4 store expansion'] = \
                store[store.index.str.contains('CH4')].sum()
            self.results.value['heat store expansion'] = \
                store[store.index.str.contains('heat')].sum()

            self.results.value['storage+store expansion'] = \
                self.results.value['battery storage expansion'] + self.results.value['store expansion']

        # links expansion

        if not network.links[network.links.p_nom_extendable].empty:

            links = _calc_sectorcoupling_link_expansion(self)
            self.results.value['fuel cell links expansion'] = links[0]
            self.results.value['electrolyzer links expansion'] = links[1]
            self.results.value['methanisation links expansion'] = links[2]
            self.results.value['Steam Methane Reformation links expansion'] = links[3]

        # grid expansion

        if not network.lines[network.lines.s_nom_extendable].empty:

            self.results.value['abs. electrical ac grid expansion'] = _calc_network_expansion(self)[0].sum()
            self.results.value['abs. electrical dc grid expansion'] = _calc_network_expansion(self)[1].sum()
            self.results.value['abs. electrical grid expansion'] = self.results.value['abs. electrical ac grid expansion'] + self.results.value['abs. electrical dc grid expansion']

            ext_lines = network.lines[network.lines.s_nom_extendable]
            ext_links = network.links[network.links.p_nom_extendable]
            ext_dc_lines = ext_links[ext_links.carrier=='DC']

            self.results.value['rel. electrical ac grid expansion'] = (_calc_network_expansion(self)[0].sum() / ext_lines.s_nom.sum()) * 100
            self.results.value['rel. electrical dc grid expansion'] = (_calc_network_expansion(self)[1].sum() / ext_dc_lines.p_nom.sum()) * 100