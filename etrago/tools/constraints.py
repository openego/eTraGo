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
Constraints.py includes additional constraints for eTraGo-optimizations
"""

from pyomo.environ import (Var, Constraint, PositiveReals, ConcreteModel)
import numpy as np
import pandas as pd



__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


def max_line_ext(network, snapshots, share=1.01):

    """
    Sets maximal share of overall network extension
    as extra functionality in LOPF

    Parameters
    ----------
    share: float
        Maximal share of network extension in p.u.
    """

    lines_snom = network.lines.s_nom.sum()
    links_pnom = network.links.p_nom.sum()

    def _rule(m):

        lines_opt = sum(m.passive_branch_s_nom[index]
                        for index
                        in m.passive_branch_s_nom_index)

        links_opt = sum(m.link_p_nom[index]
                        for index
                        in m.link_p_nom_index)

        return (lines_opt + links_opt) <= (lines_snom + links_pnom) * share
    network.model.max_line_ext = Constraint(rule=_rule)

def min_renewable_share(network, snapshots, share=0.72):
    """
    Sets minimal renewable share of generation as extra functionality in LOPF

    Parameters
    ----------
    share: float
        Minimal share of renewable generation in p.u.
    """
    renewables = ['wind_onshore', 'wind_offshore',
                  'biomass', 'solar', 'run_of_river']

    res = list(network.generators.index[
            network.generators.carrier.isin(renewables)])

    total = list(network.generators.index)
    snapshots = network.snapshots

    def _rule(m):
        """
        """
        renewable_production = sum(m.generator_p[gen, sn]
                                      for gen
                                      in res
                                      for sn in snapshots)
        total_production = sum(m.generator_p[gen, sn]
                               for gen in total
                               for sn in snapshots)

        return (renewable_production >= total_production * share)
    network.model.min_renewable_share = Constraint(rule=_rule)

def cross_border_flow(network, snapshots, export_per_load=[0,0.2]):
    """
    Limit sum of border_flows in snapshots

    Parameters
    ----------
    export_per_load: Array of minimum and maximum export from Germany to other 
                    countries (postive: export, negative: import)
                    in percent of german loads in all snapshots

    """
# Identify cross-border-lines in respect of order of buses 
    cb0 = network.lines.index[(network.lines.bus0.isin(
            network.buses.index[network.buses.country_code!='DE'])) & (
        network.lines.bus1.isin(
                network.buses.index[network.buses.country_code=='DE']))]
        
    cb1= network.lines.index[(network.lines.bus0.isin(
            network.buses.index[network.buses.country_code=='DE'])) & (
        network.lines.bus1.isin(
                network.buses.index[network.buses.country_code!='DE']))]
        
    cb0_link=network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code!='DE'])) & (
            network.links.bus1.isin(
                network.buses.index[network.buses.country_code=='DE']))]
        
    cb1_link= network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code=='DE'])) & (
            network.links.bus1.isin(
                network.buses.index[network.buses.country_code!='DE']))]
        
    snapshots = network.snapshots

    export=pd.Series(data=export_per_load)*network.loads_t.p_set[
            network.loads.index[network.loads.bus.isin(network.buses.index[
                    network.buses.country_code == 'DE'])]].sum().sum()

    def _rule_min(m):
        cb_flow=-sum(m.passive_branch_p['Line', line, sn]
                        for line in cb0
                        for sn in snapshots) + \
                sum(m.passive_branch_p['Line',line, sn]
                        for line in cb1
                        for sn in snapshots)\
                - sum(m.link_p[link, sn]
                            for link in cb0_link
                            for sn in snapshots )\
                + sum(m.link_p[link, sn]
                            for link in cb1_link
                            for sn in snapshots)
        return ((cb_flow>=export[0]))
    
    def _rule_max(m):
        cb_flow=-sum(m.passive_branch_p['Line', line, sn]
                        for line in cb0
                        for sn in snapshots) + \
                sum(m.passive_branch_p['Line',line, sn]
                        for line in cb1
                        for sn in snapshots)\
                - sum(m.link_p[link, sn]
                            for link in cb0_link
                            for sn in snapshots )\
                + sum(m.link_p[link, sn]
                            for link in cb1_link
                            for sn in snapshots)
        return ((cb_flow<=export[1]))
    
    network.model.cross_border_flows_min = Constraint(rule=_rule_min)
    network.model.cross_border_flows_max = Constraint(rule=_rule_max)

def cross_border_flows_per_country(network, snapshots,
                                   export_per_load={'AT': [0,10],
                                                       'NL': [0,20],
                                                       'CZ': [0,30],
                                                       'PL': [0,40],
                                                       'LU': [0,50], 
                                                       'DK': [0,60],
                                                       'CH': [0,70],
                                                       'FR': [0,80],
                                                       'SE': [0,90] }):
    """
    Limit sum of border_flows in snapshots per country

    Parameters
    ----------
    export_per_load: Array of minimum and maximum export from Germany to other 
                    countries (postive: export, negative: import)
                    in percent of german loads in all snapshots

    """

    snapshots=network.snapshots
    
    countries = network.buses.country_code.unique()
    export_per_country = pd.DataFrame(data=export_per_load).transpose()*\
            network.loads_t.p_set[network.loads.index[
            network.loads.bus.isin(network.buses.index[
                    network.buses.country_code == 'DE'])]].sum().sum()
    for cntr in export_per_country.index:
        if cntr in countries:
            cb0=network.lines.index[(network.lines.bus0.isin(
                    network.buses.index[network.buses.country_code==cntr])) & (
            network.lines.bus1.isin(
                network.buses.index[network.buses.country_code=='DE']))]
        
            cb1= network.lines.index[(network.lines.bus0.isin(
                    network.buses.index[network.buses.country_code=='DE'])) & (
            network.lines.bus1.isin(
                network.buses.index[network.buses.country_code==cntr]))]
        
            cb0_link=network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code==cntr])) & (
            network.links.bus1.isin(
                network.buses.index[network.buses.country_code=='DE']))]
        
            cb1_link= network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code=='DE'])) & (
            network.links.bus1.isin(
                network.buses.index[network.buses.country_code==cntr]))]
        
            def _rule_min(m):
                cb_flow=-sum(m.passive_branch_p['Line', line, sn]
                            for line in cb0
                            for sn in snapshots) + \
                        sum(m.passive_branch_p['Line',line, sn]
                            for line in cb1
                            for sn in snapshots)\
                        - sum(m.link_p[link, sn]
                            for link in cb0_link
                            for sn in snapshots )\
                        + sum(m.link_p[link, sn]
                            for link in cb1_link
                            for sn in snapshots)
                        
                return ((cb_flow>=export_per_country[cntr][0]))
            setattr(network.model,
                "min_cross_border"+cntr, Constraint(cntr, rule=_rule_min))
    
            def _rule_max(m):
                cb_flow=-sum(m.passive_branch_p['Line', line, sn]
                        for line in cb0
                        for sn in snapshots) + \
                    sum(m.passive_branch_p['Line',line, sn]
                        for line in cb1
                        for sn in snapshots)\
                        - sum(m.link_p[link, sn]
                            for link in cb0_link
                            for sn in snapshots )\
                        + sum(m.link_p[link, sn]
                            for link in cb1_link
                            for sn in snapshots)
                return ((cb_flow<=export_per_country[cntr][1]))
            setattr(network.model,
                "max_cross_border"+cntr, Constraint(cntr, rule=_rule_max))