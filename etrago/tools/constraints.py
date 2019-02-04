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
import pandas as pd


__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"
 
class Constraints:
    
    def __init__(self, args):
     self.args=args
     
    def functionality(self, network, snapshots):
        """ Add constraints to pypsa-model using extra-functionality..

        Parameters
        ----------
        network : :class:`pypsa.Network
            Overall container of PyPSA
        snapshots

        """
        

        if 'max_line_ext' in self.args['extra_functionality'].keys():
            lines_snom = network.lines.s_nom.sum()
            links_pnom = network.links.p_nom.sum()

            def _rule(m):
                lines_opt = sum(m.passive_branch_s_nom[index]
                        for index
                        in m.passive_branch_s_nom_index)

                links_opt = sum(m.link_p_nom[index]
                        for index
                        in m.link_p_nom_index)
                return (lines_opt + links_opt) <= (lines_snom + links_pnom
                       )*self.args['extra_functionality']['max_line_ext']
            network.model.max_line_ext = Constraint(rule=_rule)
        

        if 'min_renewable_share' in self.args['extra_functionality'].keys():

            renewables = ['wind_onshore', 'wind_offshore',
                  'biomass', 'solar', 'run_of_river']

            res = list(network.generators.index[
            network.generators.carrier.isin(renewables)])

            total = list(network.generators.index)
            snapshots = network.snapshots

            def _rule(m):
        
                renewable_production = sum(m.generator_p[gen, sn]
                                      for gen
                                      in res
                                      for sn in snapshots)
                total_production = sum(m.generator_p[gen, sn]
                               for gen in total
                               for sn in snapshots)

                return (renewable_production >= total_production *\
                        self.args['extra_functionality']['min_renewable_share'])
            network.model.min_renewable_share = Constraint(rule=_rule)
            
            
            if 'cross_border_flow' in self.args['extra_functionality'].keys():
            # Identify cross-border-lines in respect of order of buses 
                cb0 = network.lines.index[(network.lines.bus0.isin(
                        network.buses.index[network.buses.country_code!='DE'])
                        ) & (network.lines.bus1.isin(network.buses.index\
                        [network.buses.country_code=='DE']))]
        
                cb1= network.lines.index[(network.lines.bus0.isin(
                        network.buses.index[network.buses.country_code=='DE'])
                        ) & (network.lines.bus1.isin(network.buses.index\
                        [network.buses.country_code!='DE']))]
        
                cb0_link=network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code!='DE'])) & (
                    network.links.bus1.isin(
                    network.buses.index[network.buses.country_code=='DE']))]
        
                cb1_link= network.links.index[(network.links.bus0.isin(
                    network.buses.index[network.buses.country_code=='DE'])) & (
                    network.links.bus1.isin(
                    network.buses.index[network.buses.country_code!='DE']))]
        
                snapshots = network.snapshots

                export=pd.Series(data=self.args['extra_functionality']\
                                 ['cross_border_flow'])*network.loads_t.p_set[
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

            if 'cross_border_flow_per_country' in self.args['extra_functionality'].keys():
                snapshots=network.snapshots
    
                countries = network.buses.country_code.unique()
                export_per_country = pd.DataFrame(
                        data=self.args['extra_functionality']\
                        ['cross_border_flow_per_country']).transpose()*\
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
