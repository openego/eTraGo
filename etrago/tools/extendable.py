#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extendable.py defines function to set PyPSA-components extendable.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


def extendable (network, extendable, overlay_scn_name = None):
 
    if 'network' in extendable :
        network.lines.s_nom_extendable = True
        network.lines.s_nom_min = network.lines.s_nom
        network.lines.s_nom_max = float("inf")
        
        if not network.transformers.empty:
            network.transformers.s_nom_extendable = True
            network.transformers.s_nom_min = network.transformers.s_nom
            network.transformers.s_nom_max = float("inf")
            
        if not network.links.empty:
            network.links.loc.p_nom_extendable = True
            network.links.p_nom_min = network.links.p_nom
            network.links.p_nom_max = float("inf")
      
    if 'transformers' in extendable:
        network.transformers.s_nom_extendable = True
        network.transformers.s_nom_min = network.transformers.s_nom
        network.transformers.s_nom_max = float("inf")
        
    if 'storages' in extendable:       
        if network.storage_units.carrier[network.storage_units.carrier== 'extendable_storage'].any() == 'extendable_storage':
            network.storage_units.loc[network.storage_units.carrier=='extendable_storage','p_nom_extendable'] = True
            
    if 'generators' in extendable:       
        network.generators.p_nom_extendable = True
        network.generators.p_nom_min = network.generators.p_nom
        network.generators.p_nom_max = float("inf")
        
# Extension settings for extension-NEP 2305 scenarios
        
    if 'NEP Zubaunetz' in extendable:
       network.lines.loc[(network.lines.project != 'EnLAG') & (network.lines.scn_name == 'extension_' + overlay_scn_name), 's_nom_extendable'] = True
       network.transformers.loc[(network.transformers.project != 'EnLAG') & (network.transformers.scn_name == ('extension_' + overlay_scn_name)), 's_nom_extendable'] = True      
       network.links.loc[network.links.scn_name == ('extension_' + overlay_scn_name), 'p_nom_extendable'] = True
      
        
    if 'overlay_network' in extendable:
        network.lines.loc[network.lines.scn_name == ('extension_' + overlay_scn_name), 's_nom_extendable' ] = True
        network.links.loc[network.links.scn_name == ('extension_' + overlay_scn_name), 'p_nom_extendable'] = True
        network.transformers.loc[network.transformers.scn_name == ('extension_' + overlay_scn_name), 's_nom_extendable'] = True
        
    if 'overlay_lines' in extendable:
        network.lines.loc[network.lines.scn_name == ('extension_' + overlay_scn_name), 's_nom_extendable' ] = True
        network.links.loc[network.links.scn_name == ('extension_' + overlay_scn_name), 'p_nom_extendable'] = True
        network.lines.loc[network.lines.scn_name == ('extension_' + overlay_scn_name), 'capital_cost'] = network.lines.capital_cost +( 2 * 14166 )
    
    return network

def clean_snom(network):
        network.lines['s_nom_min'] = network.lines['s_nom']
        network.lines['s_nom_extendable'] = True
        network.lines['s_nom_max'] = 1000000
        network.lines['capital_cost'] = 1800000
        network.transformers['s_nom_min'] = network.transformers['s_nom']
        network.transformers['s_nom_extendable'] = True
        network.transformers['s_nom_max'] = 1000000
        network.transformers['capital_cost'] = 1800000
        
def cleaned_snom_to_csv(network, capacity_factor):
    #lines
    diff_lines = round((network.lines['s_nom_opt']-network.lines['s_nom']), 0)
    index_lines = diff_lines.iloc[list(diff_lines.nonzero()[0])].index
    round(network.lines['s_nom_opt'].loc[index_lines]/
          capacity_factor+0.5, 0).to_csv('lines_opt.csv')
    #transformers
    diff_transformers = round((network.transformers['s_nom_opt']-network.transformers['s_nom']), 0)
    index_transformers = diff_transformers.iloc[list(diff_transformers.nonzero()[0])].index
    round(network.transformers['s_nom_opt'].loc[index_transformers]/
          capacity_factor+0.5, 0).to_csv('transformers_opt.csv')