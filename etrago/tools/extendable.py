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
Extendable.py defines function to set PyPSA-components extendable.
"""
from etrago.tools.utilities import (
        set_line_costs,
        set_trafo_costs,
        convert_capital_costs,
        find_snapshots,
        buses_by_country)

from etrago.cluster.snapshot import snapshot_clustering

import numpy as np

import time

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


def extendable(network, args):

    if 'network' in args['extendable']:
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

        network = set_line_costs(network)
        network = set_trafo_costs(network)
    
    if 'german_network' in args['extendable']:
        buses = network.buses[~network.buses.index.isin(
                buses_by_country(network).index)]
        network.lines.loc[(network.lines.bus0.isin(buses.index)) &
                          (network.lines.bus1.isin(buses.index)),
                          's_nom_extendable'] = True
        network.lines.loc[(network.lines.bus0.isin(buses.index)) &
                          (network.lines.bus1.isin(buses.index)),
                          's_nom_min'] = network.lines.s_nom
        network.lines.loc[(network.lines.bus0.isin(buses.index)) &
                          (network.lines.bus1.isin(buses.index)),
                          's_nom_max'] = float("inf")
        
        if not network.transformers.empty:
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index),'s_nom_extendable'] = True
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index),'s_nom_min'] = network.transformers.s_nom
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index),'s_nom_max'] = float("inf")

        if not network.links.empty:
            network.links.loc[(network.links.bus0.isin(buses.index)) &
                              (network.links.bus1.isin(buses.index)),
                              'p_nom_extendable'] = True
            network.links.loc[(network.links.bus0.isin(buses.index)) &
                              (network.links.bus1.isin(buses.index)),
                          'p_nom_min'] = network.links.p_nom
            network.links.loc[(network.links.bus0.isin(buses.index)) &
                              (network.links.bus1.isin(buses.index)),
                          'p_nom_max'] = float("inf")
            
        network = set_line_costs(network)
        network = set_trafo_costs(network)
     
        
    if 'foreign_network' in args['extendable']:
        buses = network.buses[network.buses.index.isin(
                buses_by_country(network).index)]
        network.lines.loc[network.lines.bus0.isin(buses.index) |
                          network.lines.bus1.isin(buses.index) ,
                          's_nom_extendable'] = True
        network.lines.loc[network.lines.bus0.isin(buses.index) |
                          network.lines.bus1.isin(buses.index),
                          's_nom_min'] = network.lines.s_nom
        network.lines.loc[network.lines.bus0.isin(buses.index) |
                          network.lines.bus1.isin(buses.index),
                          's_nom_max'] = float("inf")
        
        if not network.transformers.empty:
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index) | network.transformers.bus1.isin(
                    buses.index) ,'s_nom_extendable'] = True
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index) | network.transformers.bus1.isin(
                    buses.index) ,'s_nom_min'] = network.transformers.s_nom
            network.transformers.loc[network.transformers.bus0.isin(
                    buses.index) | network.transformers.bus1.isin(
                    buses.index) ,'s_nom_max'] = float("inf")

        if not network.links.empty:
            network.links.loc[(network.links.bus0.isin(buses.index)) |
                              (network.links.bus1.isin(buses.index)),
                          'p_nom_extendable'] = True
            network.links.loc[(network.links.bus0.isin(buses.index)) |
                              (network.links.bus1.isin(buses.index)),
                          'p_nom_min'] = network.links.p_nom
            network.links.loc[(network.links.bus0.isin(buses.index)) |
                              (network.links.bus1.isin(buses.index)),
                          'p_nom_max'] = float("inf")
            
        network = set_line_costs(network)
        network = set_trafo_costs(network)
        

    if 'transformers' in args['extendable']:
        network.transformers.s_nom_extendable = True
        network.transformers.s_nom_min = network.transformers.s_nom
        network.transformers.s_nom_max = float("inf")
        network = set_trafo_costs(network)

    if 'storages' in args['extendable'] or 'storage' in args['extendable']:
        if network.storage_units.\
            carrier[network.
                    storage_units.carrier ==
                    'extendable_storage'].any() == 'extendable_storage':
            network.storage_units.loc[network.storage_units.carrier ==
                                      'extendable_storage',
                                      'p_nom_extendable'] = True

    if 'generators' in args['extendable']:
        network.generators.p_nom_extendable = True
        network.generators.p_nom_min = network.generators.p_nom
        network.generators.p_nom_max = float("inf")

    # Extension settings for extension-NEP 2035 scenarios
    if 'NEP Zubaunetz' in args['extendable']:
        for i in range(len(args['scn_extension'])):
            network.lines.loc[(network.lines.project != 'EnLAG') & (
            network.lines.scn_name == 'extension_' + args['scn_extension'][i]),
            's_nom_extendable'] = True
                    
            network.transformers.loc[(
                    network.transformers.project != 'EnLAG') & (
                            network.transformers.scn_name == (
                                    'extension_'+ args['scn_extension'][i])),
                                        's_nom_extendable'] = True
                    
            network.links.loc[network.links.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 'p_nom_extendable'] = True

    if 'overlay_network' in args['extendable']:
        for i in range(len(args['scn_extension'])):
            network.lines.loc[network.lines.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 's_nom_extendable'] = True
                
            network.links.loc[network.links.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 'p_nom_extendable'] = True
                
            network.transformers.loc[network.transformers.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 's_nom_extendable'] = True

    if 'overlay_lines' in args['extendable']:
        for i in range(len(args['scn_extension'])):
            network.lines.loc[network.lines.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 's_nom_extendable'] = True
                
            network.links.loc[network.links.scn_name == (
            'extension_' + args['scn_extension'][i]
            ), 'p_nom_extendable'] = True
                
            network.lines.loc[network.lines.scn_name == (
            'extension_' + args['scn_extension'][i]),
                'capital_cost'] = network.lines.capital_cost + (2 * 14166)
        
    network.lines.s_nom_min[network.lines.s_nom_extendable == False] =\
        network.lines.s_nom
    
    network.transformers.s_nom_min[network.transformers.s_nom_extendable == \
        False] = network.transformers.s_nom
                                   
    network.lines.s_nom_max[network.lines.s_nom_extendable == False] =\
        network.lines.s_nom
    
    network.transformers.s_nom_max[network.transformers.s_nom_extendable == \
        False] = network.transformers.s_nom

    return network


def extension_preselection(network, args, method, days = 3):
    
    """
    Function that preselects lines which are extendend in snapshots leading to 
    overloading to reduce nubmer of extension variables. 
    
    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    args  : dict
        Arguments set in appl.py
    method: str
        Choose method of selection:
            'extreme_situations' for remarkable timsteps 
            (e.g. minimal resiudual load)
            'snapshot_clustering' for snapshot clustering with number of days
    days: int
        Number of clustered days, only used when method = 'snapshot_clustering'

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    """

    weighting = network.snapshot_weightings

    if method == 'extreme_situations':
        snapshots = find_snapshots(network, 'residual load')
        snapshots = snapshots.append(find_snapshots(network, 'wind_onshore'))
        snapshots = snapshots.append(find_snapshots(network, 'solar'))
        snapshots = snapshots.drop_duplicates()
        snapshots = snapshots.sort_values()

    if method == 'snapshot_clustering':
        network_cluster = snapshot_clustering(network, how='daily', 
                                              clusters=days)
        snapshots = network_cluster.snapshots
        network.snapshot_weightings = network_cluster.snapshot_weightings

    # Set all lines and trafos extendable in network
    network.lines.loc[:, 's_nom_extendable'] = True
    network.lines.loc[:, 's_nom_min'] = network.lines.s_nom
    network.lines.loc[:, 's_nom_max'] = np.inf
    
    network.links.loc[:, 'p_nom_extendable'] = True
    network.links.loc[:, 'p_nom_min'] = network.links.p_nom
    network.links.loc[:, 'p_nom_max'] = np.inf

    network.transformers.loc[:, 's_nom_extendable'] = True
    network.transformers.loc[:, 's_nom_min'] = network.transformers.s_nom
    network.transformers.loc[:, 's_nom_max'] = np.inf

    network = set_line_costs(network)
    network = set_trafo_costs(network)
    network = convert_capital_costs(network, 1, 1)
    extended_lines = network.lines.index[network.lines.s_nom_opt >
                                         network.lines.s_nom]
    extended_links = network.links.index[network.links.p_nom_opt >
                                         network.links.p_nom]

    x = time.time()
    for i in range(int(snapshots.value_counts().sum())):
        if i > 0:
            network.lopf(snapshots[i], solver_name=args['solver'])
            extended_lines = extended_lines.append(
                    network.lines.index[network.lines.s_nom_opt >
                                        network.lines.s_nom])
            extended_lines = extended_lines.drop_duplicates()
            extended_links = extended_links.append(
                    network.links.index[network.links.p_nom_opt >
                                        network.links.p_nom])
            extended_links = extended_links.drop_duplicates()

    print("Number of preselected lines: ", len(extended_lines))

    network.lines.loc[~network.lines.index.isin(extended_lines),
                      's_nom_extendable'] = False
    network.lines.loc[network.lines.s_nom_extendable, 's_nom_min']\
        = network.lines.s_nom
    network.lines.loc[network.lines.s_nom_extendable, 's_nom_max']\
        = np.inf
        
    network.links.loc[~network.links.index.isin(extended_links),
                      'p_nom_extendable'] = False
    network.links.loc[network.links.p_nom_extendable, 'p_nom_min']\
        = network.links.p_nom
    network.links.loc[network.links.p_nom_extendable, 'p_nom_max']\
        = np.inf

    network.snapshot_weightings = weighting
    network = set_line_costs(network)
    network = set_trafo_costs(network)
    network = convert_capital_costs(network, args['start_snapshot'],
                                    args['end_snapshot'])

    y = time.time()
    z1st = (y - x) / 60

    print("Time for first LOPF [min]:", round(z1st, 2))

    return network
