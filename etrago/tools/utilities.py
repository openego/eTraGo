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
Utilities.py includes a wide range of useful functions.
"""

import os
import time
from pyomo.environ import (Var, Constraint, PositiveReals, ConcreteModel)
import numpy as np
import pandas as pd
import pypsa
import json
import logging
import math
logger = logging.getLogger(__name__)


__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


def buses_of_vlvl(network, voltage_level):
    """ Get bus-ids of given voltage level(s).

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    voltage_level: list

    Returns
    -------
    list
        List containing bus-ids.
    """

    mask = network.buses.v_nom.isin(voltage_level)
    df = network.buses[mask]

    return df.index


def buses_grid_linked(network, voltage_level):
    """ Get bus-ids of a given voltage level connected to the grid.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    voltage_level: list

    Returns
    -------
    list
        List containing bus-ids.
    """

    mask = ((network.buses.index.isin(network.lines.bus0) |
             (network.buses.index.isin(network.lines.bus1))) &
            (network.buses.v_nom.isin(voltage_level)))

    df = network.buses[mask]

    return df.index

def clip_foreign(network):
    """
    Delete all components and timelines located outside of Germany.
    Add transborder flows divided by country of origin as
    network.foreign_trade.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    Returns
    -------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    """

    # get foreign buses by country
    poland = pd.Series(index=network.
                       buses[(network.buses['x'] > 17)].index,
                       data="Poland")
    czech = pd.Series(index=network.
                      buses[(network.buses['x'] < 17) &
                            (network.buses['x'] > 15.1)].index,
                      data="Czech")
    denmark = pd.Series(index=network.
                        buses[((network.buses['y'] < 60) &
                               (network.buses['y'] > 55.2)) |
                              ((network.buses['x'] > 11.95) &
                               (network.buses['x'] < 11.97) &
                               (network.buses['y'] > 54.5))].
                        index,
                        data="Denmark")
    sweden = pd.Series(index=network.buses[(network.buses['y'] > 60)].index,
                       data="Sweden")
    austria = pd.Series(index=network.
                        buses[(network.buses['y'] < 47.33) &
                              (network.buses['x'] > 9) |
                              ((network.buses['x'] > 9.65) &
                               (network.buses['x'] < 9.9) &
                               (network.buses['y'] < 47.5) &
                               (network.buses['y'] > 47.3)) |
                              ((network.buses['x'] > 12.14) &
                               (network.buses['x'] < 12.15) &
                               (network.buses['y'] > 47.57) &
                               (network.buses['y'] < 47.58)) |
                              (network.buses['y'] < 47.6) &
                              (network.buses['x'] > 14.1)].index,
                        data="Austria")
    switzerland = pd.Series(index=network.
                            buses[((network.buses['x'] > 8.1) &
                                   (network.buses['x'] < 8.3) &
                                   (network.buses['y'] < 46.8)) |
                                  ((network.buses['x'] > 7.82) &
                                   (network.buses['x'] < 7.88) &
                                   (network.buses['y'] > 47.54) &
                                   (network.buses['y'] < 47.57)) |
                                  ((network.buses['x'] > 10.91) &
                                   (network.buses['x'] < 10.92) &
                                   (network.buses['y'] > 49.91) &
                                   (network.buses['y'] < 49.92))].index,
                            data="Switzerland")
    netherlands = pd.Series(index=network.
                            buses[((network.buses['x'] < 6.96) &
                                   (network.buses['y'] < 53.15) &
                                   (network.buses['y'] > 53.1)) |
                                  ((network.buses['x'] < 5.4) &
                                   (network.buses['y'] > 52.1))].index,
                            data="Netherlands")
    luxembourg = pd.Series(index=network.
                           buses[((network.buses['x'] < 6.15) &
                                  (network.buses['y'] < 49.91) &
                                  (network.buses['y'] > 49.65))].index,
                           data="Luxembourg")
    france = pd.Series(index=network.
                       buses[(network.buses['x'] < 4.5) |
                             ((network.buses['x'] > 7.507) &
                              (network.buses['x'] < 7.508) &
                              (network.buses['y'] > 47.64) &
                              (network.buses['y'] < 47.65)) |
                             ((network.buses['x'] > 6.2) &
                              (network.buses['x'] < 6.3) &
                              (network.buses['y'] > 49.1) &
                              (network.buses['y'] < 49.2)) |
                             ((network.buses['x'] > 6.7) &
                              (network.buses['x'] < 6.76) &
                              (network.buses['y'] > 49.13) &
                              (network.buses['y'] < 49.16))].index,
                       data="France")
    foreign_buses = pd.Series()
    foreign_buses = foreign_buses.append([poland, czech, denmark, sweden,
                                          austria, switzerland,
                                          netherlands, luxembourg, france])

    network.buses = network.buses.drop(
        network.buses.loc[foreign_buses.index].index)

    # identify transborder lines (one bus foreign, one bus not) and the country
    # it is coming from
    transborder_lines = pd.DataFrame(index=network.lines[
        ((network.lines['bus0'].isin(network.buses.index) == False) &
         (network.lines['bus1'].isin(network.buses.index) == True)) |
        ((network.lines['bus0'].isin(network.buses.index) == True) &
         (network.lines['bus1'].isin(network.buses.index) == False))].index)
    transborder_lines['bus0'] = network.lines['bus0']
    transborder_lines['bus1'] = network.lines['bus1']
    transborder_lines['country'] = ""
    for i in range(0, len(transborder_lines)):
        if transborder_lines.iloc[i, 0] in foreign_buses.index:
            transborder_lines['country'][i] = foreign_buses[str(
                transborder_lines.iloc[i, 0])]
        else:
            transborder_lines['country'][i] = foreign_buses[str(
                transborder_lines.iloc[i, 1])]

    # identify amount of flows per line and group to get flow per country
    transborder_flows = network.lines_t.p0[transborder_lines.index]
    for i in transborder_flows.columns:
        if network.lines.loc[str(i)]['bus1'] in foreign_buses.index:
            transborder_flows.loc[:, str(
                i)] = transborder_flows.loc[:, str(i)]*-1

    network.foreign_trade = transborder_flows.\
        groupby(transborder_lines['country'], axis=1).sum()
                

    # drop foreign components
    network.lines = network.lines.drop(network.lines[
        (network.lines['bus0'].isin(network.buses.index) == False) |
        (network.lines['bus1'].isin(network.buses.index) == False)].index)
                            
    network.links = network.links.drop(network.links[
        (network.links['bus0'].isin(network.buses.index) == False) |
        (network.links['bus1'].isin(network.buses.index) == False)].index)
                            
    network.transformers = network.transformers.drop(network.transformers[
        (network.transformers['bus0'].isin(network.buses.index) == False) |
        (network.transformers['bus1'].isin(network.
                                           buses.index) == False)].index)
    network.generators = network.generators.drop(network.generators[
        (network.generators['bus'].isin(network.buses.index) == False)].index)
    network.loads = network.loads.drop(network.loads[
        (network.loads['bus'].isin(network.buses.index) == False)].index)
    network.storage_units = network.storage_units.drop(network.storage_units[
        (network.storage_units['bus'].isin(network.
                                           buses.index) == False)].index)

    components = ['loads', 'generators', 'lines', 'buses', 'transformers', 'links']
    for g in components:  # loads_t
        h = g + '_t'
        nw = getattr(network, h)  # network.loads_t
        for i in nw.keys():  # network.loads_t.p
            cols = [j for j in getattr(
                nw, i).columns if j not in getattr(network, g).index]
            for k in cols:
                del getattr(nw, i)[k]

    return network

def set_q_foreign_loads(network, cos_phi = 1):
    # get foreign buses by country
    poland = pd.Series(index=network.
                       buses[(network.buses['x'] > 17)].index,
                       data="Poland")
    czech = pd.Series(index=network.
                      buses[(network.buses['x'] < 17) &
                            (network.buses['x'] > 15.1)].index,
                      data="Czech")
    denmark = pd.Series(index=network.
                        buses[((network.buses['y'] < 60) &
                               (network.buses['y'] > 55.2)) |
                              ((network.buses['x'] > 11.95) &
                               (network.buses['x'] < 11.97) &
                               (network.buses['y'] > 54.5))].
                        index,
                        data="Denmark")
    sweden = pd.Series(index=network.buses[(network.buses['y'] > 60)].index,
                       data="Sweden")
    austria = pd.Series(index=network.
                        buses[(network.buses['y'] < 47.33) &
                              (network.buses['x'] > 9) |
                              ((network.buses['x'] > 9.65) &
                               (network.buses['x'] < 9.9) &
                               (network.buses['y'] < 47.5) &
                               (network.buses['y'] > 47.3)) |
                              ((network.buses['x'] > 12.14) &
                               (network.buses['x'] < 12.15) &
                               (network.buses['y'] > 47.57) &
                               (network.buses['y'] < 47.58)) |
                              (network.buses['y'] < 47.6) &
                              (network.buses['x'] > 14.1)].index,
                        data="Austria")
    switzerland = pd.Series(index=network.
                            buses[((network.buses['x'] > 8.1) &
                                   (network.buses['x'] < 8.3) &
                                   (network.buses['y'] < 46.8)) |
                                  ((network.buses['x'] > 7.82) &
                                   (network.buses['x'] < 7.88) &
                                   (network.buses['y'] > 47.54) &
                                   (network.buses['y'] < 47.57)) |
                                  ((network.buses['x'] > 10.91) &
                                   (network.buses['x'] < 10.92) &
                                   (network.buses['y'] > 49.91) &
                                   (network.buses['y'] < 49.92))].index,
                            data="Switzerland")
    netherlands = pd.Series(index=network.
                            buses[((network.buses['x'] < 6.96) &
                                   (network.buses['y'] < 53.15) &
                                   (network.buses['y'] > 53.1)) |
                                  ((network.buses['x'] < 5.4) &
                                   (network.buses['y'] > 52.1))].index,
                            data="Netherlands")
    luxembourg = pd.Series(index=network.
                           buses[((network.buses['x'] < 6.15) &
                                  (network.buses['y'] < 49.91) &
                                  (network.buses['y'] > 49.65))].index,
                           data="Luxembourg")
    france = pd.Series(index=network.
                       buses[(network.buses['x'] < 4.5) |
                             ((network.buses['x'] > 7.507) &
                              (network.buses['x'] < 7.508) &
                              (network.buses['y'] > 47.64) &
                              (network.buses['y'] < 47.65)) |
                             ((network.buses['x'] > 6.2) &
                              (network.buses['x'] < 6.3) &
                              (network.buses['y'] > 49.1) &
                              (network.buses['y'] < 49.2)) |
                             ((network.buses['x'] > 6.7) &
                              (network.buses['x'] < 6.76) &
                              (network.buses['y'] > 49.13) &
                              (network.buses['y'] < 49.16))].index,
                       data="France")
    foreign_buses = pd.Series()
    foreign_buses = foreign_buses.append([poland, czech, denmark, sweden,
                                          austria, switzerland,
                                          netherlands, luxembourg, france])
                        
    network.loads_t['q_set'][network.loads.index[network.loads.bus.astype(str)\
     .isin(foreign_buses.index)]]  =  network.loads_t['p_set'][network.loads.\
     index[network.loads.bus.astype(str).isin (foreign_buses.index)]] \
     * math.tan(math.acos(cos_phi))
     
    # temporary change x of some lines to avoid infeasibilities
    #network.lines.s_nom[network.lines.bus0.astype(str).isin(france.index)] = network.lines.s_nom * 0.7
   # network.lines.s_nom[network.lines.bus0.astype(str).isin(poland.index)] = network.lines.s_nom * 0.7
    
    # for future scenarios set all generators to PV
    #network.generators.control[network.generators.control == 'PQ'] = 'PV'
    
    return network
    
def fix_bugs_for_pf(network):


    network.loads_t['q_set'][network.loads_t['q_set'].isnull()] = 0
    
    #network.lines.x[network.lines.bus0.astype(str).isin(france.index)] = network.lines.x/10
    #network.lines.x[network.lines.bus0.astype(str).isin(poland.index)] = network.lines.x/10
    #network.lines.s_nom[network.lines.bus0.astype(str).isin(france.index)] = network.lines.s_nom/10
    #network.transformers.x[network.transformers.bus0.astype(str).isin(foreign_buses.index)] = network.transformers.x * 0.00001 
    #network.transformers.x[network.transformers.x>0.5] = network.transformers.x *0.00001
    
   
  
    return network

def connected_grid_lines(network, busids):
    """ Get grid lines connected to given buses.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    busids  : list
        List containing bus-ids.

    Returns
    -------
    :class:`pandas.DataFrame
        PyPSA lines.
    """

    mask = network.lines.bus1.isin(busids) |\
        network.lines.bus0.isin(busids)

    return network.lines[mask]


def connected_transformer(network, busids):
    """ Get transformer connected to given buses.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    busids  : list
        List containing bus-ids.

    Returns
    -------
    :class:`pandas.DataFrame
        PyPSA transformer.
    """

    mask = (network.transformers.bus0.isin(busids))

    return network.transformers[mask]


def load_shedding(network, **kwargs):
    """ Implement load shedding in existing network to identify
    feasibility problems
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    marginal_cost : int
        Marginal costs for load shedding
    p_nom : int
        Installed capacity of load shedding generator
    Returns
    -------

    """

    marginal_cost_def = 10000  # network.generators.marginal_cost.max()*2
    p_nom_def = network.loads_t.p_set.max().max()

    marginal_cost = kwargs.get('marginal_cost', marginal_cost_def)
    p_nom = kwargs.get('p_nom', p_nom_def)

    network.add("Carrier", "load")
    start = network.generators.index.to_series().str.rsplit(
        ' ').str[0].astype(int).sort_values().max()+1
    index = list(range(start, start+len(network.buses.index)))
    network.import_components_from_dataframe(
        pd.DataFrame(
            dict(marginal_cost=marginal_cost,
                 p_nom=p_nom,
                 carrier='load shedding',
                 bus=network.buses.index),
            index=index),
        "Generator"
    )
    return


def data_manipulation_sh(network):
    from shapely.geometry import Point, LineString, MultiLineString
    from geoalchemy2.shape import from_shape, to_shape

    # add connection from Luebeck to Siems
    new_bus = str(network.buses.index.astype(np.int64).max()+1)
    new_trafo = str(network.transformers.index.astype(np.int64).max()+1)
    new_line = str(network.lines.index.astype(np.int64).max()+1)
    network.add("Bus", new_bus, carrier='AC',
                v_nom=220, x=10.760835, y=53.909745)
    network.add("Transformer", new_trafo, bus0="25536",
                bus1=new_bus, x=1.29960, tap_ratio=1, s_nom=1600)
    network.add("Line", new_line, bus0="26387",
                bus1=new_bus, x=0.0001, s_nom=1600)
    network.lines.loc[new_line, 'cables'] = 3.0

    # bus geom
    point_bus1 = Point(10.760835, 53.909745)
    network.buses.set_value(new_bus, 'geom', from_shape(point_bus1, 4326))

    # line geom/topo
    network.lines.set_value(new_line, 'geom', from_shape(MultiLineString(
        [LineString([to_shape(network.
                              buses.geom['26387']), point_bus1])]), 4326))
    network.lines.set_value(new_line, 'topo', from_shape(LineString(
        [to_shape(network.buses.geom['26387']), point_bus1]), 4326))

    # trafo geom/topo
    network.transformers.set_value(new_trafo,
                                   'geom', from_shape(MultiLineString(
                                       [LineString(
                                           [to_shape(network
                                                     .buses.geom['25536']),
                                            point_bus1])]), 4326))
    network.transformers.set_value(new_trafo, 'topo', from_shape(
        LineString([to_shape(network.buses.geom['25536']), point_bus1]), 4326))

    return

def _enumerate_row(row):
    row['name'] = row.name
    return row

def results_to_csv(network, args, pf_solution = None):
    """
    """

    path = args['results']

    if path == False:
        return None

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    network.export_to_csv_folder(path)
    data = pd.read_csv(os.path.join(path, 'network.csv'))
    data['time'] = network.results['Solver'].Time
    data = data.apply(_enumerate_row,  axis=1)
    data.to_csv(os.path.join(path, 'network.csv'), index=False)

    with open(os.path.join(path, 'args.json'), 'w') as fp:
        json.dump(args, fp)
    
    if type(pf_solution) != type(None):    
        pf_solution.to_csv(os.path.join(path, 'pf_solution.csv'), index=True)
    

    if hasattr(network, 'Z'):
        file = [i for i in os.listdir(
            path.strip('0123456789')) if i == 'Z.csv']
        if file:
            print('Z already calculated')
        else:
            network.Z.to_csv(path.strip('0123456789')+'/Z.csv', index=False)

    return


def parallelisation(network, start_snapshot, end_snapshot, group_size,
                    solver_name, solver_options, extra_functionality=None):

    print("Performing linear OPF, {} snapshot(s) at a time:".
          format(group_size))
    t = time.time()

    for i in range(int((end_snapshot-start_snapshot+1)/group_size)):
        if i > 0:
            network.storage_units.state_of_charge_initial = network.\
                storage_units_t.state_of_charge.loc[
                    network.snapshots[group_size*i-1]]
        network.lopf(network.snapshots[group_size*i:group_size*i+group_size],
                     solver_name=solver_name,
                     solver_options=solver_options,
                     extra_functionality=extra_functionality)
        network.lines.s_nom = network.lines.s_nom_opt

    print(time.time() - t / 60)
    return


def pf_post_lopf(network, **kwargs):

    network_pf = network

    # Update x of extended lines and transformers
    if network_pf.lines.s_nom_extendable.any() or \
        network_pf.transformers.s_nom_extendable.any():
            
        network_pf.lines.x[network.lines.s_nom_extendable] = \
        network_pf.lines.x * network.lines.s_nom /\
        network_pf.lines.s_nom_opt  
        
        network_pf.lines.r[network.lines.s_nom_extendable] = \
        network_pf.lines.r * network.lines.s_nom /\
        network_pf.lines.s_nom_opt  
    
        network_pf.lines.b[network.lines.s_nom_extendable] = \
        network_pf.lines.b * network.lines.s_nom_opt /\
        network_pf.lines.s_nom 
        
        network_pf.lines.g[network.lines.s_nom_extendable] = \
        network_pf.lines.g * network.lines.s_nom_opt /\
        network_pf.lines.s_nom 
        
        network_pf.transformers.x[network.transformers.s_nom_extendable] = \
        network_pf.transformers.x * network.transformers.s_nom / \
        network_pf.transformers.s_nom_opt
        
        network_pf.lines.s_nom_extendable = False
        network_pf.transformers.s_nom_extendable = False
        network_pf.lines.s_nom = network.lines.s_nom_opt
        network_pf.transformers.s_nom =network.transformers.s_nom_opt
        
        network_pf.lopf(solver_name = 'gurobi')
    
        # For the PF, set the P to the optimised P
    network_pf.generators_t.p_set = network_pf.generators_t.p_set.reindex(
        columns=network_pf.generators.index)
    network_pf.generators_t.p_set = network_pf.generators_t.p
    
    network_pf.storage_units_t.p_set = network_pf.storage_units_t.p_set.reindex(
            columns=network_pf.storage_units.index)
    network_pf.storage_units_t.p_set = network_pf.storage_units_t.p
    
    network_pf.links_t.p_set = network_pf.links_t.p_set.reindex(
            columns=network_pf.links.index)
    network_pf.links_t.p_set = network_pf.links_t.p0
    
    
    old_slack = network.generators.index[network.
                                         generators.control == 'Slack'][0]
    old_gens = network.generators
    gens_summed = network.generators_t.p.sum()
    old_gens['p_summed'] = gens_summed
    max_gen_buses_index = old_gens.groupby(['bus']).agg(
        {'p_summed': np.sum}).p_summed.sort_values().index

    for bus_iter in range(1, len(max_gen_buses_index)-1):
        if old_gens[(network.
                     generators['bus'] == max_gen_buses_index[-bus_iter]) &
                    (network.generators['control'] == 'PV')].empty:
            continue
        else:
            new_slack_bus = max_gen_buses_index[-bus_iter]
            break
   
    network.generators = network.generators.drop('p_summed', 1)
    new_slack_gen = network.generators.\
        p_nom[(network.generators['bus'] == new_slack_bus) & (
            network.generators['control'] == 'PV')].sort_values().index[-1]

    # check if old slack was PV or PQ control:
    if network.generators.p_nom[old_slack] > 50 and network.generators.\
            carrier[old_slack] in ('solar', 'wind'):
        old_control = 'PQ'
    elif network.generators.p_nom[old_slack] > 50 and network.generators.\
            carrier[old_slack] not in ('solar', 'wind'):
        old_control = 'PV'
    elif network.generators.p_nom[old_slack] < 50:
        old_control = 'PQ'

    network.generators = network.generators.set_value(
        old_slack, 'control', old_control)
    network.generators = network.generators.set_value(
        new_slack_gen, 'control', 'Slack')
    
    # execute non-linear pf
    pf_solution = network_pf.pf(network.snapshots, use_seed=True)
    
    pf_solve = pd.DataFrame(index =pf_solution['converged'].index)
    pf_solve['converged'] = pf_solution['converged'].values
    pf_solve['error'] = pf_solution['error'].values
    pf_solve['n_iter'] = pf_solution['n_iter'].values
    
    if not pf_solve[pf_solve.converged == False].count().max() == 0:
        logger.warning("PF of  %d snapshots not converged.",\
                pf_solve[pf_solve.converged == False].count().max())
   
    return pf_solve


def distribute_q(network, allocation = 'p_nom'):
   
    if allocation == 'p':
        network.generators_t['p_sum'] = network.generators_t['p'].\
            groupby(network.generators.bus, axis = 1).sum()\
            [network.generators.bus.sort_index()]
        
        network.generators_t['q_sum'] = network.generators_t['q'].\
            groupby(network.generators.bus, axis = 1).sum()\
            [network.generators.bus.sort_index()]

        network.generators_t['p_sum'].columns = \
            network.generators_t['p'].columns
        network.generators_t['q_sum'].columns =\
            network.generators_t['p'].columns
        
        q_distributed = network.generators_t.p / \
            network.generators_t['p_sum'].values*\
            network.generators_t['q_sum'].values
            
            
    if allocation == 'p_nom':
        
        p_nom_dist= network.generators.p_nom.sort_index()
        p_nom_dist[p_nom_dist.index.isin(network.generators.index\
            [network.generators.carrier == 'load shedding'])] = 0  
            
        q_distributed =network.generators_t['q'].\
            groupby(network.generators.bus, axis = 1).sum()\
            [network.generators.bus.sort_index()].multiply(p_nom_dist.values)/ \
            network.generators.p_nom[network.generators.carrier !=\
            'load shedding'].groupby(network.generators.bus).sum()\
            [network.generators.bus.sort_index()].values 
            
        q_distributed.columns =  network.generators.bus.sort_index().index
    

    q_distributed[q_distributed.isnull()] = 0
    q_distributed[q_distributed.abs() == np.inf] = 0
    network.generators_t.q =  q_distributed
    
    return network


def calc_line_losses(network):
    """ Calculate losses per line with PF result data
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    s0 : series
        apparent power of line
    i0 : series
        current of line
    -------

    """

    # Line losses
    # calculate apparent power S = sqrt(p² + q²) [in MW]
    s0_lines = ((network.lines_t.p0**2 + network.lines_t.q0**2).
                apply(np.sqrt))
    # calculate current I = S / U [in A]
    i0_lines = np.multiply(s0_lines, 1000000) / \
        np.multiply(network.lines.v_nom, 1000)
    # calculate losses per line and timestep network.\
    # lines_t.line_losses = I² * R [in MW]
    network.lines_t.losses = np.divide(i0_lines**2 * network.lines.r, 1000000)
    # calculate total losses per line [in MW]
    network.lines = network.lines.assign(
        losses=np.sum(network.lines_t.losses).values)

    # Transformer losses
    # https://books.google.de/books?id=0glcCgAAQBAJ&pg=PA151&lpg=PA151&dq=
    # wirkungsgrad+transformator+1000+mva&source=bl&ots=a6TKhNfwrJ&sig=
    # r2HCpHczRRqdgzX_JDdlJo4hj-k&hl=de&sa=X&ved=
    # 0ahUKEwib5JTFs6fWAhVJY1AKHa1cAeAQ6AEIXjAI#v=onepage&q=
    # wirkungsgrad%20transformator%201000%20mva&f=false
    # Crastan, Elektrische Energieversorgung, p.151
    # trafo 1000 MVA: 99.8 %
    network.transformers = network.transformers.assign(
        losses=np.multiply(network.transformers.s_nom, (1-0.998)).values)

    # calculate total losses (possibly enhance with adding these values
    # to network container)
    losses_total = sum(network.lines.losses) + sum(network.transformers.losses)
    print("Total lines losses for all snapshots [MW]:", round(losses_total, 2))
    losses_costs = losses_total * np.average(network.buses_t.marginal_price)
    print("Total costs for these losses [EUR]:", round(losses_costs, 2))

    return


def loading_minimization(network, snapshots):

    network.model.number1 = Var(
        network.model.passive_branch_p_index, within=PositiveReals)
    network.model.number2 = Var(
        network.model.passive_branch_p_index, within=PositiveReals)

    def cRule(model, c, l, t):
        return (model.number1[c, l, t] - model.number2[c, l, t] == model.
                passive_branch_p[c, l, t])

    network.model.cRule = Constraint(
        network.model.passive_branch_p_index, rule=cRule)

    network.model.objective.expr += 0.00001 * \
        sum(network.model.number1[i] + network.model.number2[i]
            for i in network.model.passive_branch_p_index)


def group_parallel_lines(network):

    # ordering of buses: (not sure if still necessary, remaining from SQL code)
    old_lines = network.lines

    for line in old_lines.index:
        bus0_new = str(old_lines.loc[line, ['bus0', 'bus1']].astype(int).min())
        bus1_new = str(old_lines.loc[line, ['bus0', 'bus1']].astype(int).max())
        old_lines.set_value(line, 'bus0', bus0_new)
        old_lines.set_value(line, 'bus1', bus1_new)

    # saving the old index
    for line in old_lines:
        old_lines['old_index'] = network.lines.index

    grouped = old_lines.groupby(['bus0', 'bus1'])

    # calculating electrical properties for parallel lines
    grouped_agg = grouped.\
        agg({'b': np.sum,
             'b_pu': np.sum,
             'cables': np.sum,
             'capital_cost': np.min,
             'frequency': np.mean,
             'g': np.sum,
             'g_pu': np.sum,
             'geom': lambda x: x[0],
             'length': lambda x: x.min(),
             'num_parallel': np.sum,
             'r': lambda x: np.reciprocal(np.sum(np.reciprocal(x))),
             'r_pu': lambda x: np.reciprocal(np.sum(np.reciprocal(x))),
             's_nom': np.sum,
             's_nom_extendable': lambda x: x.min(),
             's_nom_max': np.sum,
             's_nom_min': np.sum,
             's_nom_opt': np.sum,
             'scn_name': lambda x: x.min(),
             'sub_network': lambda x: x.min(),
             'terrain_factor': lambda x: x.min(),
             'topo': lambda x: x[0],
             'type': lambda x: x.min(),
             'v_ang_max': lambda x: x.min(),
             'v_ang_min': lambda x: x.min(),
             'x': lambda x: np.reciprocal(np.sum(np.reciprocal(x))),
             'x_pu': lambda x: np.reciprocal(np.sum(np.reciprocal(x))),
             'old_index': np.min})

    for i in range(0, len(grouped_agg.index)):
        grouped_agg.set_value(
            grouped_agg.index[i], 'bus0', grouped_agg.index[i][0])
        grouped_agg.set_value(
            grouped_agg.index[i], 'bus1', grouped_agg.index[i][1])

    new_lines = grouped_agg.set_index(grouped_agg.old_index)
    new_lines = new_lines.drop('old_index', 1)
    network.lines = new_lines

    return


def set_line_costs(network, cost110=230, cost220=290, cost380=85):
    """ Set capital costs for extendable lines in respect to PyPSA [€/MVA]
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    cost110 : capital costs per km for 110kV lines and cables
                default: 230€/MVA/km, source: costs for extra circuit in
                dena Verteilnetzstudie, p. 146)
    cost220 : capital costs per km for 220kV lines and cables
                default: 280€/MVA/km, source: costs for extra circuit in
                NEP 2025, capactity from most used 220 kV lines in model
    cost380 : capital costs per km for 380kV lines and cables
                default: 85€/MVA/km, source: costs for extra circuit in
                NEP 2025, capactity from most used 380 kV lines in NEP
    -------

    """
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    network.lines.loc[(network.lines.v_nom == 110) & network.lines.
                      s_nom_extendable,
                      'capital_cost'] = cost110 * network.lines.length
    network.lines.loc[(network.lines.v_nom == 220) & network.lines.
                      s_nom_extendable,
                      'capital_cost'] = cost220 * network.lines.length
    network.lines.loc[(network.lines.v_nom == 380) & network.lines.
                      s_nom_extendable,
                      'capital_cost'] = cost380 * network.lines.length

    return network


def set_trafo_costs(network, cost110_220=7500, cost110_380=17333,
                    cost220_380=14166):
    """ Set capital costs for extendable transformers in respect
    to PyPSA [€/MVA]
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    cost110_220 : capital costs for 110/220kV transformer
                    default: 7500€/MVA, source: costs for extra trafo in
                    dena Verteilnetzstudie, p. 146; S of trafo used in osmTGmod
    cost110_380 : capital costs for 110/380kV transformer
                default: 17333€/MVA, source: NEP 2025
    cost220_380 : capital costs for 220/380kV transformer
                default: 14166€/MVA, source: NEP 2025

    """
    network.transformers["v_nom0"] = network.transformers.bus0.map(
        network.buses.v_nom)
    network.transformers["v_nom1"] = network.transformers.bus1.map(
        network.buses.v_nom)

    network.transformers.loc[(network.transformers.v_nom0 == 110) & (
        network.transformers.v_nom1 == 220) & network.transformers.
        s_nom_extendable, 'capital_cost'] = cost110_220
    network.transformers.loc[(network.transformers.v_nom0 == 110) & (
        network.transformers.v_nom1 == 380) & network.transformers.
        s_nom_extendable, 'capital_cost'] = cost110_380
    network.transformers.loc[(network.transformers.v_nom0 == 220) & (
        network.transformers.v_nom1 == 380) & network.transformers.
        s_nom_extendable, 'capital_cost'] = cost220_380

    return network

def add_missing_components(network):
    from shapely import wkb
    from shapely.geometry import Point, LineString, MultiLineString
    from geoalchemy2.shape import from_shape, to_shape
    # Munich
    '''
     add missing transformer at Heizkraftwerk Nord in Munich:
     https://www.swm.de/privatkunden/unternehmen/energieerzeugung/heizkraftwerke.html?utm_medium=301
@@ -329,27 +334,28 @@
     to bus 25096:
     25369 (86)
     28232 (24)
     25353 to 25356 (79)
     to bus 23822: (110kV bus  of 380/110-kV-transformer)
     25355 (90)
     28212 (98)
 
     25357 to 665 (85)
     25354 to 27414 (30)
     27414 to 28212 (33)
     25354 to 28294 (32/63)
     28335 to 28294 (64)
     28335 to 28139 (28)
     Overhead lines:
     16573 to 24182 (part of 4)
     '''
    """
     Installierte Leistung der Umspannungsebene Höchst- zu Hochspannung
     (380 kV / 110 kV): 2.750.000 kVA
     https://www.swm-infrastruktur.de/strom/netzstrukturdaten/strukturmerkmale.html
    """
    new_trafo = str(network.transformers.index.astype(int).max()+1)

    network.add("Transformer", new_trafo, bus0="23648", bus1="16573",
                x=0.135/(2750/2),
                 r=0.0, tap_ratio=1, s_nom=2750/2)
 
     # trafo geom/topo
    """(network.transformers.loc[new_trafo, 'geom']
     ) = (from_shape(MultiLineString
                      ([LineString([wkb.loads(network.buses.geom['23648'],
                                              hex=True),
                                    wkb.loads(network.buses.geom['16573'],
                                              hex=True)])]), 4326))
    (network.transformers.loc[new_trafo, 'topo']
     ) = (from_shape(LineString([wkb.loads(network.buses.geom['23648'],
                                            hex=True),
                                  wkb.loads(network.buses.geom['16573'],
                                            hex=True)]), 4326))"""
 
    def add_110kv_line(bus0, bus1, overhead=False):
         new_line = str(network.lines.index.astype(int).max()+1)
         if not overhead:
             network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=280)
         else:
             network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=260)
         network.lines.loc[new_line, "scn_name"] = "Status Quo"
         network.lines.loc[new_line, "v_nom"] = 110
         network.lines.loc[new_line, "version"] = "added_manually"
         network.lines.loc[new_line, "frequency"] = 50
         network.lines.loc[new_line, "cables"] = 3.0
         network.lines.loc[new_line, "length"] = (
             pypsa.geo.haversine(network.buses.loc[bus0, ["x", "y"]],
                                 network.buses.loc[bus1, ["x", "y"]])[0][0]*1.2)
         if not overhead:
             network.lines.loc[new_line, "r"] = (network.lines.
                                                 loc[new_line, "length"]*0.0177)
             network.lines.loc[new_line, "g"] = 0
             # or: (network.lines.loc[new_line, "length"]*78e-9)
             network.lines.loc[new_line, "x"] = (network.lines.
                                                 loc[new_line, "length"]*0.3e-3)
             network.lines.loc[new_line, "b"] = (network.lines.
                                                 loc[new_line, "length"]*250e-9)

         elif overhead:
             network.lines.loc[new_line, "r"] = (network.lines.
                                                 loc[new_line, "length"] *
                                                 0.05475)
             network.lines.loc[new_line, "g"] = 0
             # or: (network.lines.loc[new_line, "length"]*40e-9)
             network.lines.loc[new_line, "x"] = (network.lines.
                                                 loc[new_line, "length"]*1.2e-3)
             network.lines.loc[new_line, "b"] = (network.lines.
                                                 loc[new_line, "length"]*9.5e-9)
 
         # line geom/topo
         """(network.lines.loc[new_line, 'geom']
          ) = from_shape(MultiLineString
                         ([LineString([wkb.loads(network.buses.geom[bus0],
                                                 hex=True),
                                       wkb.loads(network.buses.geom[bus1],
                                                 hex=True)])]), 4326)
         (network.lines.loc[new_line, 'topo']
          ) = from_shape(LineString
                         ([wkb.loads(network.buses.geom[bus0], hex=True),
                           wkb.loads(network.buses.geom[bus1], hex=True)]),
                         4326)"""
 
    add_110kv_line("16573", "28353")
    add_110kv_line("16573", "28092")
    add_110kv_line("25096", "25369")
    add_110kv_line("25096", "28232")
    add_110kv_line("25353", "25356")
    add_110kv_line("23822", "25355")
    add_110kv_line("23822", "28212")
    add_110kv_line("25357", "665")
    add_110kv_line("25354", "27414")
    add_110kv_line("27414", "28212")
    add_110kv_line("25354", "28294")
    add_110kv_line("28335", "28294")
    add_110kv_line("28335", "28139")
    add_110kv_line("16573", "24182", overhead=True)


        # Stuttgart
    """
         Stuttgart:
         Missing transformer, because 110-kV-bus is situated outside
         Heizkraftwerk Heilbronn:
    """
        # new_trafo = str(network.transformers.index.astype(int).max()1)
    network.add("Transformer", '99999', bus0="25766", bus1="18967",
                x=0.135/300, r=0.0, tap_ratio=1, s_nom=300)

    # trafo geom/topo
    """(network.transformers.loc[new_trafo, 'geom']
         ) = (from_shape(MultiLineString
                     ([LineString([wkb.loads(network.buses.geom['25766'],
                                             hex=True),
                                   wkb.loads(network.buses.geom['18967'],
                                             hex=True)])]), 4326))
         (network.transformers.loc[new_trafo, 'topo']
         ) = (from_shape(LineString([wkb.loads(network.buses.geom['25766'],
                                           hex=True),
                                 wkb.loads(network.buses.geom['18967'],
                                           hex=True)]), 4326))"""
    """
    According to:
    https://assets.ctfassets.net/xytfb1vrn7of/NZO8x4rKesAcYGGcG4SQg/b780d6a3ca4c2600ab51a30b70950bb1/netzschemaplan-110-kv.pdf
    the following lines are missing:
    """
    add_110kv_line("18967", "22449", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("21165", "24068", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("23782", "24089", overhead=True)  # visible in DSO map & OSM till 1 km from bus1
    """
    Umspannwerk Möhringen (bus 23697)
    https://de.wikipedia.org/wiki/Umspannwerk_M%C3%B6hringen
    there should be two connections:
    to Sindelfingen (2*110kV)
    to Wendingen (former 220kV, now 2*110kV)
    the line to Sindelfingen is connected, but the connection of Sindelfingen
    itself to 380kV is missing:
    """
    add_110kv_line("19962", "27671", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("19962", "27671", overhead=True)
    """
    line to Wendingen is missing, probably because it ends shortly before the
    way of the substation and is connected via cables:
    """
    add_110kv_line("23697", "24090", overhead=True)  # visible in OSM & DSO map
    add_110kv_line("23697", "24090", overhead=True)

    # Lehrte
    """
    Lehrte: 220kV Bus located outsinde way of Betriebszentrtum Lehrte and
    therefore not connected:
    """

    def add_220kv_line(bus0, bus1, overhead=False):
            new_line = str(network.lines.index.astype(int).max()+1)
            if not overhead:
                 network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=550)
            else:
                network.add("Line", new_line, bus0=bus0, bus1=bus1, s_nom=520)
            network.lines.loc[new_line, "scn_name"] = "Status Quo"
            network.lines.loc[new_line, "v_nom"] = 220
            network.lines.loc[new_line, "version"] = "added_manually"
            network.lines.loc[new_line, "frequency"] = 50
            network.lines.loc[new_line, "cables"] = 3.0
            network.lines.loc[new_line, "length"] = (
            pypsa.geo.haversine(network.buses.loc[bus0, ["x", "y"]],
                                network.buses.loc[bus1, ["x", "y"]])[0][0]*1.2)
            if not overhead:
                network.lines.loc[new_line, "r"] = (network.lines.
                                                loc[new_line, "length"]*0.0176)
                network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*67e-9)
                network.lines.loc[new_line, "x"] = (network.lines.
                                                loc[new_line, "length"]*0.3e-3)
                network.lines.loc[new_line, "b"] = (network.lines.
                                                loc[new_line, "length"]*210-9)
            elif overhead:
                    network.lines.loc[new_line, "r"] = (network.lines.
                                                loc[new_line, "length"] *
                                                0.05475)
                    network.lines.loc[new_line, "g"] = 0
            # or: (network.lines.loc[new_line, "length"]*30e-9)
            network.lines.loc[new_line, "x"] = (network.lines.
                                                loc[new_line, "length"]*1e-3)
            network.lines.loc[new_line, "b"] = (network.lines.
                                                loc[new_line, "length"]*11e-9)

        # line geom/topo
            """(network.lines.loc[new_line, 'geom']
            ) = from_shape(MultiLineString
                        ([LineString([wkb.loads(network.buses.geom[bus0],
                                                hex=True),
                                      wkb.loads(network.buses.geom[bus1],
                                                hex=True)])]), 4326)
            (network.lines.loc[new_line, 'topo']
            ) = from_shape(LineString
                        ([wkb.loads(network.buses.geom[bus0], hex=True),
                          wkb.loads(network.buses.geom[bus1], hex=True)]),
                        4326)"""

    add_220kv_line("266", "24633", overhead=True)
    return network

def convert_capital_costs(network, start_snapshot, end_snapshot, p=0.05, T=40):
    """ Convert capital_costs to fit to pypsa and caluculated time
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    p : interest rate, default 0.05
    T : number of periods, default 40 years (source: StromNEV Anlage 1)
    -------

    """
    # Add costs for converter
    network.links.capital_cost = network.links.capital_cost + 400000

    # Calculate present value of an annuity (PVA)
    PVA = (1 / p) - (1 / (p*(1 + p) ** T))

    #
    network.lines.loc[network.lines.s_nom_extendable == True,
                      'capital_cost'] = (network.lines.capital_cost /
                                         (PVA * (8760/(end_snapshot -
                                                       start_snapshot + 1))))
    network.links.loc[network.links.p_nom_extendable == True,
                      'capital_cost'] = network.\
        links.capital_cost / (PVA * (8760//(end_snapshot -
                                            start_snapshot + 1)))
    network.transformers.loc[network.transformers.s_nom_extendable == True,
                             'capital_cost'] = network.\
        transformers.capital_cost / \
        (PVA * (8760//(end_snapshot - start_snapshot + 1)))
    network.storage_units.loc[network.storage_units.
                              p_nom_extendable == True,
                              'capital_cost'] = network.\
        storage_units.capital_cost / (8760//(end_snapshot -
                                             start_snapshot + 1))

    return network


import pyomo.environ as po


def min_renewable_share(network, share=0.71):
    """
    """
    renewables = ['wind_onshore', 'wind_offshore', 'biomass', 'solar', 'run_of_river']
    #import pdb; pdb.set_trace()
    res = list(network.generators.index[network.generators.carrier.isin(renewables)])

    total = list(network.generators.index)
    snapshots = network.snapshots
    share=0.73

    def _rule(m):
        """
        """
        renewable_production = sum(m.generator_p[gen,sn]
                                  for gen
                                  in res
                                  for sn in snapshots)
        total_production = sum(m.generator_p[gen,sn]
                               for gen  in total
                               for sn in snapshots)

        return (renewable_production >= total_production* share)
    network.model.min_renewable_share = po.Constraint(rule=_rule)
    network.model.min_renewable_share.pprint()

    #return m