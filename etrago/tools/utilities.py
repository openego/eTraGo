"""
Utilities.py defines functions necessary to apply eTraGo.

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


import pandas as pd
import numpy as np
import os
import time
from pyomo.environ import (Var,Constraint, PositiveReals,ConcreteModel)


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
    Add transborder flows divided by country of origin as network.foreign_trade.
    
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
    poland = pd.Series(index=network.buses[(network.buses['x'] > 17)].index,
                                                  data="Poland")
    czech = pd.Series(index=network.buses[(network.buses['x'] < 17) &
                                            (network.buses['x'] > 15.1)].index,
                                            data="Czech")
    denmark = pd.Series(index=network.buses[((network.buses['y'] < 60) &
                                            (network.buses['y'] > 55.2)) |
                                            ((network.buses['x'] > 11.95) &
                                               (network.buses['x'] < 11.97) &
                                               (network.buses['y'] > 54.5))].index,
                                            data="Denmark")
    sweden = pd.Series(index=network.buses[(network.buses['y'] > 60)].index,
                                            data="Sweden")
    austria = pd.Series(index=network.buses[(network.buses['y'] < 47.33) &
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
    switzerland = pd.Series(index=network.buses[((network.buses['x'] > 8.1) &
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
    netherlands = pd.Series(index=network.buses[((network.buses['x'] < 6.96) &
                                               (network.buses['y'] < 53.15) &
                                               (network.buses['y'] > 53.1)) |
                                                ((network.buses['x'] < 5.4) &
                                               (network.buses['y'] > 52.1))].index,
                                                data = "Netherlands")
    luxembourg = pd.Series(index=network.buses[((network.buses['x'] < 6.15) &
                                               (network.buses['y'] < 49.91) &
                                               (network.buses['y'] > 49.65))].index,
                                                data="Luxembourg")
    france = pd.Series(index=network.buses[(network.buses['x'] < 4.5) |
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
    foreign_buses = foreign_buses.append([poland, czech, denmark, sweden, austria, switzerland,
                          netherlands, luxembourg, france])
    
    network.buses = network.buses.drop(network.buses.loc[foreign_buses.index].index)                                        
    
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
    for i in range (0, len(transborder_lines)):
        if transborder_lines.iloc[i, 0] in foreign_buses.index:
            transborder_lines['country'][i] = foreign_buses[str(transborder_lines.iloc[i, 0])]
        else:
            transborder_lines['country'][i] = foreign_buses[str(transborder_lines.iloc[i, 1])]

    # identify amount of flows per line and group to get flow per country
    transborder_flows = network.lines_t.p0[transborder_lines.index]
    for i in transborder_flows.columns:
        if network.lines.loc[str(i)]['bus1'] in foreign_buses.index:
            transborder_flows.loc[:, str(i)] = transborder_flows.loc[:, str(i)]*-1

    network.foreign_trade = transborder_flows.\
                       groupby(transborder_lines['country'], axis=1).sum()
    
    # drop foreign components     
    network.lines = network.lines.drop(network.lines[
            (network.lines['bus0'].isin(network.buses.index) == False) |
            (network.lines['bus1'].isin(network.buses.index) == False)].index)
    network.transformers = network.transformers.drop(network.transformers[
            (network.transformers['bus0'].isin(network.buses.index) == False) |
            (network.transformers['bus1'].isin(network.buses.index) == False)].index)
    network.generators = network.generators.drop(network.generators[
            (network.generators['bus'].isin(network.buses.index) == False)].index)
    network.loads = network.loads.drop(network.loads[
            (network.loads['bus'].isin(network.buses.index) == False)].index)
    network.storage_units = network.storage_units.drop(network.storage_units[
            (network.storage_units['bus'].isin(network.buses.index) == False)].index)
    
    components = ['loads', 'generators', 'lines', 'buses', 'transformers']
    for g in components: #loads_t
        h = g + '_t'
        nw = getattr(network, h) # network.loads_t
        for i in nw.keys(): #network.loads_t.p
            cols = [j for j in getattr(nw, i).columns if j not in getattr(network, g).index]
            for k in cols:
                del getattr(nw, i)[k]
    
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


def load_shedding (network, **kwargs):
    """ Implement load shedding in existing network to identify feasibility problems
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

    marginal_cost_def = 10000#network.generators.marginal_cost.max()*2
    p_nom_def = network.loads_t.p_set.max().max()

    marginal_cost = kwargs.get('marginal_cost', marginal_cost_def)
    p_nom = kwargs.get('p_nom', p_nom_def)
    
    network.add("Carrier", "load")
    start = network.generators.index.astype(int).max()+1
    index = list(range(start,start+len(network.buses.index)))
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


def data_manipulation_sh (network):
    from shapely.geometry import Point, LineString, MultiLineString
    from geoalchemy2.shape import from_shape, to_shape
    
    #add connection from Luebeck to Siems

    new_bus = str(int(network.buses.index.max())+1)
    new_trafo = str(int(network.transformers.index.max())+1)
    new_line = str(int(network.lines.index.max())+1)
    network.add("Bus", new_bus,carrier='AC', v_nom=220, x=10.760835, y=53.909745)
    network.add("Transformer", new_trafo, bus0="25536", bus1=new_bus, x=1.29960, tap_ratio=1, s_nom=1600)
    network.add("Line",new_line, bus0="26387",bus1=new_bus, x=0.0001, s_nom=1600)
    network.lines.loc[new_line,'cables']=3.0

    #bus geom
    point_bus1 = Point(10.760835,53.909745)
    network.buses.set_value(new_bus, 'geom', from_shape(point_bus1, 4326))

    #line geom/topo
    network.lines.set_value(new_line, 'geom', from_shape(MultiLineString([LineString([to_shape(network.buses.geom['26387']),point_bus1])]),4326))
    network.lines.set_value(new_line, 'topo', from_shape(LineString([to_shape(network.buses.geom['26387']),point_bus1]),4326))

    #trafo geom/topo
    network.transformers.set_value(new_trafo, 'geom', from_shape(MultiLineString([LineString([to_shape(network.buses.geom['25536']),point_bus1])]),4326))
    network.transformers.set_value(new_trafo, 'topo', from_shape(LineString([to_shape(network.buses.geom['25536']),point_bus1]),4326))

    return
    
def results_to_csv(network, path):
    """
    """
    if path==False:
        return None

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    network.export_to_csv_folder(path)
    data = pd.read_csv(os.path.join(path, 'network.csv'))
    data['time'] = network.results['Solver'].Time
    data.to_csv(os.path.join(path, 'network.csv'))

    if hasattr(network, 'Z'):
        file = [i for i in os.listdir(path.strip('0123456789')) if i=='Z.csv']
        if file:
           print('Z already calculated')
        else:
           network.Z.to_csv(path.strip('0123456789')+'/Z.csv', index=False)

    return

def parallelisation(network, start_snapshot, end_snapshot, group_size, solver_name, extra_functionality=None):

    print("Performing linear OPF, {} snapshot(s) at a time:".format(group_size))
    x = time.time()

    for i in range(int((end_snapshot-start_snapshot+1)/group_size)):
        if i>0:
            network.storage_units.state_of_charge_initial = network.storage_units_t.state_of_charge.loc[network.snapshots[group_size*i-1]]
        network.lopf(network.snapshots[group_size*i:group_size*i+group_size], solver_name=solver_name, extra_functionality=extra_functionality)
        network.lines.s_nom = network.lines.s_nom_opt

    y = time.time()
    z = (y - x) / 60
    return

def pf_post_lopf(network, scenario):
    
    network_pf = network    

    #For the PF, set the P to the optimised P
    network_pf.generators_t.p_set = network_pf.generators_t.p_set.reindex(columns=network_pf.generators.index)
    network_pf.generators_t.p_set = network_pf.generators_t.p
    
    old_slack = network.generators.index[network.generators.control == 'Slack'][0]
    old_gens = network.generators
    gens_summed = network.generators_t.p.sum()
    old_gens['p_summed']= gens_summed  
    max_gen_buses_index = old_gens.groupby(['bus']).agg({'p_summed': np.sum}).p_summed.sort_values().index
    
    for bus_iter in range(1,len(max_gen_buses_index)-1):
        if old_gens[(network.generators['bus']==max_gen_buses_index[-bus_iter])&(network.generators['control']=='PV')].empty:
            continue
        else:
            new_slack_bus = max_gen_buses_index[-bus_iter]
            break
        
    network.generators=network.generators.drop('p_summed',1)
    new_slack_gen = network.generators.p_nom[(network.generators['bus'] == new_slack_bus)&(network.generators['control'] == 'PV')].sort_values().index[-1]    
    
    # check if old slack was PV or PQ control:
    if network.generators.p_nom[old_slack] > 50 and network.generators.carrier[old_slack] in ('solar','wind'):
        old_control = 'PQ'
    elif network.generators.p_nom[old_slack] > 50 and network.generators.carrier[old_slack] not in ('solar','wind'):
        old_control = 'PV'
    elif network.generators.p_nom[old_slack] < 50:
        old_control = 'PQ'
     
    network.generators = network.generators.set_value(old_slack, 'control', old_control)
    network.generators = network.generators.set_value(new_slack_gen, 'control', 'Slack')
   
    #execute non-linear pf
    network_pf.pf(scenario.timeindex, use_seed=True)
    
    return network_pf

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
    #### Line losses
    # calculate apparent power S = sqrt(p² + q²) [in MW]
    s0_lines = ((network.lines_t.p0**2 + network.lines_t.q0**2).\
        apply(np.sqrt)) 
    # calculate current I = S / U [in A]
    i0_lines = np.multiply(s0_lines, 1000000) / np.multiply(network.lines.v_nom, 1000) 
    # calculate losses per line and timestep network.lines_t.line_losses = I² * R [in MW]
    network.lines_t.losses = np.divide(i0_lines**2 * network.lines.r, 1000000)
    # calculate total losses per line [in MW]
    network.lines = network.lines.assign(losses=np.sum(network.lines_t.losses).values)
    
    #### Transformer losses   
    # https://books.google.de/books?id=0glcCgAAQBAJ&pg=PA151&lpg=PA151&dq=wirkungsgrad+transformator+1000+mva&source=bl&ots=a6TKhNfwrJ&sig=r2HCpHczRRqdgzX_JDdlJo4hj-k&hl=de&sa=X&ved=0ahUKEwib5JTFs6fWAhVJY1AKHa1cAeAQ6AEIXjAI#v=onepage&q=wirkungsgrad%20transformator%201000%20mva&f=false
    # Crastan, Elektrische Energieversorgung, p.151
    # trafo 1000 MVA: 99.8 %
    network.transformers = network.transformers.assign(losses=np.multiply(network.transformers.s_nom,(1-0.998)).values)
        
    # calculate total losses (possibly enhance with adding these values to network container)
    losses_total = sum(network.lines.losses) + sum(network.transformers.losses)
    print("Total lines losses for all snapshots [MW]:",round(losses_total,2))
    losses_costs = losses_total * np.average(network.buses_t.marginal_price)
    print("Total costs for these losses [EUR]:",round(losses_costs,2))
  
    return
    
def loading_minimization(network,snapshots):

    network.model.number1 = Var(network.model.passive_branch_p_index, within = PositiveReals)
    network.model.number2 = Var(network.model.passive_branch_p_index, within = PositiveReals)

    def cRule(model, c, l, t):
        return (model.number1[c, l, t] - model.number2[c, l, t] == model.passive_branch_p[c, l, t])

    network.model.cRule=Constraint(network.model.passive_branch_p_index, rule=cRule)

    network.model.objective.expr += 0.00001* sum(network.model.number1[i] + network.model.number2[i] for i in network.model.passive_branch_p_index)

    
def group_parallel_lines(network):
    
    #ordering of buses: (not sure if still necessary, remaining from SQL code)
    old_lines = network.lines
    
    for line in old_lines.index:
        bus0_new = str(old_lines.loc[line,['bus0','bus1']].astype(int).min())
        bus1_new = str(old_lines.loc[line,['bus0','bus1']].astype(int).max())
        old_lines.set_value(line,'bus0',bus0_new)
        old_lines.set_value(line,'bus1',bus1_new)
        
    # saving the old index
    for line in old_lines:
        old_lines['old_index'] = network.lines.index
    
    grouped = old_lines.groupby(['bus0','bus1'])
    
    #calculating electrical properties for parallel lines
    grouped_agg = grouped.agg({ 'b': np.sum,
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
    
    for i in range(0,len(grouped_agg.index)):
        grouped_agg.set_value(grouped_agg.index[i],'bus0',grouped_agg.index[i][0])
        grouped_agg.set_value(grouped_agg.index[i],'bus1',grouped_agg.index[i][1])
        
    new_lines=grouped_agg.set_index(grouped_agg.old_index)
    new_lines=new_lines.drop('old_index',1)
    network.lines = new_lines
    
    return
