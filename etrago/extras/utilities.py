import pandas as pd
import numpy as np
import os
import time
from pyomo.environ import (Var,Constraint, PositiveReals,ConcreteModel)
from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap

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
    start = network.buses.index.astype(int).max()
    nums = len(network.buses.index)
    end = start+nums
    index = list(range(start,end))
    index = [str(x) for x in index]
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
    network.transformers.set_value(new_trafo, 'geom', from_shape(LineString([to_shape(network.buses.geom['25536']),point_bus1]),4326))

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

def parallelisation(network, start_h, end_h, group_size, solver_name, extra_functionality=None):

    print("Performing linear OPF, {} snapshot(s) at a time:".format(group_size))
    x = time.time()
    for i in range(int((end_h-start_h+1)/group_size)):
        network.lopf(network.snapshots[group_size*i:group_size*i+group_size], solver_name=solver_name, extra_functionality=extra_functionality)


    y = time.time()
    z = (y - x) / 60
    return

def pf_post_lopf(network, scenario):
    
    network_pf = network    

    #For the PF, set the P to the optimised P
    network_pf.generators_t.p_set = network_pf.generators_t.p_set.reindex(columns=network_pf.generators.index)
    network_pf.generators_t.p_set = network_pf.generators_t.p
    
    #Calculate q set from p_set with given cosphi
    #todo

    #Troubleshooting        
    #network_pf.generators_t.q_set = network_pf.generators_t.q_set*0
    #network.loads_t.q_set = network.loads_t.q_set*0
    #network.loads_t.p_set['28314'] = network.loads_t.p_set['28314']*0.5
    #network.loads_t.q_set['28314'] = network.loads_t.q_set['28314']*0.5
    #network.transformers.x=network.transformers.x['22596']*0.01
    #contingency_factor=2
    #network.lines.s_nom = contingency_factor*pups.lines.s_nom
    #network.transformers.s_nom = network.transformers.s_nom*contingency_factor
    
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
    # calculate apparent power S = sqrt(p² + q²)
    s0_lines = ((network.lines_t.p0**2 + network.lines_t.q0**2).\
        apply(np.sqrt))
    # calculate current I = S / U
    i0_lines = s0_lines / network.lines.v_nom
    # calculate losses per line and timestep network.lines_t.line_losses = I² * R
    network.lines_t.losses = i0_lines**2 * network.lines.r
    # calculate total losses per line
    network.lines.losses = np.sum(network.lines_t.losses)
        
    #### Transformer losses
    # calculate apparent power S = sqrt(p² + q²)
    s0_trafo = ((network.transformers_t.p0**2 + network.transformers_t.q0**2).\
        apply(np.sqrt))
    # calculate losses per transformer and timestep
    #    network.transformers_t.losses = s0_trafo / network.transformers.s_nom ## !!! this needs to be finalised
    # calculate fix no-load losses per transformer
    network.transformers.losses_fix = 0.00275 * network.transformers.s_nom # average value according to http://ibn.ch/HomePageSchule/Schule/GIBZ/19_Transformatoren/19_Transformatoren_Loesung.pdf
    # calculate total losses per line
    network.transformers.losses = network.transformers.losses_fix # + np.sum(network.transformers_t.losses)
        
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

def kmean_clustering(network):
    """ Implement k-mean clustering in existing network
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    Returns
    -------

    """
    def weighting_for_scenario(x):
        b_i = x.index
        g = normed(gen.reindex(b_i, fill_value=0))
        l = normed(load.reindex(b_i, fill_value=0))
      
        w= g + l
        return (w * (100. / w.max())).astype(int)

    def normed(x):
        return (x/x.sum()).fillna(0.)
    
    print('start k-mean clustering')
    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control="PV"
    network.buses['v_nom'] = 380.
    # problem our lines have no v_nom. this is implicitly defined by the connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the x of the lines which are not 380. 
    lines_v_nom_b = network.lines.v_nom != 380
    network.lines.loc[lines_v_nom_b, 'x'] *= (380./network.lines.loc[lines_v_nom_b, 'v_nom'])**2
    network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.

    trafo_index = network.transformers.index
    transformer_voltages = pd.concat([network.transformers.bus0.map(network.buses.v_nom), network.transformers.bus1.map(network.buses.v_nom)], axis=1)


    network.import_components_from_dataframe(
    network.transformers.loc[:,['bus0','bus1','x','s_nom']]
    .assign(x=network.transformers.x*(380./transformer_voltages.max(axis=1))**2)
    .set_index('T' + trafo_index),
    'Line')
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
      network.transformers_t[attr] = network.transformers_t[attr].reindex(columns=[])

    #ToDo: change conv to types minus wind and solar 
    conv_types = {'biomass', 'run_of_river', 'gas', 'oil','coal', 'waste','uranium'}
    # Attention: network.generators.carrier.unique() 
    # conv_types only for SH scenario defined!
    gen = (network.generators.loc[network.generators.carrier.isin(conv_types)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, 
        fill_value=0.) + network.storage_units.loc[network.storage_units.carrier.isin(conv_types)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, fill_value=0.))
        
    load = network.loads_t.p_set.mean().groupby(network.loads.bus).sum()

    # k-mean clustering
    # busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1,
    #       len(network.buses)), index=network.buses.index) , n_clusters= 10)
    weight = weighting_for_scenario(network.buses).reindex(network.buses.index, fill_value=1)
    busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(weight), buses_i=network.buses.index , n_clusters= 10)


    # ToDo change function in order to use bus_strategies or similar
    clustering = get_clustering_from_busmap(network, busmap)
    network = clustering.network
    #network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

    return network

    
    
   

