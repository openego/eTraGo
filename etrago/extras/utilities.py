import pandas as pd
import os
import time

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

#    future way to add the geoms of the new components, currently bugged in pandas/shapely

#    #bus geom
#    point_bus1 = Point(10.760835,53.909745)
#    network.buses.loc[new_bus,'geom'] = from_shape(point_bus1, 4326)
#    
#    #line geom/topo
#    network.lines.loc[new_line,'geom'] = from_shape(MultiLineString([LineString([to_shape(network.buses.geom['26387']),point_bus1])]),4326)
#    network.lines.loc[new_line,'topo'] = from_shape(LineString([to_shape(network.buses.geom['26387']),point_bus1]),4326)
#    
#    #trafo geom/topo
#    network.transformers.loc[new_trafo,'geom'] = from_shape(MultiLineString([LineString([to_shape(network.buses.geom['25536']),point_bus1])]),4326)
#    network.transformers.loc[new_trafo,'topo'] = from_shape(LineString([to_shape(network.buses.geom['25536']),point_bus1]),4326)
    
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

def parallelisation(network, start_h, end_h, group_size, solver_name):

    print("Performing linear OPF, {} snapshot(s) at a time:".format(group_size))
    x = time.time()
    for i in range(int((end_h-start_h+1)/group_size)):
        network.lopf(network.snapshots[group_size*i:group_size*i+group_size], solver_name=solver_name)


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
    
    #calculate p line losses
    #todo

    return network_pf

