# -*- coding: utf-8 -*-

import pandas as pd
from importlib import import_module
from etrago.tools.io import NetworkScenario
from geoalchemy2.shape import to_shape
   
def overlay_network (network, session, overlay_scn_name, set_extendable, k_mean_clustering, start_snapshot, end_snapshot, *args, **kwargs):
    
    print('Adding overlay network ' + overlay_scn_name + ' to existing network.')
            
    if (overlay_scn_name == 'nep2035_b2' or overlay_scn_name == 'NEP') and  kwargs.get('network_cluserting'):
                print('Some transformers will have buses which are not definded due to network_clustering, they will be deleted automatically.')
                
    ### Adding overlay-network to existing network                    
    scenario = NetworkScenario(session,
                               version=None,
                               prefix='EgoGridPfHvExtension',
                               method=kwargs.get('method', 'lopf'),
                               start_snapshot=start_snapshot,
                               end_snapshot=end_snapshot,
                               scn_name='extension_' + overlay_scn_name )

    network = scenario.build_network(network)
    
    ### Set coordinates for new buses   
    extension_buses = network.buses[network.buses.scn_name =='extension_' + overlay_scn_name ]
    for idx, row in extension_buses.iterrows():
            wkt_geom = to_shape(row['geom'])
            network.buses.loc[idx, 'x'] = wkt_geom.x
            network.buses.loc[idx, 'y'] = wkt_geom.y
        
    network.transformers = network.transformers[network.transformers.bus1.astype(str).isin(network.buses.index)]
    
    ### Add load shedding at new buses
    if not network.generators[network.generators.scn_name == 'extension_' + overlay_scn_name].empty:
                start = network.generators[network.generators.scn_name == 'extension_' + overlay_scn_name].index.astype(int).max()+1
                index = list(range(start,start+len(network.buses.index[network.buses.scn_name == 'extension_' + overlay_scn_name])))
                network.import_components_from_dataframe(
                        pd.DataFrame(
                                dict(marginal_cost=100000,
                                     p_nom=network.loads_t.p_set.max().max(),
                                     carrier='load shedding',
                                     bus=network.buses.index[network.buses.scn_name == 'extension_' + overlay_scn_name]),
                                     index=index),
                                     "Generator"
                                     )
    ### Set components extendable
    if set_extendable == 'NEP Zubaunetz':
        network.lines.s_nom_extendable[(network.lines.project != 'EnLAG') & (network.lines.scn_name == 'extension_' + overlay_scn_name)]= True
        network.links.p_nom_extendable[(network.links.scn_name == 'extension_' + overlay_scn_name)] = True
       # network.transformers.s_nom_extendable[(network.transformers.project != 'EnLAG') & (network.transformers.scn_name == ('extension_' + overlay_scn_name))] = True
        
    if set_extendable == 'overlay_network_and_trafos':
        network.lines.s_nom_extendable[network.lines.scn_name == ('extension_' + overlay_scn_name)] = True
        network.links.p_nom_extendable[network.links.scn_name == ('extension_' + overlay_scn_name)] = True
        network.transformers.s_nom_extendable[network.transformers.scn_name == ('extension_' + overlay_scn_name)] = True
        
    if set_extendable == 'overlay_network':
        network.lines.s_nom_extendable[network.lines.scn_name == ('extension_' + overlay_scn_name)] = True
        network.links.p_nom_extendable[network.links.scn_name == ('extension_' + overlay_scn_name)] = True
        network.lines.capital_cost = network.lines.capital_cost +( 2 * 14166 )
            
   ### Reconnect trafos without buses due to kmean_clustering to existing buses and set s_nom_min and s_nom_max so decomissioning is not needed
    if not k_mean_clustering == False:
            network.transformers.bus0[~network.transformers.bus0.isin(network.buses.index)] = (network.transformers.bus1[~network.transformers.bus0.isin(network.buses.index)]).apply(calc_nearest_point, network = network) 
            network.lines.s_nom_max[network.lines.scn_name == ('extension_' + overlay_scn_name)] = network.lines.s_nom_max - network.lines.s_nom_min
            network.lines.s_nom_min[network.lines.scn_name == ('extension_' +  overlay_scn_name)] = 0
            network.transformers.s_nom[network.transformers.scn_name == ('extension_' + overlay_scn_name)] = 10000000
            
    else: 
       decommissioning(network, session, overlay_scn_name)
        
            
    return network
        
def decommissioning(network, session, overlay_scn_name):
    ormclass = getattr(import_module('egoio.db_tables.model_draft'), 'EgoGridPfHvExtensionLine')
    
    query = session.query(ormclass).filter(
                        ormclass.scn_name == 'decommissioning_' + overlay_scn_name)
    
    df_decommisionning = pd.read_sql(query.statement,
                         session.bind,
                         index_col='line_id')
    df_decommisionning.index = df_decommisionning.index.astype(str)
    
    ### Drop lines from existing network, if they will be decommisioned      
    network.lines = network.lines[~network.lines.index.isin(df_decommisionning.index)]

    return network

def distance (x0, x1, y0, y1):
    ### Calculate square of the distance between two points (Pythagoras)
    distance = (x1.values- x0.values)*(x1.values- x0.values) + (y1.values- y0.values)*(y1.values- y0.values)
    return distance

def calc_nearest_point(bus1, network):

    bus1_index = network.buses.index[network.buses.index == bus1]
          
    x0 = network.buses.x[network.buses.index.isin(bus1_index)]
    
    y0 = network.buses.y[network.buses.index.isin(bus1_index)]
    
    comparable_buses = network.buses[~network.buses.index.isin(bus1_index)]
  
    x1 = comparable_buses.x

    y1 = comparable_buses.y
    
    distance = (x1.values- x0.values)*(x1.values- x0.values) + (y1.values- y0.values)*(y1.values- y0.values)
    
    min_distance = distance.min()
        
    bus0 = comparable_buses[(((x1.values- x0.values)*(x1.values- x0.values) + (y1.values- y0.values)*(y1.values- y0.values)) == min_distance)  ]
    bus0 = bus0.index[bus0.index == bus0.index.max()]
    bus0 = ''.join(bus0.values)

    return bus0


    