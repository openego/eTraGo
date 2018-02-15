<<<<<<< HEAD


=======
>>>>>>> 336fe2da9d5c08fcb462d8fd6aef8510705d28d8
"""
This is the application file for the tool eTraGo. 

Define your connection parameters and power flow settings before executing the function etrago.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"
"""
import numpy as np
from numpy import genfromtxt
np.random.seed()
import time
import datetime
import os


if not 'READTHEDOCS' in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly  
    from etrago.tools.io import NetworkScenario, results_to_oedb
    from etrago.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution, extension_overlay_network)

    from etrago.tools.utilities import (oedb_session, load_shedding, data_manipulation_sh, convert_capital_costs,
                                    results_to_csv, parallelisation, pf_post_lopf, 
                                    loading_minimization, calc_line_losses, group_parallel_lines)

    from etrago.cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering
    from etrago.cluster.snapshot import snapshot_clustering, daily_bounds
    from etrago.tools.nep import overlay_network
    
#from etrago.tools.nep import add_extension_network

args = {# Setup and Configuration:
        'db': 'local', # db session
        'gridversion': None, # None for model_draft or Version number (e.g. v0.2.11) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': False, # state whether you want to perform a pf after a lopf simulation
        'start_snapshot': 1, 
        'end_snapshot' : 744,
        'scn_name': 'NEP 2035', # state which scenario you want to run: Status Quo, NEP 2035, eGo100
        'solver': 'gurobi', # glpk, cplex or gurobi
        # Export options:
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or /path/tofolder
        'results': False, # state if and where you want to save results as csv: False or /path/tofolder
        'export': False, # state if you want to export the results back to the database
        # Settings:        

        'storage_extendable':False, # state if you want storages to be installed at each node if necessary.
        'generator_noise':True, # state if you want to apply a small generator noise 
        'reproduce_noise': False, # state if you want to use a predefined set of random noise for the given scenario. if so, provide path, e.g. 'noise_values.csv'
        'minimize_loading':False,
        # Clustering:

        'k_mean_clustering': 10, # state if you want to perform a k-means clustering on the given network. State False or the value k (e.g. 20).
        'network_clustering': False, # state if you want to perform a clustering of HV buses to EHV buses.
        'extra_functionality':False,
        'snapshot_clustering': True, # state if you want to perform snapshot_clustering on the given network. Move to PyPSA branch:features/snapshot_clustering
        # Simplifications:
        'parallelisation': False, 
	    'skip_snapshots':False,
        'line_grouping': False,
        'branch_capacity_factor': 0.7, #to globally extend or lower branch capacities
        'load_shedding':True,
        'comments':None,
        # Scenario variances
        'overlay_network': 'NEP', # None or new scenario name e.g. 'NEP' 
        'add_Belgium_Norway':True,  # state if you want to add Belgium and Norway as electrical neighbours, only NEP 2035
        'set_extendable' :'overlay_network' # None or wich part of NEP-scenario you want to set extandable  (NEP Zubaunetz)

        }



def etrago(args):
    """The etrago function works with following arguments:
    
    
    Parameters
    ----------
           
    db (str): 
    	'oedb', 
        Name of Database session setting stored in config.ini of oemof.db
        
    gridversion (str):
        'v0.2.11', 
        Name of the data version number of oedb: state 'None' for 
        model_draft (sand-box) or an explicit version number 
        (e.g. 'v0.2.10') for the grid schema.
         
    method (str):
        'lopf', 
        Choose between a non-linear power flow ('pf') or
        a linear optimal power flow ('lopf').
        
    pf_post_lopf (bool): 
        False, 
        Option to run a non-linear power flow (pf) directly after the 
        linear optimal power flow (and thus the dispatch) has finished.
                
    start_snapshot (int):
    	1, 
        Start hour of the scenario year to be calculated.
        
    end_snapshot (int) : 
    	2,
        End hour of the scenario year to be calculated.
        
    scn_name (str): 
    	'Status Quo',
	Choose your scenario. Currently, there are three different 
	scenarios: 'Status Quo', 'NEP 2035', 'eGo100'. If you do not 
	want to use the full German dataset, you can use the excerpt of 
	Schleswig-Holstein by adding the acronym SH to the scenario 
	name (e.g. 'SH Status Quo').
        
    solver (str): 
        'glpk', 
        Choose your preferred solver. Current options: 'glpk' (open-source),
        'cplex' or 'gurobi'.
                
    lpfile (obj): 
        False, 
        State if and where you want to save pyomo's lp file. Options:
        False or '/path/tofolder'.import numpy as np
        
    results (obj): 
        False, 
        State if and where you want to save results as csv files.Options: 
        False or '/path/tofolder'.
        
    export (bool): 
        False, 
        State if you want to export the results of your calculation 
        back to the database.
        
    storage_extendable (bool):
        True,
        Choose if you want to allow to install extendable storages 
        (unlimited in size) at each grid node in order to meet the flexibility demand. 
        
    generator_noise (bool):
        True,
        Choose if you want to apply a small random noise to the marginal 
        costs of each generator in order to prevent an optima plateau.
        
    reproduce_noise (obj): 
        False, 
        State if you want to use a predefined set of random noise for 
        the given scenario. If so, provide path to the csv file,
        e.g. 'noise_values.csv'.
        
    minimize_loading (bool):
        False,
        
    k_mean_clustering (bool): 
        False,
        State if you want to apply a clustering of all network buses down to 
        only 'k' buses. The weighting takes place considering generation and load
        at each node. If so, state the number of k you want to apply. Otherwise 
        put False. This function doesn't work together with 'line_grouping = True'
	    or 'network_clustering = True'.
    
    network_clustering (bool):
        False, 
        Choose if you want to cluster the full HV/EHV dataset down to only the EHV 
        buses. In that case, all HV buses are assigned to their closest EHV sub-station, 
        taking into account the shortest distance on power lines.
        
    parallelisation (bool):
        False,
        Choose if you want to calculate a certain number of snapshots in parallel. If
        yes, define the respective amount in the if-clause execution below. Otherwise 
        state False here.
        
    line_grouping (bool): 
        True,
        State if you want to group lines that connect the same two buses into one system.
   
    branch_capacity_factor (numeric): 
        1, 
        Add a factor here if you want to globally change line capacities (e.g. to "consider"
        an (n-1) criterion or for debugging purposes.
           
    load_shedding (bool):
        False,
        State here if you want to make use of the load shedding function which is helpful when
        debugging: a very expensive generator is set to each bus and meets the demand when regular
        generators cannot do so.
        
    comments (str): 
        None
        
    Result:
    -------
        

    """

    
    session = oedb_session(args['db'])

    # additional arguments cfgpath, version, prefix
    if args['gridversion'] == None:
        args['ormcls_prefix'] = 'EgoGridPfHv'
    else:
        args['ormcls_prefix'] = 'EgoPfHv'
        
    scenario = NetworkScenario(session,
                               version=args['gridversion'],
                               prefix=args['ormcls_prefix'],
                               method=args['method'],
                               start_snapshot=args['start_snapshot'],
                               end_snapshot=args['end_snapshot'],
                               scn_name=args['scn_name'])

    network = scenario.build_network()

    network.links.marginal_cost = 0

    # add coordinates
    network = add_coordinates(network)
      
    # TEMPORARY vague adjustment due to transformer bug in data processing
    network.transformers.x=network.transformers.x*0.0001


    # adjust capital_cost to annual payment and simulation period
    network= convert_capital_costs(network, args['start_snapshot'], args['end_snapshot'])

    # TEMPORARY vague adjustment due to transformer bug in data processing     
    if args['gridversion'] == 'v0.2.11':
        network.transformers.x=network.transformers.x*0.0001


    if args['branch_capacity_factor']:
        
        
        network.lines.s_nom = network.lines.s_nom*args['branch_capacity_factor']
        network.transformers.s_nom = network.transformers.s_nom*args['branch_capacity_factor']

    if args['generator_noise']:
        # create or reproduce generator noise 
        if not args['reproduce_noise'] == False:    
            noise_values = genfromtxt('noise_values.csv', delimiter=',')
            # add random noise to all generator
            network.generators.marginal_cost = noise_values
        else:
            noise_values = network.generators.marginal_cost + abs(np.random.normal(0,0.001,len(network.generators.marginal_cost)))
            np.savetxt("noise_values.csv", noise_values, delimiter=",")
            noise_values = genfromtxt('noise_values.csv', delimiter=',')
            # add random noise to all generator
            network.generators.marginal_cost = noise_values
      
      
    if args['storage_extendable']:
        # set virtual storages to be extendable
        if network.storage_units.carrier[network.storage_units.carrier== 'extendable_storage'].any() == 'extendable_storage':
            network.storage_units.loc[network.storage_units.carrier=='extendable_storage','p_nom_extendable'] = True
        # set virtual storage costs with regards to snapshot length
            network.storage_units.capital_cost = (network.storage_units.capital_cost /
            (8760//(args['end_snapshot']-args['start_snapshot']+1)))

    # for SH scenario run do data preperation:
    if args['scn_name'] == 'SH Status Quo' or args['scn_name'] == 'SH NEP 2035':
        data_manipulation_sh(network)
      
    # grouping of parallel lines
    if args['line_grouping']:
        group_parallel_lines(network)

    #load shedding in order to hunt infeasibilities
    if args['load_shedding']:
    	load_shedding(network)

    # network clustering
    if args['network_clustering']:
        network.generators.control="PV"
        busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
        network = cluster_on_extra_high_voltage(network, busmap, with_time=True)
    
  
    # k-mean clustering
    if not args['k_mean_clustering'] == False:
        network = kmean_clustering(network, n_clusters=args['k_mean_clustering'])
        
    # Branch loading minimization
    if args['minimize_loading']:
        extra_functionality = loading_minimization
    else:
        extra_functionality = None

    
    if args['skip_snapshots']:
        network.snapshots=network.snapshots[::args['skip_snapshots']]
        network.snapshot_weightings=network.snapshot_weightings[::args['skip_snapshots']]*args['skip_snapshots']   
        
    if args ['overlay_network'] != None:
         network = overlay_network(network, session, overlay_scn_name = args ['overlay_network'], set_extendable = args ['set_extendable'], k_mean_clustering = args ['k_mean_clustering'],start_snapshot=args['start_snapshot'], end_snapshot=args['end_snapshot'])
         network = convert_capital_costs(network, args['start_snapshot'], args['end_snapshot'])

    if args ['add_Belgium_Norway']:
         network = overlay_network(network, session, overlay_scn_name = 'BE_NO_NEP 2035', set_extendable = args ['set_extendable'],k_mean_clustering = args ['k_mean_clustering'], start_snapshot=args['start_snapshot'], end_snapshot=args['end_snapshot'] )
         network= convert_capital_costs(network, args['start_snapshot'], args['end_snapshot'])

    # snapshot clustering
    if not args['snapshot_clustering']==False:
        extra_functionality = None #daily_bounds
        x = time.time()
        network = snapshot_clustering(network, how='daily', clusters= [5,10,15,20,25,30,35,40,45,50,100,200,300])
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes

    # parallisation
    if args['parallelisation']:
        parallelisation(network, start_snapshot=args['start_snapshot'], end_snapshot=args['end_snapshot'],group_size=1,
                        solver_name=args['solver'],  solver_options={'threads':2, 'method':2, 'crossover':0, 'BarConvTol':1.e-5,'FeasibilityTol':1.e-6},  extra_functionality=extra_functionality)
    
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(network.snapshots, solver_name=args['solver'], extra_functionality=None)
        y = time.time()
        z = (y - x) / 60 
        print('Time for lopf in minutes')
        print(z)# z is time for lopf in minutes
    # start non-linear powerflow simulation
    elif args['method'] == 'pf':
        network.pf(scenario.timeindex)
       # calc_line_losses(network)
        
    if args['pf_post_lopf']:
        pf_post_lopf(network, scenario)
        calc_line_losses(network)
    
       # provide storage installation costs
    if sum(network.storage_units.p_nom_opt) != 0:
        installed_storages = network.storage_units[ network.storage_units.p_nom_opt!=0]
        storage_costs = sum(installed_storages.capital_cost * installed_storages.p_nom_opt)
        print("Investment costs for all storages in selected snapshots [EUR]:",round(storage_costs,2))   
        
    # write lpfile to path
    if not args['lpfile'] == False:
        network.model.write(args['lpfile'], io_options={'symbolic_solver_labels':True})
        
    # write PyPSA results back to database
    if args['export']:
        results_to_oedb(session, network, args, 'hv')  
        
    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, '/home/clara/Schreibtisch/network')

    # close session
    session.close()

    return network



# close session
#session.close()


if __name__ == '__main__':
    # execute etrago function
    print(datetime.datetime.now())
    network = etrago(args)
    print(datetime.datetime.now())
    # plots
    # make a line loading plot
    #plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    #plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    #storage_distribution(network)
    #extension_overlay_network(network)
