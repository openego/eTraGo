"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"

import numpy as np
from numpy import genfromtxt
np.random.seed()
import time
from etrago.tools.io import NetworkScenario, results_to_oedb
from etrago.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution)
from etrago.tools.utilities import oedb_session, load_shedding, data_manipulation_sh, results_to_csv, parallelisation, pf_post_lopf, loading_minimization, calc_line_losses, group_parallel_lines
from etrago.cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering

from etrago.cluster.disaggregation import disaggregate

args = {# Setup and Configuration:
        'db': 'oedb', # db session
        'gridversion':'v0.2.11', # None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': False, # state whether you want to perform a pf after a lopf simulation
        'start_snapshot': 1,
        'end_snapshot' : 2,
        'scn_name': 'SH Status Quo',
        'solver': 'glpk', # glpk, cplex or gurobi
        # Export options:
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder'
        'results': False, # state if and where you want to save results as csv: False or '/path/tofolder'
        'export': False, # state if you want to export the results back to the database
        # Settings:
        'storage_extendable':True, # state if you want storages to be installed at each node if necessary.
        'generator_noise':True, # state if you want to apply a small generator noise
        'reproduce_noise': False, # state if you want to use a predefined set of random noise for the given scenario. if so, provide path, e.g. 'noise_values.csv'
        'minimize_loading':False,
        # Clustering:
        'k_mean_clustering': True,
        'network_clustering': False,
        # Simplifications:
        'parallelisation':False,
        'line_grouping': False,
        'branch_capacity_factor': 0.7, #to globally extend or lower branch capacities
        'load_shedding':False,
        'comments':None }


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
        False or '/path/tofolder'.
        
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

    network_clustering (bool):
        False, 
        True or false
        
    parallelisation (bool):
        False,

    line_grouping (bool): 
        True,

    branch_capacity_factor (numeric): 
        1, 
        to globally extend or lower branch capacities
           
    load_shedding (bool):
        False,

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

    # add coordinates
    network = add_coordinates(network)
      
    # TEMPORARY vague adjustment due to transformer bug in data processing
    network.transformers.x=network.transformers.x*0.0001

    clustering = None

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
        if network.storage_units.source.any()=='extendable_storage':
            network.storage_units.p_nom_extendable = True
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
    if args['k_mean_clustering']:
        clustering = kmean_clustering(network, n_clusters=100)
        original_network = network.copy()
        network = clustering.network.copy()
        
    # Branch loading minimization
    if args['minimize_loading']:
        extra_functionality = loading_minimization
    else:
        extra_functionality=None
        
    # parallisation
    if args['parallelisation']:
        parallelisation(network, start_snapshot=args['start_snapshot'], end_snapshot=args['end_snapshot'],group_size=1, solver_name=args['solver'], extra_functionality=extra_functionality)
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'], extra_functionality=extra_functionality)
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes
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

    if clustering:
        disaggregate(scenario, original_network, network, clustering, solver=args['solver'])

    # write lpfile to path
    if not args['lpfile'] == False:
        network.model.write(args['lpfile'], io_options={'symbolic_solver_labels':
                                                     True})
    # write PyPSA results back to database
    if args['export']:
        results_to_oedb(session, network, 'hv', args)  
        
    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, args['results'])

    return network

  
# execute etrago function
network = etrago(args)

# plots

# make a line loading plot
plot_line_loading(network)
# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# plot to show extendable storages
storage_distribution(network)

# close session
#session.close()

