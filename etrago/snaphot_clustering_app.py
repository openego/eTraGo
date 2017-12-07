"""
"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "Simon Hilpert"

#import os
#import pandas as pd

import numpy as np
from numpy import genfromtxt
np.random.seed()
import time
from tools.io import NetworkScenario, results_to_oedb
from tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution)
from tools.utilities import (oedb_session, load_shedding, data_manipulation_sh,
                                    results_to_csv, parallelisation, pf_post_lopf, 
                                    loading_minimization, calc_line_losses, group_parallel_lines)
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering
from cluster.snapshot_cl import snapshot_clustering, daily_bounds

args = {'network_clustering':False, #!!Fehlermeldung assert-Statement // Solved in Feature-branch
        'db': 'oedb', # db session
        'gridversion':'v0.2.11', #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': False, #state whether you want to perform a pf after a lopf simulation
        'start_snapshot': 1,
        'end_snapshot' : 48,
        'scn_name': 'SH Status Quo',
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder'
        'results': False, #'C:/eTraGo/etrago/results', # state if and where you want to save results as csv: False or '/path/tofolder'
        'export': False, # state if you want to export the results back to the database
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':True,
        'load_shedding':False,
        'generator_noise':True,
        'extra_functionality':daily_bounds,
        'k_mean_clustering': False,
        'snapshot_clustering':True,
        'parallelisation':False,
        'line_grouping': False,
        'comments': None}


def etrago(args):  
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
    
    
    if args['branch_capacity_factor']:
        network.lines.s_nom = network.lines.s_nom*args['branch_capacity_factor']
        network.transformers.s_nom = network.transformers.s_nom*args['branch_capacity_factor']
    
    if args['generator_noise']:
        # create generator noise 
        noise_values = network.generators.marginal_cost + abs(np.random.normal(0,0.001,len(network.generators.marginal_cost)))
        np.savetxt("noise_values.csv", noise_values, delimiter=",")
        noise_values = genfromtxt('noise_values.csv', delimiter=',')
        # add random noise to all generator
        network.generators.marginal_cost = noise_values
    
    if args['storage_extendable']:
        # set virtual storages to be extendable
        if network.storage_units.carrier.any()=='extendable_storage':
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
        network = kmean_clustering(network)
        
    # snapshot clustering
    if args['snapshot_clustering']:
        network = snapshot_clustering(network, how='daily', clusters= [2])
        
    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, args['results'])

    # close session
    session.close()
    
    return network
    
network = etrago(args)

