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

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
np.random.seed()
from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario, results_to_oedb
import time
from egopowerflow.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, gen_dist, curtailment,
                                     storage_distribution)
from extras.utilities import load_shedding, data_manipulation_sh, results_to_csv, parallelisation, pf_post_lopf, loading_minimization, marginal_cost_noise
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage


args = {'network_clustering':False,
        'db': 'oedb', # db session
        'gridversion': None, #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf':False , #state whether you want to perform a pf after a lopf simulation
        'start_h': 2315,
        'end_h' : 2316,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv'
        'lpfile': 'Output.lp', # state if and where you want to save pyomo's lp file: False or '/path/tofolder/file.lp'
        'results': 'Results.lp', # state if and where you want to save results as csv: False or '/path/tofolder'
        'export': False, # state if you want to export the results back to the database
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':True,
        'load_shedding':True,
        'generator_noise':False,
        'minimize_loading':True,
        'marginal_cost_noise':True,
        'parallelisation':False,
        'comments': None}

def etrago(args):
    session = oedb_session(args['db'])

    # additional arguments cfgpath, version, prefix
    scenario = NetworkScenario(session,
                               version=args['gridversion'],
                               prefix=args['ormcls_prefix'],
                               method=args['method'],
                               start_h=args['start_h'],
                               end_h=args['end_h'],
                               scn_name=args['scn_name'])

    network = scenario.build_network()

    # add coordinates
    network = add_coordinates(network)
    
    # TEMPORARY vague adjustment due to transformer bug in data processing
    #network.transformers.x=network.transformers.x*0.01
            
        
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
        if network.storage_units.source.any()=='extendable_storage':
            network.storage_units.p_nom_extendable = True
        # set virtual storage costs with regards to snapshot length
            network.storage_units.capital_cost = (network.storage_units.capital_cost /
            (8760//(args['end_h']-args['start_h']+1)))

    # for SH scenario run do data preperation:
    if args['scn_name'] == 'SH Status Quo':
        data_manipulation_sh(network)

    #load shedding in order to hunt infeasibilities
    if args['load_shedding']:
    	load_shedding(network)
  
    # network clustering
    if args['network_clustering']:
        network.generators.control="PV"
        busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
        network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

    # Branch loading minimization
    if args['minimize_loading']:
        extra_functionality = loading_minimization
    else:
        extra_functionality=None

    # parallisation
    if args['parallelisation']:
        parallelisation(network, start_h=args['start_h'], end_h=args['end_h'],group_size=1, solver_name=args['solver'], extra_functionality=extra_functionality)
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'], extra_functionality=extra_functionality)
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes
    # start non-linear powerflow simulation
    elif args['method'] == 'pf':
        network.pf(scenario.timeindex)
    if args['pf_post_lopf']:
        pf_post_lopf(network, scenario)
    
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

    #Create marginal cost noise
    marginal_cost_noise(network, start_h=args['start_h'], end_h=args['end_h'])
    
    if args['marginal_cost_noise']:
        #do a second calculation of the lopf
        second_scenario = NetworkScenario(session,
                               version=args['gridversion'],
                               prefix=args['ormcls_prefix'],
                               method=args['method'],
                               start_h=args['start_h'],
                               end_h=args['end_h'],
                               scn_name=args['scn_name'])
        
        second_network = second_scenario.build_network()
        second_network = add_coordinates(second_network)
          
        second_network.lines.s_nom = second_network.lines.s_nom * args['branch_capacity_factor']
        second_network.transformers.s_nom = second_network.transformers.s_nom*args['branch_capacity_factor']

        if args['scn_name'] == 'SH Status Quo':
            data_manipulation_sh(second_network)
        
        if args['storage_extendable']:
        # set virtual storages to be extendable
            if second_network.storage_units.source.any()=='extendable_storage':
                second_network.storage_units.p_nom_extendable = True
            # set virtual storage costs with regards to snapshot length
                second_network.storage_units.capital_cost = (second_network.storage_units.capital_cost /
                (8760//(args['end_h']-args['start_h']+1)))    
            
        load_shedding(second_network)    
               
        #create marginal-cost-noise
        list_new_marginal_cost = genfromtxt('list_new_marginal_cost.csv',delimiter=",")
        # add marginal-cost-noise to all generators
        second_network.generators.marginal_cost = list_new_marginal_cost
        
        if args['network_clustering']:
            second_network.generators.control="PV"
            second_busmap = busmap_from_psql(second_network, session, scn_name=args['scn_name'])
            second_network = cluster_on_extra_high_voltage(second_network, second_busmap, with_time=True)
       
        # start powerflow calculations
        x = time.time()
        second_network.lopf(second_scenario.timeindex, solver_name=args['solver'])
        y = time.time()
        z = (y - x) / 60
        
        #histogram of the change in dispatch from all generators 
        def histogram (network, second_network,carrier = 'wind', filename = None):
            
            liste_numbers=[]
            differences=[]
            if carrier is None: 
                numbers = network.generators_t.p.sum() - second_network.generators_t.p.sum()
                differences.append(numbers)
                for i in numbers: 
                    if i !=0:
                        liste_numbers.append(i)
            else:
                numbers = network.generators_t.p.sum()[network.generators.carrier == carrier] - second_network.generators_t.p.sum()[second_network.generators.carrier == carrier]
                differences.append(numbers)
                for i in numbers: 
                    if i !=0:
                        liste_numbers.append(i)
            
            plt.hist(differences, bins = 50)
            plt.title("Differences in dispatch")
            plt.xlabel("Value (difference)")
            plt.ylabel("Frequency (numbers of generators)")
            
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
                plt.close()
                
            print(len(liste_numbers), max(liste_numbers),min(liste_numbers))
                          
        histogram(network,second_network, None)
        # histogram(network,second_network,'wind')
        # histogram(network,second_network,'solar')
        
        plot_stacked_gen(second_network, resolution="MW")
        
        
    return network

  
# execute etrago function
network = etrago(args)
   
#==============================================================================
# curtailment(network,carrier="wind")
# 
# # plots
# gen_dist(network)
# 
# # make a line loading plot
# plot_line_loading(network)
#==============================================================================

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

#==============================================================================
# # plot to show extendable storages
# storage_distribution(network)
# 
#==============================================================================
# close session
#session.close()

