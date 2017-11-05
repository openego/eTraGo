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

############################### import packages ##############################
from oemof.db import cfg
import time
from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario
from egopowerflow.tools.plot import add_coordinates,plot_stacked_gen
import numpy as np
from numpy import genfromtxt
from extras.utilities import data_manipulation_sh,load_shedding,parallelisation,pf_post_lopf,results_to_csv
#from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage
from line_extendable_functions import capacity_factor,overload_lines,overload_trafo,set_line_cost,set_trafo_cost
from plotting import plot_max_line_loading,plot_max_opt_line_loading,plot_max_opt_line_loading_bench,transformers_distribution,plot_dif_line_MW,plot_dif_line_percent
import pandas as pd 
  

##############################################################################
############################### START Programm ###############################
##############################################################################

# load config file
cfg.load_config('/home/mario-arbeit/Dokumente/config.ini')  

np.random.seed()

############################### Variable inputs ##############################

args = {'network_clustering':False,
        'db': 'home', # db session
        'gridversion': None, #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf':False , #state whether you want to perform a pf after a lopf simulation
        'start_h': 1,
        'end_h' : 12,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv'
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder/file.lp'
        'results': False, # state if and where you want to save results as csv: False or '/path/tofolder'
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 0.4, #to globally extend or lower branch capacities
        'storage_extendable':False,
        'load_shedding':True,
        'generator_noise':True,
        'parallelisation':False,
        'line_extendable': True, # allow extandables lines and transformators
        }
        
##############################################################################        
######################### START Calculation Function #########################
##############################################################################
        
def etrago(args):
    
    ############################ network creation ############################

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
    
    ######################### branch_capacity_factor #########################
    
    if args['branch_capacity_factor']:
        network.lines.s_nom = network.lines.s_nom*args['branch_capacity_factor']
        network.transformers.s_nom = network.transformers.s_nom*args['branch_capacity_factor']
        
    ############################# generator_noise ############################
    if args['generator_noise']:
        # create generator noise 
        noise_values = network.generators.marginal_cost + abs(np.random.normal(0,0.001,len(network.generators.marginal_cost)))
        np.savetxt("noise_values.csv", noise_values, delimiter=",")
        noise_values = genfromtxt('noise_values.csv', delimiter=',')
    
        # add random noise to all generator
        network.generators.marginal_cost = noise_values    
    
    ########################### storage_extendable ###########################
    
    if args['storage_extendable']:
        # set virtual storages to be extendable
        if network.storage_units.source.any()=='extendable_storage':
            network.storage_units.p_nom_extendable = True
        # set virtual storage costs with regards to snapshot length
            network.storage_units.capital_cost = (network.storage_units.capital_cost /
            (8760//(args['end_h']-args['start_h']+1)))
    
    ############################## SH-scenario ###############################
    
    # for SH scenario run do data preperation:
    if args['scn_name'] == 'SH Status Quo':
        data_manipulation_sh(network)
        
    ############################## load_shedding #############################
        
    #load shedding in order to hunt infeasibilities
    if args['load_shedding']:
    	load_shedding(network)
     
     ########################### network_clustering ##########################
     
     # network clustering
#    if args['network_clustering']:
#        network.generators.control="PV"
#        busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
#        network = cluster_on_extra_high_voltage(network, busmap, with_time=True)
     
     ########################### line_optimization ###########################
     
    if args['line_extendable']:
                          
        # set the capacity-factory for the first lopf
        cap_fac = 1.3
             
        # Change the capcity of lines and transformers
        network = capacity_factor(network,cap_fac)
                                            
        ############################ 1. Lopf ###########################
        parallelisation(network, start_h=args['start_h'], \
            end_h=args['end_h'],group_size=1, solver_name=args['solver'])
             
            
        # return to original capacities
        network = capacity_factor(network,(1/cap_fac))
            
            
        # plotting the loadings of lines at start
        plot_max_line_loading(network,filename = 'Start_maximum_line_loading.jpeg')
             
             
        ############################ Analyse ############################
             
        # Finding the overload lines and timesteps
        maximum_line_loading,line_time_list = overload_lines(network)
             
        # Finding the overload transformers and timestep
        maximum_trafo_loading,trafo_time_list = overload_trafo(network)
                         
        ####################### Set capital cost ########################
             
        # Set capital cost for extendable lines
        cost_1 = 60000 # 110kV extendable
        cost_2 = 1600000/2 # 220kV extendable
        cost_3 = 200000 # 380kV extendable
            
        network,lines_time,all_time = set_line_cost(network,\
                                                    line_time_list,\
                                                    maximum_line_loading,\
                                                    cost_1,\
                                                    cost_2,\
                                                    cost_3)
             
             
        # Set capital cost for extendable trafo
        cost_1 = 5200000/300 # 220/110kV or 380/110kV extendable
        cost_2 = 8500000/600# 380/220kV extendable
        cost_3 = 8500000/600 # other extendable
             
        network,trafo_time = set_trafo_cost(network,\
                                            trafo_time_list,\
                                            maximum_trafo_loading,\
                                            cost_1,\
                                            cost_2,\
                                            cost_3)
            
             
        ####################### Set all timesteps #######################
        all_time.sort() 
        i=0
        while(i<len(trafo_time)):
            if((trafo_time[i] in all_time) == True):
                i+=1
            else:
                all_time.append(trafo_time[i])
                i+=1
                    
        ######################### calc 2. Lopf ##########################
        length_time = len(all_time)
        if(length_time==0):
            timeindex = scenario.timeindex
            
        network.lines.capital_cost =\
                                    network.lines.capital_cost * length_time

        network.transformers.capital_cost =\
                             network.transformers.capital_cost * length_time
        
        all_time.sort()
        i=0
        while(i<len(all_time)):
            if i==0:
                timeindex = network.snapshots[all_time[i]:all_time[i]+1]
            else:
                timeindex =pd.DatetimeIndex.append(timeindex,\
                          other=network.snapshots[all_time[i]:all_time[i]+1])
            i+=1
               
        network.lopf(timeindex, solver_name=args['solver'])            
                        
            
        ##################### Plotting the Results #####################
        plot_max_opt_line_loading(network,lines_time,\
                                      filename='maximum_optimal_lines.jpeg')
             
        return network
                  
           

    ############################# parallisation ##############################
    elif args['parallelisation']:
        parallelisation(network, start_h=args['start_h'], end_h=args['end_h'],group_size=1, solver_name=args['solver'])
    
    ############## start linear optimal powerflow calculations ###############
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'])
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes
        
    ################# start non-linear powerflow simulation ##################
    elif args['method'] == 'pf':
        network.pf(scenario.timeindex)
        
    ###########################################################################    
    if args['pf_post_lopf']:
        pf_post_lopf(network, scenario)

    ########################## write lpfile to path ##########################
    if not args['lpfile'] == False:
        network.model.write(args['lpfile'], io_options={'symbolic_solver_labels':
                                                     True})
    ################### write PyPSA results to csv to path ###################
    if not args['results'] == False:
        results_to_csv(network, args['results'])            
        

network = etrago(args)
    
print('finish')


# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")
             
# plot to show extendable transformers
transformers_distribution(network,filename='transformers_distribution.jpeg')    

# plot the extendables power 
plot_dif_line_MW(network,filename='extendables_lines.jpeg')

# plot the extendables power in percent
plot_dif_line_percent(network,filename='extendables_lines_percent.jpeg')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
