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
from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario
import time
from egopowerflow.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist)#,
                                    # storage_distribution)
from extras.utilities import load_shedding, data_manipulation_sh, results_to_csv, parallelisation, pf_post_lopf
#from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage
from plotting import plot_max_line_loading,plot_max_opt_line_loading,transformers_distribution

from oemof.db import cfg

# only load config file
cfg.load_config('/home/mario-arbeit/Dokumente/config.ini')  

args = {'network_clustering':False,
        'db': 'home', # db session
        'gridversion': None, #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf':False , #state whether you want to perform a pf after a lopf simulation
        'start_h': 2233,
        'end_h' : 2401,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv'
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder/file.lp'
        'results': False, # state if and where you want to save results as csv: False or '/path/tofolder'
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 0.1, #to globally extend or lower branch capacities
        'storage_extendable':False,
        'load_shedding':True,
        'generator_noise':True,
        'parallelisation':False,
        'line_extendable': True,
        'calc_type' : True} # True for methodik of line_extendable  #False for all lines are extendables

def etrago(args):
    start_time = time.time()
        
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
 #   if args['network_clustering']:
    #    network.generators.control="PV"
   #     busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
   #     network = cluster_on_extra_high_voltage(network, busmap, with_time=True)
        
    # line optimization    
    if args['line_extendable']:
        h = args['calc_type']
        
        if (h == True):
            # to double the maximum capacity 
            network.lines.s_nom = network.lines.s_nom*1.2
            network.transformers.s_nom = network.transformers.s_nom*1.2
            #first lopf to select the right lines
            network.lopf(scenario.timeindex, solver_name=args['solver'])
       
            # Rechange the maximum capacity to original value       
            network.lines.s_nom = network.lines.s_nom/1.2
            network.transformers.s_nom =network.transformers.s_nom/1.2
            

            plot_max_line_loading(network,filename='line_maximum_loading_meth_start.jpg')
            ###################################################################
            
            # List for counting the maximum value for timesteps  
            # 0 : line keys 
            # 1 : maximum loadings of each line
            # 2 : timestep of maximum loading
            max_loading = [[],[],[]]   
            
            # List for counting the maximum loadings of each timestep
            # 0 : timestep index of line
            # 1 : Number of maximum loadings 
            timesteps = [[],[]]
            
            # The same list like timesteps but for sorting by maximum number of loadings
            time_list =[]
                        
            i=0
            while (i<len(network.lines_t.p0.keys())):
                max_loading[0].append(network.lines_t.p0.keys()[i])
                max_loading[1].append(max(abs(network.lines_t.p0[max_loading[0][i]])/network.lines.s_nom[max_loading[0][i]]*100))
                p = abs(network.lines_t.p0[max_loading[0][i]])
                x= 0
                while(x<len(p)):
                    if(p[x]==(max(abs(network.lines_t.p0[max_loading[0][i]])))):
                        # set the timestep of maximum loading        
                        max_loading[2].append(x)
                        break
                    else:
                        x+=1
                            
            
                value = max_loading[1][i]
                time_index = max_loading[2][i]
            
                if (((time_index in timesteps[0]) == True) and (value >= 100)):
                    index = timesteps[0].index(time_index)
                    timesteps[1][index] +=1
                elif(value >= 100):
                    timesteps[0].append(time_index)
                    timesteps[1].append(1) 
                    
                i+=1
            
            x=0
            while(x<len(timesteps[0])):
                time_list.append([timesteps[0][x],timesteps[1][x]])
                x+=1
                    
            def getKey(item):
                return item[1]
        
            time_list = sorted(time_list,key=getKey)
        
            lines_time=[]
            i = 0
            while(i<len(time_list)):
#                if(i > 0.1*(args['end_h']-args['start_h'])):
#                    break
                lines_time.append(time_list[len(time_list)-1-i][0])  
            
                index = [a for a,u in enumerate(max_loading[2]) if u==lines_time[i]]
                
                for k in index:
                    network.lines.s_nom_extendable[max_loading[0][k]] = True
                    network.lines.s_nom_min[max_loading[0][k]] = network.lines.s_nom[max_loading[0][k]]
                    network.lines.s_nom_max[max_loading[0][k]] = np.inf
                    
                    name_bus_0 = network.lines.bus0[max_loading[0][k]]
                    name_bus_1 = network.lines.bus1[max_loading[0][k]]
                    
                    U_bus_0 = network.buses.v_nom[name_bus_0]
                    U_bus_1 = network.buses.v_nom[name_bus_1]
                    
                    if(U_bus_0 == U_bus_1):
                        if(U_bus_0 == 110):
                            network.lines.capital_cost[max_loading[0][k]] = \
                                (60000*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                                (50*8760)*(args['end_h']-args['start_h'])
                        elif(U_bus_0 == 220):
                            network.lines.capital_cost[max_loading[0][k]] = \
                                (1600000*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                                (50*8760)*(args['end_h']-args['start_h'])
                        else:
                            network.lines.capital_cost[max_loading[0][k]] = \
                                (200000*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                                (50*8760)*(args['end_h']-args['start_h'])
                    else:
                        print('Error')
                i+=1
        
            ########################## Transformator ############################
            
            # List for counting the maximum value for timesteps  
            # 0 : line keys 
            # 1 : maximum loadings of each line
            # 2 : timestep of maximum loading
            max_loading_trafo = [[],[],[]]   
        
            # List for counting the maximum loadings of each timestep
            # 0 : timestep index
            # 1 : Number of maximum loadings
            timesteps_trafo = [[],[]]
        
            # The same list as timesteps but for sorting by maximum number of loadings
            time_list_trafo =[]
        
            i=0
            while (i<len(network.transformers_t.p0.keys())):
                max_loading_trafo[0].append(network.transformers_t.p0.keys()[i])
                max_loading_trafo[1].append(max(abs(network.transformers_t.p0[max_loading_trafo[0][i]])/network.transformers.s_nom[max_loading_trafo[0][i]]*100))
                p = abs(network.transformers_t.p0[max_loading_trafo[0][i]])
                x= 0
                while(x<len(p)):
                    if(p[x]==(max(abs(network.transformers_t.p0[max_loading_trafo[0][i]])))):
                        # set the timestep of maximum loading                         
                        max_loading_trafo[2].append(x)
                        break
                    else:
                        x+=1
                           
                            
            
                value = max_loading_trafo[1][i]
                time_index = max_loading_trafo[2][i]
            
                if (((time_index in timesteps_trafo[0]) == True) and (value >= 100)):
                    index = timesteps_trafo[0].index(time_index)
                    timesteps_trafo[1][index] +=1
                elif(value >= 100):
                    timesteps_trafo[0].append(time_index)
                    timesteps_trafo[1].append(1) 
            
                i+=1
        
        
            x=0
            while(x<len(timesteps_trafo[0])):
                time_list_trafo.append([timesteps_trafo[0][x],timesteps_trafo[1][x]])
                x+=1
        
            time_list_trafo = sorted(time_list_trafo,key=getKey)
        
            # List of choosen timesteps
            trafo=[]
        
            i = 0
            while(i<len(time_list_trafo)):
#                if(i > 0.1*(args['end_h']-args['start_h'])):
#                    break
                trafo.append(time_list_trafo[len(time_list_trafo)-1-i][0])  
            
                index = [a for a,u in enumerate(max_loading_trafo[2]) if u==trafo[i]]
                for k in index:
                    network.transformers.s_nom_extendable[max_loading_trafo[0][k]] = True
                    network.transformers.s_nom_min[max_loading_trafo[0][k]] = network.transformers.s_nom[max_loading_trafo[0][k]]
                    network.transformers.s_nom_max[max_loading_trafo[0][k]] = np.inf
                    
                    name_bus_0 = network.transformers.bus0[max_loading_trafo[0][k]]
                    name_bus_1 = network.transformers.bus1[max_loading_trafo[0][k]]
                    
                    U_bus_0 = network.buses.v_nom[name_bus_0]
                    U_bus_1 = network.buses.v_nom[name_bus_1]
                
                    U_OS = max(U_bus_0,U_bus_1)
                    U_US = min(U_bus_0,U_bus_1)
                                    
                    if((U_OS == 220 and U_US == 110) or (U_OS == 380 and U_US == 110)):
                        network.transformers.capital_cost[max_loading_trafo[0][k]] = \
                            ((5200000/300)/(40*8760)*(args['end_h']-args['start_h']))
                    elif(U_OS == 380 and U_US == 220):
                        network.transformers.capital_cost[max_loading_trafo[0][k]] = \
                            ((8500000/600)/(40*8760)*(args['end_h']-args['start_h']))
                    else:
                        network.transformers.capital_cost[max_loading_trafo[0][k]] = \
                            ((8500000/600)/(40*8760)*(args['end_h']-args['start_h']))                        
                        print('Other Transformator' + str(k))
            
                i+=1

            i=0
            while(i<len(trafo)):
                if(trafo[i] in lines_time == True):
                    i+=1
                else:
                    lines_time.append(trafo[i])
                    i+=1
        
        
            for k in lines_time:
                network.lopf(network.snapshots[k:k+1], solver_name=args['solver'])
 
        else:
                
            network.lines.s_nom_extendable = True
            network.lines.s_nom_min = network.lines.s_nom
            network.lines.s_nom_max = np.inf
            
            i=0
            while(i<len(network.lines)):
                name_bus_0 = network.lines.bus0[i]
                name_bus_1 = network.lines.bus1[i]
                    
                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]
                    
                if(U_bus_0 == U_bus_1):
                    if(U_bus_0 == 110):
                        network.lines.capital_cost[i] = \
                            (60000*network.lines.length[i]/network.lines.s_nom[i])/\
                            (50*8760)*(args['end_h']-args['start_h'])
                    elif(U_bus_0 == 220):
                        network.lines.capital_cost[i] = \
                            (1600000*network.lines.length[i]/network.lines.s_nom[i])/\
                            (50*8760)*(args['end_h']-args['start_h'])
                    else:
                        network.lines.capital_cost[i] = \
                            (200000*network.lines.length[i]/network.lines.s_nom[i])/\
                            (50*8760)*(args['end_h']-args['start_h'])
                else:
                    print('Error')
                
                i+=1
                    
            ####################### Transformer #############################        
            network.transformers.s_nom_extendable = True
            network.transformers.s_nom_min = network.transformers.s_nom
            network.transformers.s_nom_max = np.inf
            
            i=0
            while(i<len(network.transformers)):
                name_bus_0 = network.transformers.bus0[i]
                name_bus_1 = network.transformers.bus1[i]
                    
                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]
                
                U_OS = max(U_bus_0,U_bus_1)
                U_US = min(U_bus_0,U_bus_1)
                                    
                if((U_OS == 220 and U_US == 110) or (U_OS == 380 and U_US == 110)):
                    network.transformers.capital_cost[i] = \
                        ((5200000/300)/(40*8760)*(args['end_h']-args['start_h']))
                elif(U_OS == 380 and U_US == 220):
                    network.transformers.capital_cost[i] = \
                        ((8500000/600)/(40*8760)*(args['end_h']-args['start_h']))
                else:
                    network.transformers.capital_cost[i] = \
                        ((8500000/600)/(40*8760)*(args['end_h']-args['start_h']))
                    print('Other Transformator' + str(i))
                
                i+=1            
            
           
            network.lopf(scenario.timeindex, solver_name=args['solver'])
            
    # parallisation
    elif args['parallelisation']:
        parallelisation(network, start_h=args['start_h'], end_h=args['end_h'],group_size=1, solver_name=args['solver'])
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'])
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
    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, args['results'])

    
    return network,start_time,lines_time,max_loading,max_loading_trafo



# execute etrago function
#network,start_time = etrago(args)
# execute etrago function
network,start_time,lines_time,max_loading,max_loading_trafo = etrago(args)
end_time = time.time()
z = (end_time - start_time) / 60 # z is time for lopf in minutes
# plots

# make a line loading plot
plot_line_loading(network,filename='line_maximum_loading_meth_2_vergleich.jpg')

plot_max_line_loading(network,filename='line_maximum_loading_meth_2.jpg')

plot_max_opt_line_loading(network,filename='line_maximum_loading_meth_opt.jpg')

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# plot to show extendable transformers
transformers_distribution(network,filename='plot_transformer_meth.jpg')
# plot to show extendable storages
#storage_distribution(network)

# close session
#session.close()
import csv
  
with open('lines_meth.csv', 'w') as csvfile:
    fieldnames = ['line_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    i = 0
    while (i<len(network.lines.s_nom)):    
        writer.writerow({'line_key': network.lines.s_nom.keys()[i],
                         's_nom_extendable': network.lines.s_nom_extendable[i],
                         's_nom': network.lines.s_nom[i],
                         'p': max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]])),
                         'loading_old': round(max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]]))/network.lines.s_nom[i]*100,2),
                         's_nom_opt':network.lines.s_nom_opt[i],
                         'loading_new': round(max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]]))/network.lines.s_nom_opt[i]*100,2),
                         'dif': network.lines.s_nom_opt[i]-network.lines.s_nom[i]}) 
        i+=1
          


with open('transformer_meth.csv', 'w') as csvfile:
    fieldnames = ['transformer_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    i = 0
    while (i<len(network.transformers.s_nom)):    
        writer.writerow({'transformer_key': network.transformers.s_nom.keys()[i],
                         's_nom_extendable': network.transformers.s_nom_extendable[i],
                         's_nom': network.transformers.s_nom[i],
                         'p': max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]])),
                         'loading_old': round(max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]]))/network.transformers.s_nom[i]*100,2),
                         's_nom_opt':network.transformers.s_nom_opt[i],
                         'loading_new': round(max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]]))/network.transformers.s_nom_opt[i]*100,2),
                         'dif': network.transformers.s_nom_opt[i]-network.transformers.s_nom[i]}) 
        i+=1          
