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
import csv
from math import sqrt 
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
        'end_h' : 24,
        'scn_name': 'NEP 2035',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv'
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder/file.lp'
        'results': False, # state if and where you want to save results as csv: False or '/path/tofolder'
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':False,
        'load_shedding':True,
        'generator_noise':True,
        'parallelisation':False,
        'line_extendable': True,
        'calc_type' : True,# True for methodik of line_extendable  #False for all lines are extendables
        'line_ext_vers' : '5_DE_NEP2035_24h_1.3' }
        
##############################################################################        
######################### START Calculation Function #########################
##############################################################################
        
def etrago(args):
    
    ############################# Set start time #############################
    start_time = time.time()
    
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
        
        ############################## methode ##############################
        # If calc_type == True -> Methodik
        if args['calc_type']:
            file_name_method = 'method'   
             
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
            filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.jpg'
            plot_max_line_loading(network,filename = filename)
             
             
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
                    
            ####################### Creation of Lists #######################
            
#            # For saving the optimization capacities of lines
#            opt_line_capacities = []
#            
#            i=0
#            while(i<len(network.lines.s_nom)):
#                opt_line_capacities.append([])
#                i+=1
#    
#            # For saving the optimization capacities of transformers
#            opt_trafo_capacities = []
#             
#            i=0
#            while(i<len(network.transformers.s_nom)):
#                opt_trafo_capacities.append([])
#                i+=1    
   
            ######################### calc 2. Lopf ##########################
            length_time = len(all_time)
            if(length_time==0):
                timeindex = scenario.timeindex
            network.lines.capital_cost = network.lines.capital_cost * length_time
            network.transformers.capital_cost = network.transformers.capital_cost * length_time
            all_time.sort()
            i=0
            while(i<len(all_time)):
                if i==0:
                    timeindex = network.snapshots[all_time[i]:all_time[i]+1]
                else:
                    timeindex = pd.DatetimeIndex.append(timeindex,other=network.snapshots[all_time[i]:all_time[i]+1])
                i+=1
               
            network.lopf(timeindex, solver_name=args['solver'])            
                        
            
            
            
#            i=0
#            while(i<len(all_time)):
#                network.lopf(network.snapshots[all_time[i]:all_time[i]+1], solver_name=args['solver'])
#                objective[0][all_time[i]] = network.objective
#                x=0
#                while(x<len(network.lines.s_nom_opt)):
#                    opt_line_capacities[x].append(network.lines.s_nom_opt[x])
#                    x+=1
#                y=0
#                while(y<len(network.transformers.s_nom_opt)):
#                    opt_trafo_capacities[y].append(network.transformers.s_nom_opt[y])
#                    y+=1
#                i+=1
#                
             ############ Set the maximum optimization capacity #############
             
#            i = 0
#            while(i<len(network.lines.s_nom_opt)):
#                s_nom_opt = max(opt_line_capacities[i])
#                network.lines.s_nom_opt[i]=s_nom_opt
#                i+=1
#           
#            i = 0
#            while(i<len(network.transformers.s_nom_opt)):
#                s_nom_opt = max(opt_trafo_capacities[i])
#                network.transformers.s_nom_opt[i]=s_nom_opt
#                i+=1
            ##################### Plotting the Results #####################
            filename = args['line_ext_vers'] + '_02_Opt_line_maximum_loading_' + file_name_method +'.jpg'    
            plot_max_opt_line_loading(network,lines_time,filename=filename)
             
            ################### Saving Datas in csv ########################
            filename = args['line_ext_vers'] + '_01_Lines_' + file_name_method +'.csv'
             
            with open(filename, 'w') as csvfile:
                fieldnames = ['line_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
    
                i = 0
                while (i<len(network.lines.s_nom)):
                    p = []
                    q = []
                    s_current = []
                    x=0
                    while(x<len(lines_time)):
                        p.append(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]].loc[network.snapshots[lines_time[x]]]))
                        if network.lines_t.q0.empty:
                            q.append(0)
                            s_current.append(p[x])
                        else:
                            q.append(abs(network.lines_t.q0[network.lines_t.q0.keys()[i]].loc[network.snapshots[lines_time[x]]]))
                            s_current.append(abs(sqrt(p[x]**2+q[x]**2)))
                        x+=1
           
                    
        
                    s_nom = network.lines.s_nom_opt[i]
                    loading=(max(s_current)/s_nom*100)
    
                    writer.writerow({'line_key': network.lines.s_nom.keys()[i],
                                     's_nom_extendable': network.lines.s_nom_extendable[i],
                                     's_nom': network.lines.s_nom[i],
                                     'p': max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]])),
                                     'loading_old': round(max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]]))/network.lines.s_nom[i]*100,2),
                                     's_nom_opt':network.lines.s_nom_opt[i],
                                     'loading_new': round(loading,2),
                                     'dif': network.lines.s_nom_opt[i]-network.lines.s_nom[i]}) 
                    i+=1
              
            # Transformers
            filename = args['line_ext_vers'] + '_02_Transformers_' + file_name_method +'.csv'
            with open(filename, 'w') as csvfile:
                fieldnames = ['transformer_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()    
    
    
                i = 0
                while (i<len(network.transformers.s_nom)):
                    p = []
                    q = []
                    s_current = []
                    x=0
                    while(x<len(trafo_time)):
                        p.append(abs(network.transformers_t.p0[network.transformers_t.p0.keys()[i]].loc[network.snapshots[trafo_time[x]]]))
                        if network.transformers_t.q0.empty:
                            s_current.append(p[x])
                        else:
                            q.append(abs(network.transformers_t.q0[network.transformers_t.q0.keys()[i]].loc[network.snapshots[trafo_time[x]]]))
                            s_current.append(abs(sqrt(p[x]**2+q[x]**2)))
                        x+=1
                    
                    if(len(s_current)==0):
                        s_current=[0]
                        
                    s_nom = network.transformers.s_nom_opt[i]
                    loading=(max(s_current)/s_nom*100)
        
        
                    writer.writerow({'transformer_key': network.transformers.s_nom.keys()[i],
                                     's_nom_extendable': network.transformers.s_nom_extendable[i],
                                     's_nom': network.transformers.s_nom[i],
                                     'p': max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]])),
                                     'loading_old': round(max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]]))/network.transformers.s_nom[i]*100,2),
                                     's_nom_opt':network.transformers.s_nom_opt[i],
                                     'loading_new': round(loading,2),
                                     'dif': network.transformers.s_nom_opt[i]-network.transformers.s_nom[i]}) 
                    i+=1 
              
            print('Anzahl lines_time : ' + str(len(lines_time)))
            print('Anzahl Trafo_time : ' + str(len(trafo_time)))
            print('Anzahl All_time : ' + str(len(all_time)))
              
            return network,start_time,lines_time,trafo_time,all_time,maximum_line_loading,maximum_trafo_loading,file_name_method,line_time_list
                  
        ############################# benchmark ##############################          
        else:
            
            file_name_method = 'benchmark'          
            
            # set lines extendable
            network.lines.s_nom_extendable = True
            network.lines.s_nom_min = network.lines.s_nom
            network.lines.s_nom_max = np.inf
            
            #set trafo extendable
            network.transformers.s_nom_extendable = True
            network.transformers.s_nom_min = network.transformers.s_nom
            network.transformers.s_nom_max = np.inf
            
            ####################### Set capital cost ########################
             
            # Set capital cost for extendable lines
            cost_1 = 60000 # 110kV extendable
            cost_2 = 1600000/2 # 220kV extendable
            cost_3 = 200000 # 380kV extendable
            
            i=0
            while(i<len(network.lines)):
                name_bus_0 = network.lines.bus0[i]
                name_bus_1 = network.lines.bus1[i]
                    
                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]
                    
                if(U_bus_0 == U_bus_1):
                    if(U_bus_0 == 110):
                        network.lines.capital_cost[i] = \
                            (cost_1*network.lines.length[i]/network.lines.s_nom[i])/\
                            (90*8760)*(args['end_h']-args['start_h'])
                    elif(U_bus_0 == 220):
                        network.lines.capital_cost[i] = \
                            (cost_2*network.lines.length[i]/network.lines.s_nom[i])/\
                            (90*8760)*(args['end_h']-args['start_h'])
                    else:
                        network.lines.capital_cost[i] = \
                            (cost_3*network.lines.length[i]/network.lines.s_nom[i])/\
                            (90*8760)*(args['end_h']-args['start_h'])
                else:
                    print('Error')
                
                i+=1
                
             
            # Set capital cost for extendable trafo
            cost_1 = 5200000/300 # 220/110kV or 380/110kV extendable
            cost_2 = 8500000/600 # 380/220kV extendable
            cost_3 = 8500000/600 # other extendable
            
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
                        ((cost_1)/(40*8760)*(args['end_h']-args['start_h']))
                elif(U_OS == 380 and U_US == 220):
                    network.transformers.capital_cost[i] = \
                        ((cost_2)/(40*8760)*(args['end_h']-args['start_h']))
                else:
                    network.transformers.capital_cost[i] = \
                        ((cost_3)/(40*8760)*(args['end_h']-args['start_h']))
                    print('Other Transformator' + str(i))
                
                i+=1 
                
            ########################### calc LOPF ############################
            all_time =  [0,
 1,
 3,
 8,
 9,
 10,
 11,
 12,
 13,
 14,
 17,
 19,
 32,
 33,
 34,
 35,
 37,
 41,
 56,
 57,
 64,
 65,
 66,
 91,
 114,
 115,
 123,
 127,
 128,
 129,
 135,
 138,
 139,
 141,
 155,
 180,
 183,
 187,
 224,
 258,
 259,
 283,
 307,
 337,
 355,
 372,
 427,
 509,
 522,
 523,
 532,
 571,
 588,
 619,
 627,
 634,
 661,
 708,
 710,
 715,
 739]




            i=0
            while(i<len(all_time)):
                if i==0:
                    timeindex = network.snapshots[all_time[i]:all_time[i]+1]
                else:
                    timeindex = pd.DatetimeIndex.append(timeindex,other=network.snapshots[all_time[i]:all_time[i]+1])
                i+=1
               
            network.lopf(timeindex, solver_name=args['solver'])                    
                
                
#            network.lopf(scenario.timeindex, solver_name=args['solver'])
            objective=[[0],[network.objective]]
            
            ###################### Plotting the Results ######################
            filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.jpg'
            plot_max_line_loading(network,filename = filename)
            
            filename = args['line_ext_vers'] + '_02_Opt_line_maximum_loading_' + file_name_method +'.jpg'
            plot_max_opt_line_loading_bench(network,filename = filename)
            
            #################### Saving Datas in csv #########################
            filename = args['line_ext_vers'] + '_01_Lines_' + file_name_method +'.csv'
            with open(filename, 'w') as csvfile:
                fieldnames = ['line_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
    
                i = 0
                while (i<len(network.lines.s_nom)):

                    if network.lines_t.q0.empty:
                        s_current=max(abs(network.lines_t.p0[network.lines_t.p0.keys()[i]]))
                    else:
                        s_current = max(sqrt(network.lines_t.p0[network.lines_t.p0.keys()[i]]**2 + network.lines_t.q0[network.lines_t.q0.keys()[i]]**2))

                    s_nom = network.lines.s_nom_opt[i]
                    loading=(s_current/s_nom*100)
        
                    writer.writerow({'line_key': network.lines.s_nom.keys()[i],
                                     's_nom_extendable': network.lines.s_nom_extendable[i],
                                     's_nom': network.lines.s_nom[i],
                                     'p': max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]])),
                                     'loading_old': round(max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]]))/network.lines.s_nom[i]*100,2),
                                     's_nom_opt':network.lines.s_nom_opt[i],
                                     'loading_new': round(loading,2),
                                     'dif': network.lines.s_nom_opt[i]-network.lines.s_nom[i]}) 
                    i+=1
                    
                    
            filename = args['line_ext_vers'] + '_02_Trafo_' + file_name_method +'.csv'
            with open(filename, 'w') as csvfile:
                fieldnames = ['transformer_key','s_nom_extendable','s_nom','p','loading_old','s_nom_opt','loading_new','dif']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader() 
    
                i = 0
                while (i<len(network.transformers.s_nom)):
                    p = abs(network.transformers_t.p0[network.transformers_t.p0.keys()[i]])
                    if network.transformers_t.q0.empty:
                        s_current=max(abs(network.transformers_t.p0[network.transformers_t.p0.keys()[i]]))
                    else:
                        s_current = max(sqrt(network.transformers_t.p0[network.transformers_t.p0.keys()[i]]**2 + network.transformers_t.p0[network.transformers_t.q0.keys()[i]]**2))

                    s_nom = network.transformers.s_nom_opt[i]
                    loading=(s_current/s_nom*100)        
        
                    writer.writerow({'transformer_key': network.transformers.s_nom.keys()[i],
                                     's_nom_extendable': network.transformers.s_nom_extendable[i],
                                     's_nom': network.transformers.s_nom[i],
                                     'p': max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]])),
                                     'loading_old': round(max(abs(network.transformers_t.p0[network.transformers.s_nom.keys()[i]]))/network.transformers.s_nom[i]*100,2),
                                     's_nom_opt':network.transformers.s_nom_opt[i],
                                     'loading_new': round(loading,2),
                                     'dif': network.transformers.s_nom_opt[i]-network.transformers.s_nom[i]}) 
                    i+=1
            
            return network,start_time,objective,file_name_method           
    

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
        
if ((args['line_extendable'] == True) and (args['calc_type'] == True)):
    network,\
    start_time,\
    lines_time,\
    trafo_time,\
    all_time,\
    maximum_line_loading,\
    maximum_trafo_loading,\
    file_name_method,\
    line_time_list= \
    etrago(args)
    
    objective=[[0],[network.objective]]
    all_time_ = len(all_time)
    line_time_ = len(lines_time)
    trafo_time_ = len(trafo_time)
    
    end_time = time.time()
    z = (end_time - start_time) / 60 # z is time for lopf in minutes    
    
    filename = args['line_ext_vers'] + '_03_objective_' + file_name_method +'.csv'
    with open(filename, 'w') as csvfile:
        fieldnames = ['objective','time','Anzahl_line_time','Anzahl_trafo_time', 'Anzahl_all_time','line_time','trafo_time','all_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader() 
    
        i = 0
        while (i<len(objective[0])):          
            writer.writerow({'objective': objective[1][i],
                         'time': z,
                         'Anzahl_line_time' : line_time_ ,
                         'Anzahl_trafo_time' : trafo_time_ ,
                         'Anzahl_all_time' : all_time_,
                         'line_time' : lines_time ,
                         'trafo_time' : trafo_time ,
                         'all_time' : all_time})
            i+=1     
        
elif((args['line_extendable'] == True) and (args['calc_type'] == False)):
    network,\
    start_time,\
    objective,\
    file_name_method = \
    etrago(args)
    all_time_ = 0
    line_time_ = 0
    trafo_time_ = 0
    
    end_time = time.time()
    z = (end_time - start_time) / 60 # z is time for lopf in minutes
    
    filename = args['line_ext_vers'] + '_03_objective_' + file_name_method +'.csv'
    with open(filename, 'w') as csvfile:
        fieldnames = ['objective','time','Anzahl_line_time','Anzahl_trafo_time', 'Anzahl_all_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader() 
    
        i = 0
        while (i<len(objective[0])):          
            writer.writerow({'objective': objective[1][i],
                         'time': z,
                         'Anzahl_line_time' : line_time_ ,
                         'Anzahl_trafo_time' : trafo_time_ ,
                         'Anzahl_all_time' : all_time_})
            i+=1   
else:
    network = etrago(args)
    
print('finish')
end_time = time.time()
z = (end_time - start_time) / 60 # z is time for lopf in minutes


# plot stacked sum of nominal power for each generator type and timestep
filename = args['line_ext_vers'] + '_04_stacked_gen' + file_name_method +'.jpg'
plot_stacked_gen(network, resolution="MW",filename = filename)
             
# plot to show extendable transformers
filename = args['line_ext_vers'] + '_03_transformer_distribution_' + file_name_method +'.jpg'
transformers_distribution(network,filename=filename)    
    
filename = args['line_ext_vers'] + '_04_line_distribution_MW_' + file_name_method +'.jpg'
plot_dif_line_MW(network,filename=filename)

filename = args['line_ext_vers'] + '_05_line_distribution_Percent_' + file_name_method +'.jpg'
plot_dif_line_percent(network,filename=filename)    
    

    
    
print(z)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
