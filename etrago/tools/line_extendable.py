"""line extendables functions"""
import numpy as np
from numpy import genfromtxt
np.random.seed()
import progressbar
import time
from math import sqrt
import os
import pandas as pd
if not 'READTHEDOCS' in os.environ:
    from tools.io import NetworkScenario, results_to_oedb
    from tools.plot import (plot_line_loading, plot_stacked_gen,add_coordinates,
                            curtailment, gen_dist, storage_distribution,
                            plot_max_line_loading, plot_max_opt_line_loading,
                            plot_max_opt_line_loading_bench,transformers_distribution,
                            plot_dif_line_MW,plot_dif_line_percent)
    from tools.utilities import (oedb_session, load_shedding, data_manipulation_sh,
                                 results_to_csv, parallelisation, pf_post_lopf,
                                 loading_minimization, calc_line_losses,
                                 group_parallel_lines)
    from cluster.networkclustering import (busmap_from_psql, cluster_on_extra_high_voltage,
                                           kmean_clustering)
    from appl import etrago                                       
import csv
# toDo reduce import




def capacity_factor(network,cap_fac):
    """
    This function is for changing the capacities of lines and
    transformers.

    ########################### input parameters ###########################

    network: The whole network, which are to calculate
    cap_fac: cap_fac is a variable for the capacitiy factor.

    ########################### output parameters ##########################
    network : The function return the network with new capacities of lines
              and transformers

    """


    network.lines.s_nom = network.lines.s_nom * cap_fac
    network.transformers.s_nom = network.transformers.s_nom * cap_fac

    return network


def overload_lines(network):

    """
    This function is for finding the overload lines.
    First the loadings of all lines would calculate for each timestep.
    Seconde the maximum loadings of all lines are setting.
    After that it will check the value. If the value is over 100% the line
    will be consider. After finding all lines it will be found the timesteps
    of the maximum loading of the lines which are considered.
    Last the timesteps which are the same will be counted and sorted by the
    greatest number.

    ########################### input parameters ###########################

    network: The whole network, which are to calculate

    ########################### output parameters ##########################
    The function return the maximum loadings of each line (max_loading) and
    the timesteps which are count in (time_list).

    """

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
        # set current line key
        max_loading[0].append(network.lines_t.p0.keys()[i])

        # set maximum loading of the current line
        if(network.lines_t.q0.empty):
            s_current = abs(network.lines_t.p0[max_loading[0][i]])
        else:
            p = network.lines_t.p0[max_loading[0][i]]
            q = network.lines_t.q0[max_loading[0][i]]
            s_current = sqrt(p**2+q**2)

        max_loading[1].append(max(abs(s_current)/network.lines.s_nom[max_loading[0][i]]*100))

        # Find the timestep of maximum loading
        x= 0
        while(x<len(s_current)):
            if(s_current[x]==(max(abs(s_current)))):
                # set the timestep of maximum loading
                max_loading[2].append(x)
                break
            else:
                x+=1

        # filtering the lines which has a loading of over 100%
        loading = max_loading[1][i]
        time_index = max_loading[2][i]

        if (((time_index in timesteps[0]) == True) and (loading >= 100)):
            index = timesteps[0].index(time_index)
            timesteps[1][index] +=1
        elif(loading >= 100):
            timesteps[0].append(time_index)
            timesteps[1].append(1)

        i+=1

    # save the Result in a other way
    x=0
    while(x<len(timesteps[0])):
        time_list.append([timesteps[0][x],timesteps[1][x]])
        x+=1

    # For sorting the List by the item 1
    def getKey(item):
                return item[1]

    time_list = sorted(time_list,key=getKey)

    return max_loading,time_list

def overload_trafo(network):

    """
    This function is for finding the overload transformators.
    First the loadings of all transformators would calculate for each
    timestep.
    Seconde the maximum loadings of all lines are setting.
    After that it will check the value. If the value is over 100% the line
    will be consider. After finding all lines it will be found the timesteps
    of the maximum loading of the transformators which are considered.
    Last the timesteps which are the same will be counted and sorted by the
    greatest number.

    ########################### input parameters ###########################

    network: The whole network, which are to calculate

    ########################### output parameters ##########################
    The function return the maximum loadings of each transformators
    (max_loading) and the timesteps which are count in (time_list).

    """

    # List for counting the maximum value for timesteps
    # 0 : trafo keys
    # 1 : maximum loadings of each trafo
    # 2 : timestep of maximum loading
    max_loading = [[],[],[]]

    # List for counting the maximum loadings of each timestep
    # 0 : timestep index of trafo
    # 1 : Number of maximum loadings
    timesteps = [[],[]]

    # The same list like timesteps but for sorting by maximum number of loadings
    time_list =[]

    i=0
    while (i<len(network.transformers_t.p0.keys())):
        # set current trafo key
        max_loading[0].append(network.transformers_t.p0.keys()[i])

        # set maximum loading of the current trafo
        if(network.transformers_t.q0.empty):
            s_current = abs(network.transformers_t.p0[max_loading[0][i]])
        else:
            p = network.transformers_t.p0[max_loading[0][i]]
            q = network.transformers_t.q0[max_loading[0][i]]
            s_current = sqrt(p**2+q**2)

        max_loading[1].append(max(abs(s_current)/network.transformers.s_nom[max_loading[0][i]]*100))

        # Find the timestep of maximum loading
        x= 0
        while(x<len(s_current)):
            if(s_current[x]==(max(abs(s_current)))):
                # set the timestep of maximum loading
                max_loading[2].append(x)
                break
            else:
                x+=1

        # filtering the trafo which has a loading of over 100%
        loading = max_loading[1][i]
        time_index = max_loading[2][i]

        if (((time_index in timesteps[0]) == True) and (loading >= 100)):
            index = timesteps[0].index(time_index)
            timesteps[1][index] +=1
        elif(loading >= 100):
            timesteps[0].append(time_index)
            timesteps[1].append(1)

        i+=1

    # save the Result in a other way
    x=0
    while(x<len(timesteps[0])):
        time_list.append([timesteps[0][x],timesteps[1][x]])
        x+=1

    # For sorting the List by the item 1
    def getKey(item):
                return item[1]

    time_list = sorted(time_list,key=getKey)

    return max_loading,time_list


def set_line_cost(network,time_list,max_loading,cost_1,cost_2,cost_3):
    """
    This function set the capital cost of lines which ar extendable.
    This function set at the same time that the choosen lines are extendable
    for the calculation.

    ########################### input parameters ###########################

    network: The whole network, which are to calculate.
    time_list : List whith all timesteps which are considering for the
                calculation.
    max_loading: List with all maximum loadings of each line.
                 Index 0 : line keys
                 Index 1 : maximum loadings of each line
                 Index 2 : timestep of maximum loading

    cost_1 : The capital cost for extendable 110kV-lines
    cost_2 : The capital cost for extendable 220kV-lines
    cost_3 : The capital cost for extendable 380kV-lines

    ########################### output parameters ##########################
    The function return the network. In this variable the capital costs of
    the lines which are concidered are setted.
    It return also to extra lists.
    line_time: List with all timesteps which are considered.
    all_time: is the same list, but is for adding another datas.

    """

    #save the result in differnt variables
    lines_time=[]
    all_time = []

    i = 0
    while(i<len(time_list)):
#        if(i > 0.1*(args['end_h']-args['start_h'])):
#                break
        lines_time.append(time_list[len(time_list)-1-i][0])
        all_time.append(time_list[len(time_list)-1-i][0])

        index = [a for a,u in enumerate(max_loading[2]) if u==lines_time[i]]

        for k in index:
            if(max_loading[1][k]>100):
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
                            (cost_1*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                    elif(U_bus_0 == 220):
                        network.lines.capital_cost[max_loading[0][k]] = \
                            (cost_2*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                    else:
                        network.lines.capital_cost[max_loading[0][k]] = \
                            (cost_3*network.lines.length[max_loading[0][k]]/network.lines.s_nom[max_loading[0][k]])/\
                            (90*8760)
                else:
                    print('Error')
        i+=1

    return network,lines_time,all_time


def set_trafo_cost(network,time_list,max_loading,cost_1,cost_2,cost_3):

    """
    This function set the capital cost of transformators which ar extendable.
    This function set at the same time that the choosen transformators are
    extendable for the calculation.

    ########################### input parameters ###########################

    network: The whole network, which are to calculate.
    time_list : List whith all timesteps which are considering for the
                calculation.
    max_loading: List with all maximum loadings of each transformators.
                 Index 0 : transformators keys
                 Index 1 : maximum loadings of each transformators
                 Index 2 : timestep of maximum loading

    cost_1 : The capital cost for extendable 110kV-220kV and 110kV-380kV
             Transformators.
    cost_2 : The capital cost for extendable 220kV-380kV Transformers
    cost_3 : The capital cost for extendable Transformers (rest)

    ########################### output parameters ##########################
    The function return the network. In this variable the capital costs of
    the transformators which are concidered are setted.
    It return also to extra lists.
    trafo_time: List with all timesteps which are considered.

    """


    # List of choosen timesteps
    trafo_time=[]

    i = 0
    while(i<len(time_list)):
#        if(i > 0.1*(args['end_h']-args['start_h'])):
#            break
        trafo_time.append(time_list[len(time_list)-1-i][0])

        index = [a for a,u in enumerate(max_loading[2]) if u==trafo_time[i]]
        for k in index:
            if(max_loading[1][k]>100):
                network.transformers.s_nom_extendable[max_loading[0][k]] = True
                network.transformers.s_nom_min[max_loading[0][k]] = network.transformers.s_nom[max_loading[0][k]]
                network.transformers.s_nom_max[max_loading[0][k]] = np.inf

                name_bus_0 = network.transformers.bus0[max_loading[0][k]]
                name_bus_1 = network.transformers.bus1[max_loading[0][k]]

                U_bus_0 = network.buses.v_nom[name_bus_0]
                U_bus_1 = network.buses.v_nom[name_bus_1]

                U_OS = max(U_bus_0,U_bus_1)
                U_US = min(U_bus_0,U_bus_1)

                if((U_OS == 220 and U_US == 110) or (U_OS == 380 and U_US == 110)):
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_1)/(40*8760))
                elif(U_OS == 380 and U_US == 220):
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_2)/(40*8760))
                else:
                    network.transformers.capital_cost[max_loading[0][k]] = \
                        ((cost_3)/(40*8760))
                    print('Other Transformator' + str(k))

        i+=1

    return network,trafo_time


def line_extendable_short(network, args, scenario):
    """
    Function witch prepare and run a
    line_extendable calculation.

    """  
   
   
   # set the capacity-factory for the first lopf
    cap_fac = 1.3
         
    # Change the capcity of lines and transformers
    network = capacity_factor(network,cap_fac)
                                        
    ############################ 1. Lopf ###########################
    parallelisation(network, start_snapshot=args['start_snapshot'], \
        end_snapshot=args['end_snapshot'],group_size=1, solver_name=args['solver'])
         
        
    # return to original capacities
    network = capacity_factor(network,(1/cap_fac))
        
        
    # plotting the loadings of lines at start
    plot_max_line_loading(network,filename = 'Start_maximum_line_loading.png')
         
         
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
                                  filename='maximum_optimal_lines.png')
         
    return network




def line_extendable(network, args, scenario, start_time):
    """
    Function witch prepare and run a
    line_extendable calculation.




    """
  
    
    ############################## methode ##############################
    # If calc_type == True -> Methodik
    if args['calc_type']:
        file_name_method = 'method'

        # set the capacity-factory for the first lopf
        cap_fac =1.3 # 1.3

        # Change the capcity of lines and transformers
        network = capacity_factor(network,cap_fac)

        ############################ 1. Lopf ###########################
        #parallelisation(network, start_snapshot=args['start_snapshot'], \
        #    end_snapshot=args['end_snapshot'],group_size=1, solver_name=args['solver']
        #    , extra_functionality=extra_functionality)

        # return to original capacities
        network = capacity_factor(network,(1/cap_fac))


        # plotting the loadings of lines at start
        filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.png'
        #plot_max_line_loading(network,filename = filename)


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
        filename = args['line_ext_vers'] + '_02_Opt_line_maximum_loading_' + file_name_method +'.png'
        #plot_max_opt_line_loading(network,lines_time,filename=filename)   # bug in plot function max(p)

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
                loading=(max(s_current)/s_nom*100)    # bug ValueError: max() arg is an empty sequence

                writer.writerow({'line_key': network.lines.s_nom.keys()[i],
                                 's_nom_extendable': network.lines.s_nom_extendable[i],
                                 's_nom': network.lines.s_nom[i],
                                 'p': max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]])),
                                 'loading_old': round(max(abs(network.lines_t.p0[network.lines.s_nom.keys()[i]]))/network.lines.s_nom[i]*100,2),
                                 's_nom_opt':network.lines.s_nom_opt[i],
                                 'loading_new': round(loading,2),               # due to bug
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
                        (90*8760)*(args['end_snapshot']-args['start_snapshot'])
                elif(U_bus_0 == 220):
                    network.lines.capital_cost[i] = \
                        (cost_2*network.lines.length[i]/network.lines.s_nom[i])/\
                        (90*8760)*(args['end_snapshot']-args['start_snapshot'])
                else:
                    network.lines.capital_cost[i] = \
                        (cost_3*network.lines.length[i]/network.lines.s_nom[i])/\
                        (90*8760)*(args['end_snapshot']-args['start_snapshot'])
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
                    ((cost_1)/(40*8760)*(args['end_snapshot']-args['start_snapshot']))
            elif(U_OS == 380 and U_US == 220):
                network.transformers.capital_cost[i] = \
                    ((cost_2)/(40*8760)*(args['end_snapshot']-args['start_snapshot']))
            else:
                network.transformers.capital_cost[i] = \
                    ((cost_3)/(40*8760)*(args['end_snapshot']-args['start_snapshot']))
                print('Other Transformator' + str(i))

            i+=1

        ########################### calc LOPF ############################
        all_time =  [0, 1, 3, 8, 9, 10, 11, 12, 13, 14, 17, 19, 32, 33, 34,
        35, 37, 41, 56, 57, 64, 65, 66, 91, 114, 115, 123, 127, 128, 129, 135,
        138, 139, 141, 155, 180, 183, 187, 224, 258, 259, 283, 307, 337, 355,
        372, 427, 509, 522, 523, 532, 571, 588, 619, 627, 634, 661, 708, 710,
        715, 739]

        i=0
        while(i<len(all_time)):
            if i==0:
                timeindex = network.snapshots[all_time[i]:all_time[i]+1]
            else:
                timeindex = pd.DatetimeIndex.append(timeindex,other=network.snapshots[all_time[i]:all_time[i]+1])
            i+=1

      # change position of lopf
      
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'], extra_functionality=extra_functionality)
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes

        #network.lopf(timeindex, solver_name=args['solver'])


        #   network.lopf(scenario.timeindex, solver_name=args['solver'])
        objective=[[0],[network.objective]]
        print(str(network.objective))
        
        
        
        
        

        ###################### Plotting the Results ######################
        filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.png'
        #plot_max_line_loading(network,filename = filename)

        filename = args['line_ext_vers'] + '_02_Opt_line_maximum_loading_' + file_name_method +'.png'
        plot_max_opt_line_loading_bench(network,filename = filename)            #bug in function

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

        return network, start_time


def line_extendable_ma(network, args,start_time):
    """
    Calculation based on MA and own method
    """

    # calculation


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
    filename = args['line_ext_vers'] + '_04_stacked_gen' + file_name_method +'.png'
    plot_stacked_gen(network, resolution="MW",filename = filename)

    # plot to show extendable transformers
    filename = args['line_ext_vers'] + '_03_transformer_distribution_' + file_name_method +'.png'
    transformers_distribution(network,filename=filename)

    filename = args['line_ext_vers'] + '_04_line_distribution_MW_' + file_name_method +'.png'
    plot_dif_line_MW(network,filename=filename)

    filename = args['line_ext_vers'] + '_05_line_distribution_Percent_' + file_name_method +'.png'
    plot_dif_line_percent(network,filename=filename)


    print(z)
