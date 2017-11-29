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

"""

__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"

import numpy as np
from numpy import genfromtxt
np.random.seed()
import progressbar
import time
from etrago.tools.io import NetworkScenario, results_to_oedb
from etrago.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution)
from etrago.tools.utilities import oedb_session, load_shedding, data_manipulation_sh, results_to_csv, parallelisation, pf_post_lopf, loading_minimization, calc_line_losses, group_parallel_lines
from etrago.cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering


from etrago.tools.plot import plot_max_line_loading,plot_max_opt_line_loading,plot_max_opt_line_loading_bench,transformers_distribution,plot_dif_line_MW,plot_dif_line_percent
from etrago.tools.line_extendable_functions import capacity_factor,overload_lines,overload_trafo,set_line_cost,set_trafo_cost
import pandas as pd
import csv
from math import sqrt


################################################################################

## Angepasst von file line_optimization(Masterthesis Dokumentation).py

################################################################################



args = {# Setup and Configuration:
        'db': 'oedb', # db session
        'gridversion': 'v0.2.10', # None for model_draft or Version number (e.g. v0.2.11) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': False, # state whether you want to perform a pf after a lopf simulation
        'start_snapshot': 1,
        'end_snapshot' : 24,
        'scn_name': 'SH NEP 2035', # state which scenario you want to run: Status Quo, NEP 2035, eGo100
        'solver': 'gurobi', # glpk, cplex or gurobi
        # Export options:
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or /path/tofolder
        'results': False, # state if and where you want to save results as csv: False or /path/tofolder
        'export': False, # state if you want to export the results back to the database
        # Settings:
        'storage_extendable':True, # state if you want storages to be installed at each node if necessary.
        'generator_noise':True, # state if you want to apply a small generator noise
        'reproduce_noise': False, # state if you want to use a predefined set of random noise for the given scenario. if so, provide path, e.g. 'noise_values.csv'
        'minimize_loading':False,
        #
        'line_extendable':True,
        'calc_type' : False,      # True for methodik of line_extendable  #False for all lines are extendables
        'line_ext_vers' : '5_DE_NEP2035_24h_1.3',
        # Clustering:
        'k_mean_clustering': False, # state if you want to perform a k-means clustering on the given network. State False or the value k (e.g. 20).
        'network_clustering': False, # state if you want to perform a clustering of HV buses to EHV buses.
        # Simplifications:
        'parallelisation':False, # state if you want to run snapshots parallely.
        'line_grouping': True, # state if you want to group lines running between the same buses.
        'branch_capacity_factor':1, # globally extend or lower branch capacities
        'load_shedding':True, # meet the demand at very high cost; for debugging purposes.
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
        State if you want to apply a clustering of all network buses down to
        only 'k' buses. The weighting takes place considering generation and load
        at each node.
        If so, state the number of k you want to apply. Otherwise put False.
	    This function doesn't work together with 'line_grouping = True'
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
    # Set start time
    start_time = time.time()


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

    if args['line_extendable']:

        ############################## methode ##############################
        # If calc_type == True -> Methodik
        if args['calc_type']:
            file_name_method = 'method'

            # set the capacity-factory for the first lopf
            cap_fac =1.3

            # Change the capcity of lines and transformers
            network = capacity_factor(network,cap_fac)

            ############################ 1. Lopf ###########################
            parallelisation(network, start_snapshot=args['start_snapshot'], \
                end_snapshot=args['end_snapshot'],group_size=1, solver_name=args['solver'])


            # return to original capacities
            network = capacity_factor(network,(1/cap_fac))


            # plotting the loadings of lines at start
            filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.png'
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

            network.lopf(timeindex, solver_name=args['solver'])


#            network.lopf(scenario.timeindex, solver_name=args['solver'])
            objective=[[0],[network.objective]]

            ###################### Plotting the Results ######################
            filename = args['line_ext_vers'] + '_01_Start_line_maximum_loading_' + file_name_method +'.png'
            plot_max_line_loading(network,filename = filename)

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

            return network,start_time,objective,file_name_method



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

    # write lpfile to path
    if not args['lpfile'] == False:
        network.model.write(args['lpfile'], io_options={'symbolic_solver_labels':
                                                     True})
    # write PyPSA results back to database
    if args['export']:
        results_to_oedb(session, network, args, 'hv')

    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, args['results'])

    return network


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
