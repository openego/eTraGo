# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
sclopf.py defines functions for contingency analysis. 
"""
import numpy as np
import pandas as pd
import time
import datetime
import logging
logger = logging.getLogger(__name__)
from etrago.tools.utilities import results_to_csv, update_electrical_parameters
from pypsa.opt  import l_constraint
from pypsa.opf import define_passive_branch_flows_with_kirchhoff, network_lopf_solve, define_passive_branch_flows
import multiprocessing as mp
import csv

def iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre):
    """
    Function that runs iterations of lopf without building new models. 
    Currently only working with model_formulation = 'kirchhoff'
    
    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    l_snom_pre: pandas.Series
        s_nom of ac-lines in previous iteration
    t_snom_pre: pandas.Series
        s_nom of transformers in previous iteration
    """
    # Delete flow constraints for each possible model formulation
    x = time.time()
    network.model.del_component('cycle_constraints')
    network.model.del_component('cycle_constraints_index')
    network.model.del_component('cycle_constraints_index_0')
    network.model.del_component('cycle_constraints_index_1')

    
    if args['model_formulation']=='kirchhoff':
        define_passive_branch_flows_with_kirchhoff(network,network.snapshots,skip_vars=True)
    else:
        
        logger.error('Currently only implemented for kirchhoff-formulation.')
    y = time.time()
    logger.info("Flow constraints updated in [min] " + str((y-x)/60))
    network = network_lopf_solve(network,
                                 network.snapshots,
                                 formulation=args['model_formulation'],
                                 solver_options = args['solver_options'])

    return network


def post_contingency_analysis_lopf(network, args,
                                       branch_outages, 
                                       n_process = 4):
    x = time.time()
    nw = network.copy()
    nw.lines.s_nom = (nw.lines.s_nom_opt/0.7).copy()
    nw.generators_t.p_set = nw.generators_t.p_set.reindex(columns=nw.generators.index)
    nw.generators_t.p_set = nw.generators_t.p
    nw.storage_units_t.p_set = nw.storage_units_t.p_set.reindex(columns=nw.storage_units.index)
    nw.storage_units_t.p_set = nw.storage_units_t.p
    nw.links_t.p_set = nw.links_t.p_set.reindex(columns=nw.links.index)
    nw.links_t.p_set = nw.links_t.p0
    
    snapshots_set={}
    length = int(nw.snapshots.size / n_process)
    for i in range(n_process):
        snapshots_set[str(i+1)]=nw.snapshots[i*length : (i+1)*length]
    manager = mp.Manager()
    d = manager.dict()
    snapshots_set[str(n_process)] = nw.snapshots[i*length :]


    def multi_con(nw, snapshots, d):

        for sn in snapshots:  
        #Check no lines are overloaded with the linear contingency analysis
            p0_test = nw.lpf_contingency(branch_outages=branch_outages, snapshots=sn) # rows: branch outage, index = monitorred line
        
        #check loading as per unit of s_nom in each contingency
            load = abs(p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)).drop(['base'], axis = 1) # columns: branch_outages
            # schon im base_case leitungen teilweise überlastet. Liegt das an x Anpassung? 
            overload = (load -1)# columns: branch_outages
            overload[overload<0] = 0

            if not len(overload) == 0:
                d[sn]=overload

    processes = [mp.Process(target=multi_con, args=(nw, snapshots_set[i], d)) for i in snapshots_set]

    
    # Run processes
    for p in processes:
        p.start()
        

# Exit the completed processes
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    if args['csv_export'] != False:
        import os
        if not os.path.exists(args['csv_export'] + '/post_contingency_analysis/'):
            os.mkdir(args['csv_export'] + '/post_contingency_analysis/')
            
        for s in d.keys():
            d[s].to_csv(args['csv_export'] + '/post_contingency_analysis/' +str(s)+'.csv')
            
    y = (time.time() - x)/60
    print("Post contingengy check finished in "+ str(round(y, 2))+ " minutes.")
   # import pdb; pdb.set_trace()
    return d


def post_contingency_analysis(network, branch_outages, delta = 0.05):
    nw = network.copy()
    nw.lines.s_nom = nw.lines.s_nom_opt.copy()
    
    nw.lines = nw.lines.drop(nw.lines.index[nw.lines.s_nom<10])
    #import pdb; pdb.set_trace()
    b_x = 1./nw.lines.x_pu
    logger.info(b_x.min())

    if np.isnan(b_x).any():
        import pdb; pdb.set_trace()
    branch_outages = branch_outages[branch_outages.isin(nw.lines.index)]
    nw.generators_t.p_set = nw.generators_t.p_set.reindex(
            columns=nw.generators.index)
    nw.generators_t.p_set = nw.generators_t.p
    nw.storage_units_t.p_set = nw.storage_units_t.p_set.reindex(
            columns=nw.storage_units.index)
    nw.storage_units_t.p_set = nw.storage_units_t.p
    nw.links_t.p_set = nw.links_t.p_set.reindex(columns=nw.links.index)
    nw.links_t.p_set = nw.links_t.p0

    d = {}
    for sn in network.snapshots:

            p0_test = nw.lpf_contingency(branch_outages=branch_outages,
                                         snapshots=sn)
            # rows: branch outage, index = monitorred line
            #check loading as per unit of s_nom in each contingency
            load_signed = (p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)
                    ).drop(['base'], axis = 1)
            load = abs(p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)
                    ).drop(['base'], axis = 1) # columns: branch_outages
            if True:# noch experimentell
                load_per_outage_over = load_signed.transpose()[load_signed.abs().max() > (1+delta)].transpose()
                
                out = load_per_outage_over.columns.values
                mon = load_per_outage_over.abs().idxmax().values
                
                sign = []
                for i in range(len(out)): 
                    sign.append(np.sign(load_per_outage_over[out[i]][mon[i]]).astype(int))
                combinations = [out, mon, sign]

            else:
                overloaded = (load>(1 + delta))# columns: branch_outages

                array_mon = []
                array_out = []
                array_sign = []
    
                for col in overloaded:
                    mon = overloaded.index[overloaded[col]].tolist()
                    out = [col]*len(mon)
    
                    sign = np.sign(load_signed[overloaded[col]][col]).values
                    
                    if mon != []:
                        array_mon.extend(mon)
                        array_out.extend(out)
                        array_sign.extend(sign)
                combinations = [array_out, array_mon, array_sign]
            if not len(combinations[0]) == 0:
                d[sn]=combinations

    return d

def post_contingency_analysis_per_line(network, 
                                       branch_outages, 
                                       n_process = 4, 
                                       delta = 0.01):
    x = time.time()
    nw = network.copy()
    nw.lines.s_nom = nw.lines.s_nom_opt.copy()
    
    nw.lines = nw.lines.drop(nw.lines.index[nw.lines.s_nom<10])

    branch_outages = branch_outages[branch_outages.isin(nw.lines.index)]
    nw.generators_t.p_set = nw.generators_t.p_set.reindex(
            columns=nw.generators.index)
    nw.generators_t.p_set = nw.generators_t.p
    nw.storage_units_t.p_set = nw.storage_units_t.p_set.reindex(
            columns=nw.storage_units.index)
    nw.storage_units_t.p_set = nw.storage_units_t.p
    nw.links_t.p_set = nw.links_t.p_set.reindex(columns=nw.links.index)
    nw.links_t.p_set = nw.links_t.p0
    snapshots_set={}
    length = int(nw.snapshots.size / n_process)

    for i in range(n_process):
        snapshots_set[str(i+1)]=nw.snapshots[i*length : (i+1)*length]
    snapshots_set[str(n_process)] = nw.snapshots[i*length :]

    manager = mp.Manager()
    d = manager.dict()

    def multi_con(nw, snapshots, d):

        for sn in snapshots:  
            # Check no lines are overloaded with the linear contingency analysis
            p0_test = nw.lpf_contingency(branch_outages=branch_outages,
                                         snapshots=sn)
            # rows: branch outage, index = monitorred line
            #check loading as per unit of s_nom in each contingency
            load_signed = (p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)
                    ).drop(['base'], axis = 1)
            load = abs(p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)
                    ).drop(['base'], axis = 1) # columns: branch_outages
            if True:# noch experimentell
                load_per_outage_over = load_signed.transpose()[load_signed.abs().max() > (1+delta)].transpose()
                
                out = load_per_outage_over.columns.values
                mon = load_per_outage_over.abs().idxmax().values
                
                sign = []
                for i in range(len(out)): 
                    sign.append(np.sign(load_per_outage_over[out[i]][mon[i]]).astype(int))
                combinations = [out, mon, sign]

            else:
                overloaded = (load>(1 + delta))# columns: branch_outages

                array_mon = []
                array_out = []
                array_sign = []
    
                for col in overloaded:
                    mon = overloaded.index[overloaded[col]].tolist()
                    out = [col]*len(mon)
    
                    sign = np.sign(load_signed[overloaded[col]][col]).values
                    
                    if mon != []:
                        array_mon.extend(mon)
                        array_out.extend(out)
                        array_sign.extend(sign)
                combinations = [array_out, array_mon, array_sign]
            if not len(combinations[0]) == 0:
                d[sn]=combinations

    processes = [mp.Process(
            target=multi_con, args=(nw, snapshots_set[i], d)
            ) for i in snapshots_set]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
   # import pdb; pdb.set_trace()
    y = (time.time() - x)/60
    logger.info("Post contingengy check finished in "
                + str(round(y, 2))+ " minutes.")

    return d


def add_all_contingency_constraints(network,combinations, track_time):
    
    x = time.time()
    
    branch_outage_keys = []
    contingency_flow = {}
    n_buses = 0

    # choose biggest sub_network
    if len(network.sub_networks.obj.index)> 1:
        for s in network.sub_networks.obj.index:
            n = len(network.sub_networks.obj[s].buses())
        
            if n > n_buses:
                n_buses = n
                sub = network.sub_networks.obj[s]
    else: 
        sub = network.sub_networks.obj[0]
        
    sub._branches = sub.branches()
    sub.calculate_BODF()
    sub._branches["_i"] = range(sub._branches.shape[0])
    sub._extendable_branches =  sub._branches[ sub._branches.s_nom_extendable]
    sub._fixed_branches = sub._branches[~  sub._branches.s_nom_extendable]

    
    # set dict with s_nom containing fixed or extendable s_nom
    s_nom = sub._branches.s_nom.to_dict()

    for idx in sub._extendable_branches.index.values:
         s_nom[idx] = network.model.passive_branch_s_nom[idx]

    for sn in combinations.keys(): # Könnte parallelisiert werden, klappt aber noch nicht 
        if len(combinations[sn][0]) > 0:
           out= combinations[sn][0]# branch in pypsa
           mon = combinations[sn][1] # b in pypsa
           sign = combinations[sn][2]

           branch_outage_keys.extend([(out[i][0],out[i][1], 
                                       mon[i][0],mon[i][1], sn)
                                    for i in range(len(out))])
                    
            # avoid duplicate values in branch_outage_keys
           branch_outage_keys=list(set(branch_outage_keys))
           # set contingengy flow constraint
           if sub._fixed_branches.empty:
               contingency_flow.update({(
                       out[i][0], out[i][1], mon[i][0], mon[i][1], sn) : [[
                            (sign[i], network.model.passive_branch_p[mon[i], sn]),
                            (sign[i]*sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                             network.model.passive_branch_p[out[i], sn]), 
                             (-1,s_nom[mon[i]])],"<=",0] 
                            for i in range(len(mon))})

           elif sub._extendable_branches.empty:
               contingency_flow.update({(
                       out[i][0], out[i][1], mon[i][0], mon[i][1], sn) : [[
                            (sign[i], network.model.passive_branch_p[mon[i], sn]),
                            (sign[i]*sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                             network.model.passive_branch_p[out[i], sn])]
                            ,"<=",s_nom[mon[i]]] 
                            for i in range(len(mon))})
    
           else:
               print('Not implemented!')

    z = time.time()
    track_time[datetime.datetime.now()]=  'Contingency constraints calculated'
    logger.info("Security constraints calculated in [min] " + str((z-x)/60))

    # remove constraints from previous iteration
    network.model.del_component('contingency_flow')
    network.model.del_component('contingency_flow_index')

    # Delete rows with small BODFs to avoid nummerical problems
    for c in list(contingency_flow.keys()):
        if ((abs(contingency_flow[c][0][1][0]) < 5e-8) & \
                (abs(contingency_flow[c][0][1][0]) != 0)) | (abs(contingency_flow[c][0][1][0]) >1.1):
             contingency_flow.pop(c)
#        elif (abs(contingency_flow[c][0][1][0]) >1.1) :
#             contingency_flow.pop(c)
    
    # set constraints for new iteration
    l_constraint(network.model,"contingency_flow",
                 contingency_flow,
                 #branch_outage_keys)
                 contingency_flow.keys())

    y = time.time()
    logger.info("Security constraints updated in [min] " + str((y-x)/60))
    
    return len(contingency_flow)

def calc_new_sc_combinations(combinations, new):
    for sn in new.keys():  
        com = [[],[],[]]
        com[0] = combinations[sn][0] + list(new[sn][0])
        com[1] = combinations[sn][1] + list(new[sn][1])
        com[2] = combinations[sn][2] + new[sn][2]

        df = pd.DataFrame([com[0], com[1], com[2]])
        data = df.transpose().drop_duplicates().transpose()    
        combinations[sn] = data.values.tolist()
    return combinations

def iterate_sclopf(network, 
                       args, 
                       branch_outages, 
                       extra_functionality, 
                       n_process,
                       delta,
                       n_overload = 0,
                       post_lopf = False):
    track_time = pd.Series()
    l_snom_pre = network.lines.s_nom.copy()
    t_snom_pre = network.transformers.s_nom.copy()
    add_all_contingency_constraints.counter = 0
    n=0
    track_time[datetime.datetime.now()]= 'Iterative SCLOPF started'
    x = time.time()
    results_to_csv.counter=0

    # 1. LOPF without SC
    solver_options_lopf=args['solver_options']
    solver_options_lopf['FeasibilityTol'] = 1e-5
    solver_options_lopf['BarConvTol'] = 1e-6

    if post_lopf:

        network.lines.s_nom_opt = network.lines.s_nom_opt *\
                                network.lines.s_nom_total/network.lines.s_nom
        network.lines.s_nom = network.lines.s_nom_opt.copy()
        network.links.p_nom = network.links.p_nom_opt.copy()
        network.lines.s_nom_extendable = False
        network.links.p_nom_extendable = False
        network.storage_units.p_nom = network.storage_units.p_nom_opt.copy()
        network.storage_units.p_nom_extendable = False
        path_name = '/post_sclopf_iteration_'

    else:
        network.lopf(   network.snapshots,
                        solver_name=args['solver'],
                        solver_options=solver_options_lopf,
                        extra_functionality=extra_functionality,
                        formulation=args['model_formulation'])
        path_name = '/sclopf_iteration_'
        track_time[datetime.datetime.now()]= 'Solve SCLOPF'
        if args['csv_export'] != False:
            path=args['csv_export'] + '/sclopf_iteration_0'
            results_to_csv(network, args, path, with_time = False)
            track_time[datetime.datetime.now()]= 'Export results'
            
            if network.results["Solver"][0]["Status"].key!='ok':
                raise  Exception('SCLOPF not solved.')
            
                if args['csv_export'] != False:
                    network.model.write(args['csv_export']  + '/last_lp.lp', io_options={
                            'symbolic_solver_labels': True})

    # Update electrical parameters if network is extendable
    if network.lines.s_nom_extendable.any():
        l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network, 
                                                 l_snom_pre, t_snom_pre)
        track_time[datetime.datetime.now()]= 'Adjust impedances'

    # Calc SC
    nb = 0
    new = post_contingency_analysis_per_line(
                network, 
                branch_outages, 
                n_process,
                delta)

    track_time[datetime.datetime.now()]= 'Overall post contingency analysis'

    # Initalzie dict of SC
    combinations =  dict.fromkeys(network.snapshots, [[], [], []])
    size = 0
    for i in range(len(new.keys())):
        size = size + len(new.values()[i][0]) * network.snapshot_weightings[new.keys()[i]]

    while size > n_overload:
        if  n < 50:
            logger.info(str(size) + ' overloadings')
            
            combinations = calc_new_sc_combinations(combinations, new)

            nb = int(add_all_contingency_constraints(
                        network, combinations,track_time))/2

            track_time[datetime.datetime.now()]= 'Update Contingency constraints'

            logger.info("SCLOPF No. "+ str(n+1) + " started with " 
                            + str(2*nb) + " SC-constraints.")

            iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre)
            track_time[datetime.datetime.now()]= 'Solve SCLOPF'

            if network.results["Solver"][0]["Status"].key!='ok':
                raise  Exception('SCLOPF '+ str(n) + ' not solved.')
            
                if args['csv_export'] != False:
                    network.model.write(args['csv_export']  + '/last_lp.lp', io_options={
                            'symbolic_solver_labels': True})

                path = args['csv_export'] + path_name + str(n+1)
                results_to_csv(network, args, path, with_time = False)
                    
                with open(path + '/sc_combinations.csv', 'w') as f:
                    for key in combinations.keys():
                        f.write("%s,%s\n"%(key,combinations[key]))
                track_time[datetime.datetime.now()]= 'Export results'

            # nur mit dieser Reihenfolge (x anpassen, dann lpf_check) kann Netzausbau n-1 sicher werden
            if network.lines.s_nom_extendable.any():
                l_snom_pre, t_snom_pre = \
                        update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)
                track_time[datetime.datetime.now()]= 'Adjust impedances'

            new = post_contingency_analysis_per_line(
                        network, 
                        branch_outages, 
                        n_process,
                        delta)

            size = 0
            for i in range(len(new.keys())):
                size = size + len(new.values()[i][0])* network.snapshot_weightings[new.keys()[i]]

            track_time[datetime.datetime.now()]= 'Overall post contingency analysis'
            n+=1

        else: 
            print('Maximum number of iterations reached.')
            break

    if args['csv_export'] != False:
                path = args['csv_export'] + path_name + str(n+1)
                results_to_csv(network, args, path, with_time = True)
                track_time.to_csv(args['csv_export']+ '/track-time.csv')

    y = (time.time() - x)/60
    
    logger.info("SCLOPF with " + str(2*nb) +
                    " constraints solved in " + str(n) + 
                    " iterations in "+ str(round(y, 2))+ " minutes.")

def iterate_full_sclopf(network, args, branch_outages, extra_functionality, 
                   method={'n_iter':4}, delta_s_max=0):

    """
    Run optimization of lopf. If network extension is included, the specified 
    number of iterations is calculated to consider reactance changes. 

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    args: dict
        Settings in appl.py
    extra_functionality: str
        Define extra constranits. 
    method: dict
        Choose 'n_iter' and integer for fixed number of iterations or
        'threshold' and derivation of objective in percent for variable number
        of iteration until the threshold of the objective function is reached
    delta_s_max: float
        Increase of maximal extension of each line in p.u.
        Currently only working with method n_iter

    """
    results_to_csv.counter=0
    
    # if network is extendable, iterate lopf 
    # to include changes of electrical parameters
    if network.lines.s_nom_extendable.any():
        
        max_ext_line=network.lines.s_nom_max/network.lines.s_nom
        max_ext_link=network.links.p_nom_max/network.links.p_nom
        max_ext_trafo=network.transformers.s_nom_max/\
            network.transformers.s_nom

        # Initialise s_nom_pre (s_nom_opt of previous iteration) 
        # to s_nom for first lopf:
        l_snom_pre = network.lines.s_nom.copy()
        t_snom_pre = network.transformers.s_nom.copy()

        # calculate fixed number of iterations
        if 'n_iter' in method:
            n_iter = method['n_iter']

            for i in range (1,(1+n_iter)):
                x = time.time()
                network.lines.s_nom_max=\
                 (max_ext_line-(n_iter-i)*delta_s_max)*network.lines.s_nom
                network.transformers.s_nom_max=\
                 (max_ext_trafo-(n_iter-i)*delta_s_max)*\
                network.transformers.s_nom                
                network.links.p_nom_max=\
                 (max_ext_link-(n_iter-i)*delta_s_max)*network.links.p_nom
            
                network.sclopf(
                    network.snapshots,
                    branch_outages=branch_outages,
                    solver_name=args['solver'],
                    solver_options=args['solver_options'],
                    extra_functionality=extra_functionality,
                    formulation=args['model_formulation'])
                y = time.time()
                z = (y - x) / 60

                if network.results["Solver"][0]["Status"].key!='ok':
                    raise  Exception('SCLOPF '+ str(i) + ' not solved.')

                print("Time for SCLOPF [min]:", round(z, 2))
                if args['csv_export'] != False:
                    path=args['csv_export'] + '/sclopf_iteration_'+ str(i)
                    results_to_csv(network, args, path)

                if i < n_iter:
                    l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network, 
                                                 l_snom_pre, t_snom_pre)
        
        # Calculate variable number of iterations until threshold of objective 
        # function is reached

        if 'threshold' in method:
            thr = method['threshold']
            x = time.time()
            network.sclopf(
                    network.snapshots,
                    branch_outages=branch_outages,
                    solver_name=args['solver'],
                    solver_options=args['solver_options'],
                    extra_functionality=extra_functionality,
                    formulation=args['model_formulation'])
            y = time.time()
            z = (y - x) / 60
            
            print("Time for SCLOPF [min]:", round(z, 2))

            diff_obj=network.objective*thr/100

            i = 1
            
            # Stop after 100 iterations to aviod unending loop
            while i <= 100:
                
                if i ==100:
                    print('Maximum number of iterations reached.')
                    break
                
                l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network, 
                                                 l_snom_pre, t_snom_pre)
                pre = network.objective
                
                x = time.time()
                network.sclopf(
                    network.snapshots,
                    branch_outages=branch_outages,
                    solver_name=args['solver'],
                    solver_options=args['solver_options'],
                    extra_functionality=extra_functionality,
                    formulation=args['model_formulation'])
                y = time.time()
                z = (y - x) / 60
            
                print("Time for SCLOPF [min]:", round(z, 2))
                
                if network.results["Solver"][0]["Status"].key!='ok':
                    raise  Exception('SCLOPF '+ str(i) + ' not solved.')

                i += 1

                if args['csv_export'] != False:
                    path=args['csv_export'] + '/sclopf_iteration_'+ str(i)
                    results_to_csv(network, args, path)
                    
                if abs(pre-network.objective) <=diff_obj:
                    print('Threshold reached after ' + str(i) + ' iterations.')
                    break
                    
    else:
            x = time.time()
            network.sclopf(
                    network.snapshots,
                    branch_outages=branch_outages,
                    solver_name=args['solver'],
                    solver_options=args['solver_options'],
                    extra_functionality=extra_functionality,
                    formulation=args['model_formulation'])
            y = time.time()
            z = (y - x) / 60
            print("Time for SCLOPF [min]:", round(z, 2))
        
            if args['csv_export'] != False:
                path=args['csv_export']+ '/sclopf'
                results_to_csv(network, args, path)
            
    return network