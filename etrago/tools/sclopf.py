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
    network.model.del_component('cycle_constraints')
    network.model.del_component('cycle_constraints_index')
    network.model.del_component('cycle_constraints_index_0')
    network.model.del_component('cycle_constraints_index_1')

    
    if args['model_formulation']=='kirchhoff':
        define_passive_branch_flows_with_kirchhoff(network,network.snapshots,skip_vars=True)
    else:
        
        logger.error('Currently only implemented for kirchhoff-formulation.')

    network = network_lopf_solve(network, network.snapshots, formulation=args['model_formulation'], solver_options = args['solver_options'])

    return network



def post_contingency_analysis(network, delta = 0.05):
    
    network.lines.s_nom = network.lines.s_nom_opt.copy()
    network.generators_t.p_set = network.generators_t.p_set.reindex(columns=network.generators.index)
    network.generators_t.p_set = network.generators_t.p
    network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(columns=network.storage_units.index)
    network.storage_units_t.p_set = network.storage_units_t.p
    network.links_t.p_set = network.links_t.p_set.reindex(columns=network.links.index)
    network.links_t.p_set = network.links_t.p0


    df = pd.DataFrame(index =network.snapshots, columns = network.lines.index)
    for sn in network.snapshots:
        
        #Check no lines are overloaded with the linear contingency analysis
        p0_test = network.lpf_contingency(sn)
        # rows: branch outage, index = monitorred line
        
        #check loading as per unit of s_nom in each contingency
        overloaded = (abs(p0_test.divide(network.passive_branches().s_nom,axis=0))>(1 + delta)).drop(['base'], axis=1)# rows: branch outage, index = monitorred line
        
        relevant_outages =overloaded.any()[overloaded.any().index !='base']#.reset_index()[0][1:]#.transpose()  
        #pd.Series(index=relevant_outages.index, data = relevant_outages[0].values)
        df[df.index==sn]=pd.DataFrame(relevant_outages).transpose().values
    # returns lines which outages causes overloadings
    return df

def post_contingency_analysis_per_line(network, 
                                       branch_outages, 
                                       n_process = 4, 
                                       delta = 0.05):
    x = time.time()
    nw = network.copy()
    nw.lines.s_nom = nw.lines.s_nom_opt.copy()
    nw.generators_t.p_set = nw.generators_t.p_set.reindex(columns=nw.generators.index)
    nw.generators_t.p_set = nw.generators_t.p
    nw.storage_units_t.p_set = nw.storage_units_t.p_set.reindex(columns=nw.storage_units.index)
    nw.storage_units_t.p_set = nw.storage_units_t.p
    nw.links_t.p_set = nw.links_t.p_set.reindex(columns=nw.links.index)
    nw.links_t.p_set = nw.links_t.p0
    import multiprocessing as mp
    snapshots_set={}
    length = int(nw.snapshots.size / n_process)
    for i in range(n_process):
        snapshots_set[str(i+1)]=nw.snapshots[i*length : (i+1)*length]
    manager = mp.Manager()
    d = manager.dict()
    snapshots_set[str(n_process-1)] = nw.snapshots[i*length :]
    def multi_con(nw, snapshots, d):

        for sn in snapshots:  
            #d = {}
        #Check no lines are overloaded with the linear contingency analysis
            p0_test = nw.lpf_contingency(branch_outages=branch_outages, snapshots=sn)
        # rows: branch outage, index = monitorred line
        
        #check loading as per unit of s_nom in each contingency
            load = abs(p0_test.divide(nw.passive_branches().s_nom_opt,axis=0)).drop(['base'], axis = 1) # columns: branch_outages
            # schon im base_case leitungen teilweise überlastet. Liegt das an x Anpassung? 
            overloaded = (load>(1 + delta))# columns: branch_outages
           # combinations = [network.lines.index[np.where(overloaded.values)[0]],
                                          #      network.lines.index[np.where(overloaded.values)[1]]]
            array_mon = []
            array_out = []
            for col in overloaded:
                mon = overloaded.index[overloaded[col]].tolist()
                out = [col]*len(mon)
                if mon != []:
                    array_mon.extend(mon)
                    array_out.extend(out)
            combinations = [array_out, array_mon]
            if not len(combinations[0]) == 0:
                d[sn]=combinations

    processes = [mp.Process(target=multi_con, args=(nw, snapshots_set[i], d)) for i in snapshots_set]

    
    # Run processes
    for p in processes:
        p.start()
        

# Exit the completed processes
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

# Get process results from the output queue
    #results = [output.get() for p in processes]
        # solve problem that np. start counting with 0, pypsa with 1!

    y = (time.time() - x)/60
    logger.info("Post contingengy check finished in "+ str(round(y, 2))+ " minutes.")
   # import pdb; pdb.set_trace()
    return d


def add_reduced_contingency_constraints(network,combinations):
    add_reduced_contingency_constraints.counter += 1
    branch_outage_keys = []
    flow_upper = {}
    flow_lower = {}
    sub = network.sub_networks.obj[0]
    sub._branches = sub.branches()
    #if add_reduced_contingency_constraints.counter ==1:
    sub.calculate_B_H()
    sub.calculate_PTDF()
    sub.calculate_BODF()
    #print(sub.BODF.max())
    sub._branches["_i"] = range(sub._branches.shape[0])
    sub._extendable_branches =  sub._branches[ sub._branches.s_nom_extendable]
    sub._fixed_branches = sub._branches[~  sub._branches.s_nom_extendable]
    sub._extendable_branches = sub._branches[sub._branches.s_nom_extendable]
    for sn in combinations.keys():
       # import pdb; pdb.set_trace()
       if len(combinations[sn][0])>0:
           out= combinations[sn][0]# branch in pypsa
           mon = combinations[sn][1] # b in pypsa
          # out= (combinations[sn][0]).astype(str)# branch in pypsa
         #  mon = (combinations[sn][1]).astype(str) # b in pypsa
         #  mon_ext = np.where(np.isin(mon[1], sub._extendable_branches.index.get_level_values(1)))[0]
         #  mon_fix = np.where(~np.isin(mon, sub._extendable_branches.index.get_level_values(1)))[0]
           mon_ext = [t for t in mon if t in sub._extendable_branches.index]
           mon_fix = [t for t in mon if t in sub._fixed_branches.index]
           branch_outage_keys.extend([(out[i][0],out[i][1], mon[i][0],mon[i][1], sn)
                                    for i in range(len(out))])

           flow_upper.update({
                ( out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                        (1, network.model.passive_branch_p[mon[i], sn]),
                        (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                        network.model.passive_branch_p[out[i], sn])],
                        "<=", sub._fixed_branches.at[ mon[i],"s_nom"]] 
        for i in range(len(mon_fix))})
    
           flow_upper.update({
                ( out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                (1, network.model.passive_branch_p[ mon[i], sn]),
                (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                network.model.passive_branch_p[ out[i], sn]),
                 (-1,network.model.passive_branch_s_nom[ mon[i]])],"<=",0] 
        for i in range(len(mon_ext))})

    
           flow_lower.update({
                ( out[i][0], out[i][1], mon[i][0],mon[i][1], sn) : [[
                        (1, network.model.passive_branch_p[mon[i], sn]),
                        (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                        network.model.passive_branch_p[out[i], sn])],
                        ">=", -sub._fixed_branches.at[ mon[i],"s_nom"]] 
        for i in range(len(mon_fix))})
    
           flow_lower.update({
                (  out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                (1, network.model.passive_branch_p[ mon[i], sn]),
                (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                network.model.passive_branch_p[ out[i], sn]),
                 (1,network.model.passive_branch_s_nom[ mon[i]])],">=",0] 
        for i in range(len(mon_ext))})


    l_constraint(network.model,"contingency_flow_upper_"+str(add_reduced_contingency_constraints.counter),flow_upper,branch_outage_keys)
    print(len(branch_outage_keys))
    l_constraint(network.model,"contingency_flow_lower_"+str(add_reduced_contingency_constraints.counter),flow_lower,branch_outage_keys)
    return len(branch_outage_keys)

def add_all_contingency_constraints(network,combinations):
    branch_outage_keys = []
    flow_upper = {}
    flow_lower = {}
    sub = network.sub_networks.obj[0]
    sub._branches = sub.branches()
    sub.calculate_B_H()
    sub.calculate_PTDF()
    sub.calculate_BODF()
    sub._branches["_i"] = range(sub._branches.shape[0])
    sub._extendable_branches =  sub._branches[ sub._branches.s_nom_extendable]
    sub._fixed_branches = sub._branches[~  sub._branches.s_nom_extendable]
    sub._extendable_branches = sub._branches[sub._branches.s_nom_extendable]
    for sn in combinations.keys():
        if len(combinations[sn][0])>0:
           out= combinations[sn][0]# branch in pypsa
           mon = combinations[sn][1] # b in pypsa
           mon_ext = [t for t in mon if t in sub._extendable_branches.index]
           mon_fix = [t for t in mon if t in sub._fixed_branches.index]
           #[t for t in sub._extendable_branches.index if t in mon]
           branch_outage_keys.extend([(out[i][0],out[i][1], mon[i][0],mon[i][1], sn)
                                    for i in range(len(out))])
           branch_outage_keys=list(set(branch_outage_keys))
           flow_upper.update({
                ( out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                        (1, network.model.passive_branch_p[mon[i], sn]),
                        (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                        network.model.passive_branch_p[out[i], sn])],
                        "<=", sub._fixed_branches.at[ mon[i],"s_nom"]] 
                for i in range(len(mon_fix))}) 
    # geht aktuell nur wenn alle Ltg extendable bzw. fix sind
    
           flow_upper.update({
                ( out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                (1, network.model.passive_branch_p[ mon[i], sn]),
                (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                network.model.passive_branch_p[ out[i], sn]),
                 (-1,network.model.passive_branch_s_nom[ mon[i]])],"<=",0] 
                for i in range(len(mon_ext))})

    
           flow_lower.update({( out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                        (1, network.model.passive_branch_p['Line', mon[i],sn]),
                        (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                        network.model.passive_branch_p['Line', out[i], sn])],
                        ">=", -sub._fixed_branches.at[('Line', mon[i]),"s_nom"]] 
                    for i in range(len(mon_fix))})
    
           flow_lower.update({
                (out[i][0],out[i][1], mon[i][0],mon[i][1], sn) : [[
                (1, network.model.passive_branch_p[ mon[i], sn]),
                (sub.BODF[int(mon[i][1])-1, int(out[i][1])-1],
                network.model.passive_branch_p[ out[i], sn]),
                 (1,network.model.passive_branch_s_nom[ mon[i]])],">=",0] 
                for i in range(len(mon_ext))})

    
    network.model.del_component('contingency_flow_upper')
    network.model.del_component('contingency_flow_lower')
    network.model.del_component('contingency_flow_upper_index')
    network.model.del_component('contingency_flow_lower_index')
    l_constraint(network.model,"contingency_flow_upper",flow_upper,branch_outage_keys)
    print(len(branch_outage_keys))
    l_constraint(network.model,"contingency_flow_lower",flow_lower,branch_outage_keys)
    return len(branch_outage_keys)

def sclopf_post_lopf(network, args, n_iter = 5, n_process=2):
    # Aktuell verwendbar wenn kein Netzausbau möglich ist, dann auch schnellste und beste Lösung
    logger.info("Contingengcy analysis started at "+ str(datetime.datetime.now()))
    x = time.time()
    add_reduced_contingency_constraints.counter = 0

    if (network.lines.groupby(['bus0', 'bus1']).size()>1).any():
        idx = network.lines.groupby(
                ['bus0', 'bus1'])['s_nom'].transform(max) \
                == network.lines['s_nom']
        can1 =  network.lines[idx]
        idx2 = can1.groupby(['bus0', 'bus1'])['x'].transform(min) == can1['x']
        can2 = can1[idx2]
        can3 = can2.groupby(['bus0', 'bus1'])['index'].transform(min) == can2['index']
        branch_outages=can2[can3].index

    else:
        branch_outages=network.lines.index[network.lines.country=='DE']
    n=0
    nb=0
    combinations = post_contingency_analysis_per_line(network, branch_outages, 4)
    while len(combinations) > 0:
        if  n < 10:
            nb = nb + add_reduced_contingency_constraints(network,combinations)
            logger.info("SCLOPF No. "+ str(n+1) + " started with "
                        + str(2*nb) + " SC-constraints.")
            network_lopf_solve(network, 
                       network.snapshots, 
                       formulation=args['model_formulation'], 
                       solver_options=args['solver_options'])
            if args['csv_export'] != False:
                    path=args['csv_export'] + '/post_sclopf_iteration_'+ str(n)
                    results_to_csv(network, args, path)
            n+=1

            branch_outages=network.lines.index[network.lines.country=='DE']
            combinations  = post_contingency_analysis_per_line(network,branch_outages, 4)

        else: 
            print('Maximum number of iterations reached.')
            break
    y = (time.time() - x)/60
    

    logger.info("Contingengy analysis with " + str(2*nb) + " constraints solved in " + str(n) + " iterations in "+ str(round(y, 2))+ " minutes.")

def calc_new_sc_combinations(combinations, new):
    for sn in new.keys():         
        combinations[sn][0].extend(new[sn][0])
        combinations[sn][1].extend(new[sn][1])
        df = pd.DataFrame([combinations[sn][0], combinations[sn][1]])
        data = df.transpose().drop_duplicates().transpose()    
        combinations[sn] = data.values.tolist()
    return combinations

def iterate_sclopf_new(network, 
                       args, 
                       branch_outages, 
                       extra_functionality, 
                       n_process = 4,
                       method={'combinations':1}):
    # Aktuell verwendbar auch wenn Netzausbau möglich ist. Führt dann aber zu deutlich mer Iterationen. 
    # Ohne Netzausbau ist pf_post_lopf schneller, da nur einzelnen NB geschrieben werden 
    l_snom_pre = network.lines.s_nom.copy()
    t_snom_pre = network.transformers.s_nom.copy()
    add_all_contingency_constraints.counter = 0
    n=0
    x = time.time()
    results_to_csv.counter=0
    # 1. LOPF without SC
    network.lopf(
                    network.snapshots,
                    solver_name=args['solver'],
                    solver_options=args['solver_options'],
                    extra_functionality=extra_functionality,
                    formulation=args['model_formulation'])
    

    
    # Update electrical parameters if network is extendable
    if network.lines.s_nom_extendable.any():
        l_snom_pre, t_snom_pre = \
                    update_electrical_parameters(network, 
                                                 l_snom_pre, t_snom_pre)
    # Calc SC
    new = post_contingency_analysis_per_line(
                network, branch_outages, n_process)

    # Initalzie dict of SC
    combinations =  dict.fromkeys(network.snapshots, [[], []])

    if 'combinations' in method:
        while len(new) > 0:
            if  n < 20:

                combinations = calc_new_sc_combinations(combinations, new)
                
                nb = add_all_contingency_constraints(network, combinations)

                logger.info("SCLOPF No. "+ str(n+1) + " started with " 
                            + str(2*nb) + " SC-constraints.")
               # network.model.write(('2_lp_' + str(n) + '.lp'), io_options={
                #'symbolic_solver_labels': True})
                iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre)
                

                # nur mit dieser Reihenfolge (x anpassen, dann lpf_check) kann Netzausbau n-1 sicher werden
                if network.lines.s_nom_extendable.any():
                    l_snom_pre, t_snom_pre = \
                        update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)

                new = post_contingency_analysis_per_line(
                            network, branch_outages, n_process)

                if args['csv_export'] != False:
                    path=args['csv_export'] + '/post_sclopf_iteration_'+ str(n)
                    results_to_csv(network, args, path)
                n+=1
                
            else: 
                print('Maximum number of iterations reached.')
                break
            
    if 'expansion_threshold' in method:
        # Threshold über aden Ausbau aller Ltg. evtl max diff einzelner Ltg?
        thr = method['expansion_threshold']
        diff=(network.lines.s_nom_opt-network.lines.s_nom_min).sum()*thr/100
        n = 0
            # Stop after 100 iterations to aviod unending loop
        while n <= 100:
                
            if n ==100:
                    print('Maximum number of iterations reached.')
                    break

            combinations = calc_new_sc_combinations(combinations, new)
            
            nb = add_all_contingency_constraints(network, combinations)

            logger.info("SCLOPF No. "+ str(n+1) + " started with " 
                            + str(2*nb) + " SC-constraints.")
            
            pre = (network.lines.s_nom_opt-network.lines.s_nom_min).sum()

            iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre)

            # nur mit dieser Reihenfolge (x anpassen, dann lpf_check) 
            # kann Netzausbau n-1 sicher werden, dann aber auch im n-0 Fall Überlastungen!!
            if network.lines.s_nom_extendable.any():
                    l_snom_pre, t_snom_pre = \
                        update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)

            new = post_contingency_analysis_per_line(
                            network, network.lines.index, n_process)
            if args['csv_export'] != False:
                    path=args['csv_export'] + '/post_sclopf_iteration_'+ str(n)
                    results_to_csv(network, args, path)
            n+=1
                   
            if abs(pre-(network.lines.s_nom_opt-network.lines.s_nom_min).sum()
                ) <= diff:
                    import pdb; pdb.set_trace()
                    print('Expansion-threshold reached after ' +
                          str(n) + ' iterations.')
                    break
                    
            
    if 'threshold' in method:
        thr = method['threshold']
        diff_obj=network.objective*thr/100
        i = 0
            # Stop after 100 iterations to aviod unending loop
        while i <= 100:
                
            if i ==100:
                    print('Maximum number of iterations reached.')
                    break
            
            for sn in new:
                    df = pd.DataFrame([
                        np.append(combinations[sn][0],new[sn][0]).astype(int),
                        np.append(combinations[sn][1],new[sn][1]).astype(int)])
                    data = df.transpose().drop_duplicates().transpose()    
                    combinations[sn] = data.values
            nb = add_all_contingency_constraints(network, combinations)

            logger.info("SCLOPF No. "+ str(i+1) + " started with " 
                            + str(2*nb) + " SC-constraints.")
            pre = network.objective
            iterate_lopf_calc(network, args, l_snom_pre, t_snom_pre)

                # nur mit dieser Reihenfolge (x anpassen, dann lpf_check) kann Netzausbau n-1 sicher werden
            if network.lines.s_nom_extendable.any():
                    l_snom_pre, t_snom_pre = \
                        update_electrical_parameters(network,
                                                 l_snom_pre, t_snom_pre)

            new = post_contingency_analysis_per_line(
                            network, network.lines.index, n_process)
            if args['csv_export'] != False:
                    path=args['csv_export'] + '/post_sclopf_iteration_'+ str(n)
                    results_to_csv(network, args, path)
            i+=1
                   
            if abs(pre-network.objective) <=diff_obj:
                    print('Threshold reached after ' + str(i) + ' iterations.')
                    break
        
        
    y = (time.time() - x)/60
    
    logger.info("SCLOPF with " + str(2*nb) +
                    " constraints solved in " + str(n) + 
                    " iterations in "+ str(round(y, 2))+ " minutes.")

    #print(post_contingency_analysis_per_line(
                    #    network, network.lines.index, 4).values())
def iterate_sclopf(network, args, branch_outages, extra_functionality, 
                   method={'n_iter':4}, delta_s_max=0.1):

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