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
from pypsa.opf import network_lopf_solve
def post_contingency_analysis(network):
    
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
        overloaded = (abs(p0_test.divide(network.passive_branches().s_nom,axis=0))>1.00001).drop(['base'], axis=1)# rows: branch outage, index = monitorred line
        
        relevant_outages =overloaded.any()[overloaded.any().index !='base']#.reset_index()[0][1:]#.transpose()  
        #pd.Series(index=relevant_outages.index, data = relevant_outages[0].values)
        df[df.index==sn]=pd.DataFrame(relevant_outages).transpose().values
    # returns lines which outages causes overloadings
    return df

def post_contingency_analysis_per_line(network, n_process):
    x = time.time()
    network.lines.s_nom = network.lines.s_nom_opt.copy()
    network.generators_t.p_set = network.generators_t.p_set.reindex(columns=network.generators.index)
    network.generators_t.p_set = network.generators_t.p
    network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(columns=network.storage_units.index)
    network.storage_units_t.p_set = network.storage_units_t.p
    network.links_t.p_set = network.links_t.p_set.reindex(columns=network.links.index)
    network.links_t.p_set = network.links_t.p0

    import multiprocessing as mp
   # output = mp.Queue()
    snapshots_set={}
    length = int(network.snapshots.size / n_process)
    for i in range(n_process):
        snapshots_set[str(i+1)]=network.snapshots[i*length : (i+1)*length]
    manager = mp.Manager()
    d = manager.dict()
    snapshots_set[str(n_process-1)] = network.snapshots[i*length :]
    def multi_con(network, snapshots, d):
        for sn in snapshots:  
            #d = {}
        #Check no lines are overloaded with the linear contingency analysis
            p0_test = network.lpf_contingency(sn)
        # rows: branch outage, index = monitorred line
        
        #check loading as per unit of s_nom in each contingency
            overloaded = (abs(
                p0_test.divide(network.passive_branches().s_nom,axis=0)
                )>1.00001).drop(['base'], axis=1).transpose()# rows: monitorred line, index = branch outage
            combinations = np.where(overloaded.values)
            
            if not combinations[0].size == 0:
                d[sn]=combinations
           # output.put(d)
    processes = [mp.Process(target=multi_con, args=(network, snapshots_set[i], d)) for i in snapshots_set]
    
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
    return d


def add_reduced_contingency_constraints(network,combinations):
    add_reduced_contingency_constraints.counter += 1
    branch_outage_keys = []
    flow_upper = {}
    flow_lower = {}
    sub = network.sub_networks.obj[0]
    sub._branches = sub.branches()
    if add_reduced_contingency_constraints.counter ==1:
        sub.calculate_BODF()
    sub._branches["_i"] = range(sub._branches.shape[0])
    sub._extendable_branches =  sub._branches[ sub._branches.s_nom_extendable]
    sub._fixed_branches = sub._branches[~  sub._branches.s_nom_extendable]
    sub._extendable_branches = sub._branches[sub._branches.s_nom_extendable]
    for sn in combinations:
        out= (combinations[sn][0]+1).astype(str)# branch in pypsa
        mon = (combinations[sn][1]+1).astype(str) # b in pypsa
        mon_ext = np.where(np.isin(mon, sub._extendable_branches.index.get_level_values(1)))[0]
        mon_fix = np.where(~np.isin(mon, sub._extendable_branches.index.get_level_values(1)))[0]
        
        branch_outage_keys.extend([('Line', out[i], 'Line',mon[i], sn)
                                    for i in range(len(out))])

        flow_upper.update({
                ('Line', out[i],'Line', mon[i], sn) : [[
                        (1, network.model.passive_branch_p['Line', mon[i], sn]),
                        (sub.BODF[mon[i].astype(int)-1, out[i].astype(int)-1],
                        network.model.passive_branch_p['Line', out[i], sn])],
                        "<=", sub._fixed_branches.at[('Line', mon[i]),"s_nom"]] 
        for i in mon_fix})
    
        flow_upper.update({
                ('Line', out[i],'Line', mon[i], sn) : [[
                (1, network.model.passive_branch_p['Line', mon[i], sn]),
                (sub.BODF[mon[i].astype(int)-1, out[i].astype(int)-1],
                network.model.passive_branch_p['Line', out[i], sn]),
                 (-1,network.model.passive_branch_s_nom['Line', mon[i]])],"<=",0] 
        for i in mon_ext})

    
        flow_lower.update({('Line', out[i], 'Line', mon[i], sn) : [[
                        (1, network.model.passive_branch_p['Line', mon[i],sn]),
                        (sub.BODF[mon[i].astype(int)-1, out[i].astype(int)-1],
                        network.model.passive_branch_p['Line', out[i], sn])],
                        ">=", -sub._fixed_branches.at[('Line', mon[i]),"s_nom"]] 
        for i in mon_fix})
    
        flow_lower.update({
                ('Line', out[i],'Line', mon[i], sn) : [[
                (1, network.model.passive_branch_p['Line', mon[i], sn]),
                (sub.BODF[mon[i].astype(int)-1, out[i].astype(int)-1],
                network.model.passive_branch_p['Line', out[i], sn]),
                 (1,network.model.passive_branch_s_nom['Line', mon[i]])],">=",0] 
        for i in mon_ext})


    l_constraint(network.model,"contingency_flow_upper_"+str(add_reduced_contingency_constraints.counter),flow_upper,branch_outage_keys)
    print((branch_outage_keys))
    l_constraint(network.model,"contingency_flow_lower_"+str(add_reduced_contingency_constraints.counter),flow_lower,branch_outage_keys)
    return len(branch_outage_keys)


def sclopf_post_lopf(network, args, n_iter_fix = 5, n_process=2):
    logger.info("Contingengcy analysis started at "+ str(datetime.datetime.now()))
    x = time.time()
    add_reduced_contingency_constraints.counter = 0
    combinations = post_contingency_analysis_per_line(network, 4)
    n=0
    nb=0
    while len(combinations) > 0:
        if  n < 10:
                
            logger.info("SCLOPF No. "+ str(n+1) + " started with " + str(2*nb) + " SC-constraints.")
            

            nb = nb + add_reduced_contingency_constraints(network,combinations)
            network_lopf_solve(network, 
                       network.snapshots, 
                       formulation=args['model_formulation'], 
                       solver_options=args['solver_options'])
            if args['csv_export'] != False:
                    path=args['csv_export'] + '/post_sclopf_iteration_'+ str(n)
                    results_to_csv(network, args, path)
            n+=1
            combinations = post_contingency_analysis_per_line(network,4)
            if network.lines.extendable.any():
                l_snom_pre = network.lines.s_nom.copy()
                t_snom_pre = network.transformers.s_nom.copy()
                for i in range(n_iter_fix):

                    l_snom_pre, t_snom_pre = \
                        update_electrical_parameters(network, 
                                                 l_snom_pre, t_snom_pre)
                        
                    # Anpassung x über update der cycyle constraints? Sonst muss das model immer wieder gebaut werden 
        else: 
            print('Maximum number of iterations reached.')
            break
    y = (time.time() - x)/60
    
    logger.info("Contingengy analysis with " + str(2*nb) + " constraints solved in " + str(n) + " iterations in "+ str(round(y, 2))+ " minutes.")
    
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
