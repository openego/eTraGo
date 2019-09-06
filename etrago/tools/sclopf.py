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

def post_contingency_analysis_per_line(network):
    
    network.lines.s_nom = network.lines.s_nom_opt.copy()
    network.generators_t.p_set = network.generators_t.p_set.reindex(columns=network.generators.index)
    network.generators_t.p_set = network.generators_t.p
    network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(columns=network.storage_units.index)
    network.storage_units_t.p_set = network.storage_units_t.p
    network.links_t.p_set = network.links_t.p_set.reindex(columns=network.links.index)
    network.links_t.p_set = network.links_t.p0


    d = dict.fromkeys(network.snapshots)
    
    for sn in network.snapshots:
        
        #Check no lines are overloaded with the linear contingency analysis
        p0_test = network.lpf_contingency(sn)
        # rows: branch outage, index = monitorred line
        
        #check loading as per unit of s_nom in each contingency
        overloaded = (abs(p0_test.divide(network.passive_branches().s_nom,axis=0))>1.00001).drop(['base'], axis=1).transpose()# rows: branch outage, index = monitorred line
        combinations = np.where(overloaded.values)
        # solve problem that np. start counting with 0, pypsa with 1!
        d[sn]=combinations
    return d

def add_reduced_contingency_constraints(network,snapshots):
        combinations = post_contingency_analysis_per_line(network)
        #a list of tuples with branch_outage and passive branches in same sub_network
        branch_outage_keys = []
        flow_upper = {}
        flow_lower = {}
        sub = network.sub_networks.obj[0]
        sub._branches["_i"] = range(sub._branches.shape[0])
        sub._extendable_branches =  sub._branches[ sub._branches.s_nom_extendable]
        sub._fixed_branches = sub._branches[~  sub._branches.s_nom_extendable]
        for sn in combinations:
            line_outage= (combinations[sn][0]+1).astype(str)
            monitorred_line = (combinations[sn][1]+1).astype(str)
            
            branch_outage_keys.extend([('Line',line_outage[i],'Line',monitorred_line[i], sn) for i in range(len(line_outage))])

            flow_upper.update({('Line',line_outage[i],'Line',monitorred_line[i],sn) : 
                [[(1,network.model.passive_branch_p['Line',monitorred_line[i],sn]),
                  (sub.BODF[monitorred_line[i].astype(int)-1,line_outage[i].astype(int)-1],
                   network.model.passive_branch_p['Line',line_outage[i],sn])],
                   "<=",sub._fixed_branches.at[('Line', monitorred_line[i]),"s_nom"]] 
                 for i in range(len(line_outage))})
    
            flow_lower.update({('Line',line_outage[i],'Line',monitorred_line[i],sn) : 
                [[(1,network.model.passive_branch_p['Line',monitorred_line[i],sn]),
                  (sub.BODF[monitorred_line[i].astype(int)-1,line_outage[i].astype(int)-1],
                   network.model.passive_branch_p['Line',line_outage[i],sn])],
                   ">=",-sub._fixed_branches.at[('Line', monitorred_line[i]),"s_nom"]] 
                 for i in range(len(line_outage))})  

        l_constraint(network.model,"contingency_flow_upper",flow_upper,branch_outage_keys)
       # import pdb; pdb.set_trace()

        l_constraint(network.model,"contingency_flow_lower",flow_lower,branch_outage_keys)

        #return combinations


def sclopf_post_lopf(network, args):
    logger.info("Contingengcy analysis started at "+ str(datetime.datetime.now()))
    x = time.time()
    add_reduced_contingency_constraints(network,network.snapshots)
    network_lopf_solve(network, 
                       network.snapshots, 
                       formulation=args['model_formulation'], 
                       solver_options=args['solver_options'])
    
    """network.lopf(solver_name=args['solver'], skip_pre=True,
                 extra_functionality=add_reduced_contingency_constraints,
                 solver_options=args['solver_options'], 
                 formulation='kirchhoff')"""
    y = (time.time() - x)/60
    
    if post_contingency_analysis(network).any().any():
        logger.warning("Contingengy analysis failed in "+ str(round(y, 2))+ " minutes. Try iterating a second SCLOPF with additional branch outages.")
    
    else:
        logger.info("Contingengy analysis finished in "+ str(round(y, 2))+ " minutes.")
        
    """relevant_outages = post_contingency_analysis(network)
    branch_outages = relevant_outages.any()[relevant_outages.any()==True].index
    n = 0
    while len(branch_outages) > 0:
        print(branch_outages)
        iterate_sclopf(network, args, branch_outages, extra_functionality=None, 
                   method={'n_iter':5}, delta_s_max=0)
        print(network.model.contingency_flow_lower_index.display())
        relevant_outages = post_contingency_analysis(network)
        branch_outages = relevant_outages.any()[relevant_outages.any()==True].index
        n+=1
    else:
        logger.info("Contingency solved with " + str(n) + " SCLOPF(s).")"""


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