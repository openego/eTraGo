"""
"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "Simon Hilpert"

import os
import pandas as pd

import numpy as np
from numpy import genfromtxt
np.random.seed()
import time
from etrago.tools.io import NetworkScenario, results_to_oedb
from etrago.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution)
from etrago.tools.utilities import (oedb_session, load_shedding, data_manipulation_sh,
                                    results_to_csv, parallelisation, pf_post_lopf, 
                                    loading_minimization, calc_line_losses, group_parallel_lines)
from etrago.cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering

from etrago.cluster.snapshot import group, linkage, fcluster, get_medoids
from pypsa.opf import network_lopf
import pyomo.environ as po

###############################################################################
def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """
    if network.cluster:

        sus = network.storage_units

        network.model.period_ends = pd.DatetimeIndex(
                [i for i in network.snapshot_weightings.index[0::24]] +
                [network.snapshot_weightings.index[-1]])


        network.model.storages = sus.index
        def week_rule(m, s, p):
            return m.state_of_charge[s, p] == (sus.at[s, 'max_hours'] *
                                               0.5 * m.storage_p_nom[s])
        network.model.period_bound = po.Constraint(network.model.storages,
                                                   network.model.period_ends,
                                                   rule=week_rule)

######################## Functions to plot / store results ####################
def results_to_csv(network, path):
    """
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    network.export_to_csv_folder(path)
    data = pd.read_csv(os.path.join(path, 'network.csv'))
    data['time'] = network.results['Solver'].Time
    data.to_csv(os.path.join(path, 'network.csv'))

    if hasattr(network, 'Z'):
        file = [i for i in os.listdir(path.strip('0123456789')) if i=='Z.csv']
        if file:
           print('Z already calculated')
        else:
           network.Z.to_csv(path.strip('0123456789')+'/Z.csv', index=False)


def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf

def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels':True})

def fix_storage_capacity(resultspath, n_clusters):
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-'+resultspath

    return resultspath

def prepare_pypsa_timeseries(network, normed=False):
    """
    """

    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_renewables = network.generators_t.p_max_pu

        df = pd.concat([normed_renewables,
                        normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set
        renewables = network.generators_t.p_set
        df = pd.concat([renewables, loads], axis=1)


    return df

def update_data_frames(network, medoids):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    medoids : dictionary
        dictionary with medoids created by 'cluster'-function (s.above)


    Returns
    -------
    network

    """
    # merge all the dates
    dates = medoids[1]['dates'].append(other=[medoids[m]['dates']
                                       for m in medoids])
    # remove duplicates
    dates = dates.unique()
    # sort the index
    dates = dates.sort_values()

    # set snapshots weights
    network.snapshot_weightings = network.snapshot_weightings.loc[dates]
    for m in medoids:
        network.snapshot_weightings[medoids[m]['dates']] = medoids[m]['size']

    # set snapshots based on manipulated snapshot weighting index
    network.snapshots = network.snapshot_weightings.index
    network.snapshots = network.snapshots.sort_values()

    return network



def run(network, path, write_results=False, n_clusters=None, how='daily',
        normed=False):
    """
    """
    # reduce storage costs due to clusters

    if n_clusters is not None:
        path = os.path.join(path, str(n_clusters))

        network.cluster = True

        # calculate clusters

        timeseries_df = prepare_pypsa_timeseries(network, normed=normed)

        df, n_groups = group(timeseries_df, how=how)

        Z = linkage(df, n_groups)

        network.Z = pd.DataFrame(Z)

        clusters = fcluster(df, Z, n_groups, n_clusters)

        medoids = get_medoids(clusters)

        update_data_frames(network, medoids)

        snapshots = network.snapshots

    else:
        network.cluster = False
        path = os.path.join(path, 'original')

    snapshots = network.snapshots
    # start powerflow calculations
    network_lopf(network, snapshots, extra_functionality=daily_bounds,
                 solver_name='gurobi')



    # write results to csv
    if write_results:
        results_to_csv(network, path)

        write_lpfile(network, path=os.path.join(path, "file.lp"))

    return network


###############################################################################

args = {'network_clustering':False, #!!Fehlermeldung assert-Statement // Solved in Feature-branch
        'db': 'oedb', # db session
        'gridversion':'v0.2.11', #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': False, #state whether you want to perform a pf after a lopf simulation
        'start_snapshot': 1,
        'end_snapshot' : 168,
        'scn_name': 'SH NEP 2035',
        'lpfile': '/home/openego/file.lp', # state if and where you want to save pyomo's lp file: False or '/path/tofolder'
        'results': False , # state if and where you want to save results as csv: False or '/path/tofolder'
        'export': False, # state if you want to export the results back to the database
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':True,
        'load_shedding':False,
        'generator_noise':True,
        'extra_functionality':daily_bounds,
        'k_mean_clustering': False,
        'parallelisation':False,
        'line_grouping': False,
        'comments': None}


  
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

# network clustering
if args['network_clustering']:
    network.generators.control="PV"
    busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
    network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

# k-mean clustering
if args['k_mean_clustering']:
    network = kmean_clustering(network)


###############################################################################
# Run scenarios .....
###############################################################################

how = 'daily'
clusters =[i for i in range(1,2)]#[7] +  [i for i in range(1,2)]
write_results = True

home = os.path.expanduser("~")
resultspath = os.path.join(home, 'snapshot-clustering-results', args['scn_name'])

# This will calculate the original problem
run(network=network.copy(), path=resultspath,
    write_results=write_results, n_clusters=None)

if clusters:
    # This will claculate the aggregated problems
    for c in clusters:
        path = os.path.join(resultspath, how)

        run(network=network.copy(), path=path,
            write_results=write_results, n_clusters=c,
            how=how, normed=False)

session.close()

