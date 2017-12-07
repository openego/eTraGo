# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:31:39 2017

@author: Kim
"""

import os
import pandas as pd
import pyomo.environ as po
from cluster.snapshot import group, linkage, fcluster, get_medoids
from pypsa.opf import network_lopf

write_results = False
home = os.path.expanduser("~")
resultspath = os.path.join(home, 'snapshot-clustering-results',) # args['scn_name'])

def snapshot_clustering(network, how='daily', clusters= []):

    for c in clusters:
        path = os.path.join(resultspath, how)

        run(network=network.copy(), path=path,
            write_results=write_results, n_clusters=c,
            how=how, normed=False)
        
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

    #snapshots = network.snapshots
    
    # start powerflow calculations
    network_lopf(network, snapshots, extra_functionality = daily_bounds,
                 solver_name='gurobi')
#==============================================================================
#     # write results to csv
#     if write_results:
#         results_to_csv(network, path)
# 
#         write_lpfile(network, path=os.path.join(path, "file.lp"))
#==============================================================================

    return network        

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

####################################??????????????????????????????????????
def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf

def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels':True})

def fix_storage_capacity(network,resultspath, n_clusters): ###"network" dazugefügt
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-'+resultspath

    return resultspath









