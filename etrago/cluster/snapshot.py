# -*- coding: utf-8 -*-
""" This module contains functions for calculating representative days/weeks
based on a pyPSA network object. It is designed to be used the the `lopf`
method.

Use:
    clusters = cluster(network, n_clusters=10)
    medoids = medoids(clusters)
    update_data_frames(network, medoids)

Remaining questions/tasks:

- Does it makes sense to cluster normed values?
- Include scaling method for yearly sums
"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "Simon Hilpert"

import os
import pandas as pd
import pyomo.environ as po
from pypsa.opf import network_lopf
import logging
import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.linalg import norm
from etrago.tools.utilities import results_to_csv

write_results = True
home = os.path.expanduser('C:/eTraGo/etrago')
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
    
    # write results to csv
    if write_results:
        results_to_csv(network, path)

        write_lpfile(network, path=os.path.join(path, "file.lp"))

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

def group(df, how='daily'):
    """ Hierachical clustering of timeseries returning the linkage matrix

    Parameters
    -----------
    df : pandas DataFrame with timeseries to cluster

    how : string
       String indicating how to cluster: 'weekly', 'daily' or 'hourly'
    """

    if df.index.name != 'datetime':
        logging.info('Setting the name of your pd-DataFrame index to: datetime.')
        df.index.name = 'datetime'


    if how == 'daily':
        df['group'] = df.index.dayofyear
        hours = 24
        n_groups = int(len(df.index) / hours) # for one year: 365

        # set new index for dataframe
        df.set_index(['group'], append=True, inplace=True)
        # move 'group' to the first index
        df.index = df.index.swaplevel(0, 'group')

    if how == 'weekly':
        #raise NotImplementedError('The week option is not implemented')
        df['group'] = df.index.weekofyear

        hours = 168
        # for weeks we need to do -1 to exclude the last week, as it might
        # not be 168 hours (is dirty should be done a little more sophistic.)


        # set new index for dataframe
        df.set_index(['group'], append=True, inplace=True)
        # move 'group' to the first index

        df.index = df.index.swaplevel(0, 'group')


        # drop incomplete weeks...
        for g in df.index.get_level_values('group').unique():
            if len(df.loc[g]) != 168:
                df.drop(g, inplace=True)

        n_groups = int(len(df.index) / hours)

    if how == 'hourly':
        raise NotImplementedError('The hourly option is not implemented')

    return df, n_groups

def linkage(df, n_groups, method='ward', metric='euclidean'):
    """
    """

    logging.info("Computing distance matrix...")
    # create the distance matrix based on the forbenius norm: |A-B|_F where A is
    # a 24 x N matrix with N the number of timeseries inside the dataframe df
    # TODO: We can save have time as we only need the upper triangle once as the
    # distance matrix is symmetric
    if True:
        Y = np.empty((n_groups, n_groups,))
        Y[:] = np.NAN
        for i in range(len(Y)):
            for j in range(len(Y[i,:])):
                A = df.loc[i+1].values
                B = df.loc[j+1].values
                #print('Computing distance of:{},{}'.format(i,j))
                Y[i,j] = norm(A-B, ord='fro')

    # condensed distance matrix as vector for linkage (upper triangle as a vector)
    y = Y[np.triu_indices(n_groups, 1)]
    # create linkage matrix with wards algorithm and euclidean norm

    logging.info("Computing linkage Z with method: {0}" \
                 " and metric: {1}...".format(method, metric))
    Z = hac.linkage(y, method=method, metric=metric)
    # R = hac.inconsistent(Z, d=10)
    return Z

def fcluster(df, Z, n_groups, n_clusters):
    """
    """
    # create flat cluster, i.e. maximal number of clusters...
    T = hac.fcluster(Z, criterion='maxclust', depth=2, t=n_clusters)

    # add cluster id to original dataframe
    df['cluster_id'] = np.NAN
    # group is either days (1-365) or weeks (1-52)

    #for d in df.index.get_level_values('group').unique():
    for g in range(1, n_groups+1):
        # T[d-1] because df.index is e.g. 1-365 (d) and T= is 0...364
        df.ix[g, 'cluster_id'] = T[g-1]
    # add the cluster id to the index
    df.set_index(['cluster_id'], append=True, inplace=True)
    # set cluster id as first index level for easier looping through cluster_ids
    df.index = df.index.swaplevel(0, 'cluster_id')
    # just to have datetime at the last level of the multiindex df
    df.index = df.index.swaplevel('datetime', 'group')

    return df

def get_medoids(df):
    """

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe returned by `cluster` function

    Returns
    --------
     Nested dictionary, first key is cluster id. Nested keys are:

        'data': with the representative data for each cluster (medoid)
        'size' : size of the cluster in days/weeks (weight)
        'dates': pandas.datetimeindex with dates of original data of clusters.
    """
    # calculate hours of the group (e.g. 24 for day, 168 for week etc)
    hours = int(len(df) / len(set(df.index.get_level_values('group'))))

    # Finding medoids, clustersizes, etc.
    cluster_group = {}
    cluster_size = {}
    medoids = {}

    # this is necessary fors weeks, because there might be no cluster for
    # elements inside the dataframe, i.e. no complete weeks
    cluster_ids = [i for i in df.index.get_level_values('cluster_id').unique()
                   if not np.isnan(i)]
    for c in cluster_ids:
        logging.info('Computing medoid for cluster: {})'.format(c))
        # days in the cluster is the df subset indexed by the cluster id 'c'
        cluster_group[c] = df.loc[c]
        # the size for daily clusters is the length of all hourly vals / 24
        cluster_size[c] = len(df.loc[c]) / hours

        # store the cluster 'days' i.e. all observations of cluster in 'cluster'
        # TODO: Maybe rather use copy() to keep cluster_days untouched (reference problem)
        cluster = cluster_group[c]

        #pdb.set_trace()
        # Finding medoids (this is a little hackisch but should work correctly):
        # 1) create emtpy distance matrix with size of cluster
        # 2) loop through matrix and add the distance between two 'days'
        # 3) As we want days, we have to slice 24*i...
        Yc = np.empty((int(cluster_size[c]), int(cluster_size[c]),))
        Yc[:] = np.NAN
        for i in range(len(Yc)):
            for j in range(len(Yc[i,:])):
                A = cluster.iloc[hours*i:hours*i+hours].values
                B = cluster.iloc[hours*j:hours*j+hours].values
                Yc[i,j] = norm(A-B, ord='fro')
        # taking the index with the minimum summed distance as medoid
        mid = np.argmin(Yc.sum(axis=0))

        # store data about medoids
        medoids[c] = {}
        # find medoid
        medoids[c]['data'] = cluster.iloc[hours*mid:hours*mid+hours]
        # size ( weight)
        medoids[c]['size'] = cluster_size[c]
        # dates from original data
        medoids[c]['dates'] = medoids[c]['data'].index.get_level_values('datetime')

    return medoids

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




