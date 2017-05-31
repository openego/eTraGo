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


import pdb
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.linalg import norm

def prepare_network(network, how='daily', normed=True):
    """ Hierachical clustering of timeseries returning the linkage matrix

    Parameters
    -----------
    network : pyPSA network object

    how : string
       String indicating how to cluster: 'weekly', 'daily' or 'hourly'
    normed : boolean
        If True: normed timeseries will be used for clustering, if False,
        absolute  timeseries of loads and renewable-production will be
        used for clustering


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

def linkage(df, n_groups):
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
    # create linkage matrix with wards algorithm an euclidean norm
    Z = hac.linkage(y, method='ward', metric='euclidean')
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
        print('Computing medoid for cluster: ', c, '...')
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

def update_data_frames(network, medoids, squeze=False):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    medoids : dictionary
        dictionary with medoids created by 'cluster'-function (s.above)
    squeze : Boolean
        Remove data from pyPSA dataframe except the selected medoids (this
        is not necessary as if the snapshots are set correctly this data
        will be skipped anyway. But it can be beneficial for processing.)

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

    if squeze:
        # replace p_sets with new dataframes, therefore create empty data
        l = network.loads_t.p_set.loc[dates]
        network.loads_t.p_set = l
        network.loads_t.p_set.dropna(inplace=True)

        g = network.generators_t.p_max_pu.loc[dates]
        network.generators_t.p_max_pu = g
        network.generators_t.p_max_pu.dropna(inplace=True)

        g1 = network.generators_t.p_set.loc[dates]
        network.generators_t.p_set = g1
        network.generators_t.p_set.dropna(inplace=True)
        # p (results)
#        s = network.storage_units_t.p.loc[dates]
#        network.storage_units_t.p = s
#        network.storage_units_t.p.dropna(inplace=True)
#
#        l1 = network.loads_t.p.loc[dates]
#        network.loads_t.p = l1
#        network.loads_t.p.dropna(inplace=True)
#
#        g2 = network.generators_t.p.loc[dates]
#        network.generators_t.p = g2
#        network.generators_t.p.dropna(inplace=True)

    # set snapshots based on manipulated snapshot weighting index
    network.snapshots = network.snapshot_weightings.index
    network.snapshots = network.snapshots.sort_values()

    return network

