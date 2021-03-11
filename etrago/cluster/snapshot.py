# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems


# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description for read-the-docs
""" This module contains functions for calculating representative days/weeks
based on a pyPSA network object. It is designed to be used for the `lopf`
method. Essentially the tsam package
( https://github.com/FZJ-IEK3-VSA/tsam ), which is developed by
Leander Kotzur is used.

Remaining questions/tasks:

- Does it makes sense to cluster normed values?
- Include scaling method for yearly sums
"""

import pandas as pd
import os
if 'READTHEDOCS' not in os.environ:
    import pyomo.environ as po
    import tsam.timeseriesaggregation as tsam

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "Simon Hilpert"


def snapshot_clustering(self):
    """
    """
    if self.args['snapshot_clustering']['active']:

        self.network = run(network=self.network.copy(),
                      n_clusters=self.args['snapshot_clustering']['n_clusters'],
                      how=self.args['snapshot_clustering']['how'],
                      normed=False)



def tsam_cluster(timeseries_df,
                 typical_periods=10,
                 how='daily',
                 extremePeriodMethod = 'None'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster
    extremePeriodMethod: {'None','append','new_cluster_center',
                           'replace_cluster_center'}, default: 'None'
        Method how to integrate extreme Periods
        into to the typical period profiles.
        None: No integration at all.
        'append': append typical Periods to cluster centers
        'new_cluster_center': add the extreme period as additional cluster
            center. It is checked then for all Periods if they fit better
            to the this new center or their original cluster center.
        'replace_cluster_center': replaces the cluster center of the
            cluster where the extreme period belongs to with the periodly
            profile of the extreme period. (Worst case system design)

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
        period = ' days'
    if how == 'weekly':
        hours = 168
        period = ' weeks'

    print('Snapshot clustering to ' + str(typical_periods) + period +
          ' using extreme period method: ' + extremePeriodMethod)

    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        extremePeriodMethod = extremePeriodMethod,
        addPeakMin = ['residual_load'],
        addPeakMax = ['residual_load'],
        rescaleClusterPeriods=False,
        hoursPerPeriod=hours,
        clusterMethod='hierarchical')


    timeseries = aggregation.createTypicalPeriods()
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterOrder =aggregation.clusterOrder
    clusterCenterIndices= aggregation.clusterCenterIndices

    if extremePeriodMethod  == 'new_cluster_center':
        for i in aggregation.extremePeriods.keys():
            clusterCenterIndices.insert(
                    aggregation.extremePeriods[i]['newClusterNo'],
                    aggregation.extremePeriods[i]['stepNo'])

    if extremePeriodMethod  == 'append':
        for i in aggregation.extremePeriods.keys():
            clusterCenterIndices.insert(
                    aggregation.extremePeriods[i]['clusterNo'],
                    aggregation.extremePeriods[i]['stepNo'])

    # get all index for every hour of that day of the clusterCenterIndices
    start = []
    # get the first hour of the clusterCenterIndices (days start with 0)
    for i in clusterCenterIndices:
        start.append(i * hours)

    # get a list with all hours belonging to the clusterCenterIndices
    nrhours = []
    for j in start:
        nrhours.append(j)
        x = 1
        while x < hours:
            j = j + 1
            nrhours.append(j)
            x = x + 1

    # get the origial Datetimeindex
    dates = timeseries_df.iloc[nrhours].index

    #get list of representative days
    representative_day=[]

    #cluster:medoid des jeweiligen Clusters
    dic_clusterCenterIndices = dict(enumerate(clusterCenterIndices))
    for i in clusterOrder:
        representative_day.append(dic_clusterCenterIndices[i])

    #get list of last hour of representative days
    last_hour_datetime=[]
    for i in representative_day:
        last_hour = i * hours + hours - 1
        last_hour_datetime.append(timeseries_df.index[last_hour])

    #create a dataframe (index=nr. of day in a year/candidate)
    df_cluster =  pd.DataFrame({
                        'Cluster': clusterOrder, #Cluster of the day
                        'RepresentativeDay': representative_day, #representative day of the cluster
                        'last_hour_RepresentativeDay': last_hour_datetime}) #last hour of the cluster
    df_cluster.index = df_cluster.index + 1
    df_cluster.index.name = 'Candidate'

    #create a dataframe each timeseries (h) and its candiddate day (i) df_i_h
    nr_day = []
    x = len(timeseries_df.index)/hours+1

    for i in range(1,int(x)):
        j=1
        while j <= hours:
            nr_day.append(i)
            j=j+1
    df_i_h = pd.DataFrame({'Timeseries': timeseries_df.index,
                        'Candidate_day': nr_day})
    df_i_h.set_index('Timeseries',inplace=True)

    return df_cluster, cluster_weights, dates, hours, df_i_h


def run(network, n_clusters=None, how='daily',
        normed=False):
    """
    """

    # calculate clusters
    df_cluster, cluster_weights, dates, hours, df_i_h= tsam_cluster(
                prepare_pypsa_timeseries(network),
                typical_periods=n_clusters,
                how='daily',
                extremePeriodMethod = 'None')
    network.cluster = df_cluster
    network.cluster_ts = df_i_h

    update_data_frames(network, cluster_weights, dates, hours)

    return network


def prepare_pypsa_timeseries(network, normed=False):
    """
    """
    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_loads.columns = 'L' + normed_loads.columns
        normed_renewables = network.generators_t.p_max_pu
        normed_renewables.columns = 'G' + normed_renewables.columns

        df = pd.concat([normed_renewables,
                        normed_loads], axis=1)
    else:
        loads = network.loads_t.p_set.copy()
        loads.columns = 'L' + loads.columns
        renewables = network.generators_t.p_max_pu.mul(
                network.generators.p_nom[
                network.generators_t.p_max_pu.columns], axis = 1).copy()
        renewables.columns = 'G' + renewables.columns
        residual_load=pd.DataFrame()
        residual_load['residual_load']=loads.sum(axis=1)-renewables.sum(axis=1)
        df = pd.concat([renewables, loads, residual_load], axis=1)

    return df


def update_data_frames(network, cluster_weights, dates, hours):
    """ Updates the snapshots, snapshots weights and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    -----------
    network : pyPSA network object
    cluster_weights: dictionary
    dates: Datetimeindex


    Returns
    -------
    network

    """
    network.snapshot_weightings = network.snapshot_weightings.loc[dates]
    network.snapshots = network.snapshot_weightings.index

    # set new snapshot weights from cluster_weights
    snapshot_weightings = []
    for i in cluster_weights.values():
        x = 0
        while x < hours:
            snapshot_weightings.append(i)
            x += 1
    for i in range(len(network.snapshot_weightings)):
        network.snapshot_weightings[i] = snapshot_weightings[i]

    # put the snapshot in the right order
    network.snapshots.sort_values()
    network.snapshot_weightings.sort_index()

    return network


def skip_snapshots(self):
    n_skip = self.args['skip_snapshots']

    if n_skip:
        self.network.snapshots = self.network.snapshots[::n_skip]

        self.network.snapshot_weightings = \
            self.network.snapshot_weightings[::n_skip] * n_skip

####################################
def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf


def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels': True})


def fix_storage_capacity(network, resultspath, n_clusters):  # "network" added
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-' + resultspath
