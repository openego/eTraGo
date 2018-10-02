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


def snapshot_clustering(network, how='daily', clusters=10):
    """
    """

    network = run(network=network.copy(), n_clusters=clusters,
                  how=how, normed=False)

    return network


def tsam_cluster(timeseries_df, typical_periods=10, how='daily'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster

    Returns
    -------
    timeseries : pd.DataFrame
        Clustered timeseries
    """

    if how == 'daily':
        hours = 24
    if how == 'weekly':
        hours = 168

    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        rescaleClusterPeriods=False,
        hoursPerPeriod=hours,
        clusterMethod='hierarchical')

    timeseries = aggregation.createTypicalPeriods()
    cluster_weights = aggregation.clusterPeriodNoOccur

    # get the medoids/ the clusterCenterIndices
    clusterCenterIndices = aggregation.clusterCenterIndices

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

    return timeseries, cluster_weights, dates, hours


def run(network, n_clusters=None, how='daily',
        normed=False):
    """
    """

    # calculate clusters
    tsam_ts, cluster_weights, dates, hours = tsam_cluster(
        prepare_pypsa_timeseries(network), typical_periods=n_clusters,
        how=how)

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
        loads = network.loads_t.p_set
        loads.columns = 'L' + loads.columns
        renewables = network.generators_t.p_set
        renewables.columns = 'G' + renewables.columns
        df = pd.concat([renewables, loads], axis=1)

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


def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """

    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    def day_rule(m, s, p):
        """
        Sets the soc of the every first hour to the soc of the last hour
        of the day (i.e. + 23 hours)
        """
        return (
              m.state_of_charge[s, p] ==
               m.state_of_charge[s, p + pd.Timedelta(hours=23)])

    network.model.period_bound = po.Constraint(
            network.model.storages, network.model.period_starts, rule=day_rule)


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
