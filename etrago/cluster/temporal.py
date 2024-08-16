# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
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
""" This module contains functions for reducing the complexity of a PyPSA
 network in temporal dimension by
a) downsampling to every n-th snapshot
b) clustering to typical periods (eg days, weeks)
c) clustering to segments of variable length
Essentially used is the tsam package
( https://github.com/FZJ-IEK3-VSA/tsam ) developed by Leander Kotzur et al.
"""

import os

import pandas as pd

if "READTHEDOCS" not in os.environ:
    import tsam.timeseriesaggregation as tsam

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ClaraBuettner, ulfmueller, KathiEsterl, simnh, wheitkoetter,
 BartelsJ, AmeliaNadal"""


def snapshot_clustering(self):
    """
    Function to call the snapshot clustering function with the respecting
    method and settings.

    Raises
    ------
    ValueError
        When calling a non-available function.

    Returns
    -------
    None.

    """

    if self.args["snapshot_clustering"]["active"]:
        # save second network for optional dispatch disaggregation
        if self.args["temporal_disaggregation"]["active"]:
            self.network_tsa = self.network.copy()

        if self.args["snapshot_clustering"]["method"] == "segmentation":
            self.network = run(
                network=self.network.copy(),
                n_clusters=1,
                segmented_to=self.args["snapshot_clustering"]["n_segments"],
                extreme_periods=self.args["snapshot_clustering"][
                    "extreme_periods"
                ],
            )

        elif self.args["snapshot_clustering"]["method"] == "typical_periods":
            self.network = run(
                network=self.network.copy(),
                n_clusters=self.args["snapshot_clustering"]["n_clusters"],
                how=self.args["snapshot_clustering"]["how"],
                extreme_periods=self.args["snapshot_clustering"][
                    "extreme_periods"
                ],
            )
        else:
            raise ValueError(
                """Type of clustering should be 'typical_periods' or
                'segmentation'"""
            )


def tsam_cluster(
    timeseries_df,
    typical_periods=10,
    how="daily",
    extremePeriodMethod="None",
    segmentation=False,
    segment_no=10,
    segm_hoursperperiod=24,
):
    """
    Conducts the clustering of the snapshots for temporal aggregation with the
    respecting method.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        Dataframe wit timeseries to cluster.
    typical_periods : int, optional
        Number of clusters for typical_periods. The default is 10.
    how : {'daily', 'weekly', 'monthly'}, optional
        Definition of period for typical_periods. The default is 'daily'.
    extremePeriodMethod : {'None','append','new_cluster_center',
        'replace_cluster_center'}, optional Method to consider extreme
        snapshots in reduced timeseries. The default is 'None'.
    segmentation : boolean, optional
        Argument to activate segmenation method. The default is False.
    segment_no : int, optional
        Number of segments for segmentation. The default is 10.
    segm_hoursperperiod : int, optional
        Only for segmentation, ensures to cluster to segments considering all
        snapshots. The default is 24.

    Returns
    -------
    df_cluster : pd.DataFrame
        Information on cluster after clustering to typical periods.
    cluster_weights : dict
        Weightings per cluster after clustering to typical periods.
    dates : DatetimeIndex
        Dates of clusters after clustering to typical periods.
    hours : int
        Hours per typical period.
    df_i_h : pd.DataFrame
        Information on cluster after clustering to typical periods.
    timeseries : pd.DataFrame
        Information on segments after segmentation.

    """

    if how == "daily":
        hours = 24
        period = " days"

    elif how == "weekly":
        hours = 168
        period = " weeks"

    elif how == "monthly":
        hours = 720
        period = " months"

    elif how == "hourly":
        hours = 1
        period = " hours"

    if segmentation:
        hoursPerPeriod = segm_hoursperperiod
        hours = 1
    else:
        hoursPerPeriod = hours

    # define weight for weightDict:
    # residual load should not impact cluster findings,
    # but only be the optional parameter to choose an extreme period
    weight = pd.Series(data=1, index=timeseries_df.columns)
    weight["residual_load"] = 0
    weight = weight.to_dict()

    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        extremePeriodMethod=extremePeriodMethod,
        addPeakMin=["residual_load"],
        addPeakMax=["residual_load"],
        rescaleClusterPeriods=False,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="hierarchical",
        segmentation=segmentation,
        noSegments=segment_no,
        weightDict=weight,
    )

    if segmentation:
        print(
            "Snapshot clustering to "
            + str(segment_no)
            + " segments"
            + "\n"
            + "Using extreme period method: "
            + extremePeriodMethod
        )

    else:
        print(
            "Snapshot clustering to "
            + str(typical_periods)
            + period
            + "\n"
            + "Using extreme period method: "
            + extremePeriodMethod
        )

    timeseries_creator = aggregation.createTypicalPeriods()
    timeseries = timeseries_creator.copy()

    # If Segmentation is True, insert 'Dates' and 'SegmentNo' column in
    # timeseries
    if segmentation:
        weights = timeseries.index.get_level_values(2)
        dates_df = timeseries_df.index.get_level_values(0)
        dates = []
        segmentno = []
        wcount = 0
        count = 0
        for weight in weights:
            dates.append(dates_df[wcount])
            wcount = wcount + weight
            segmentno.append(count)
            count = count + 1
        timeseries.insert(0, "dates", dates, True)
        timeseries.insert(1, "SegmentNo", segmentno, True)
        timeseries.insert(2, "SegmentDuration", weights, True)
        timeseries.set_index(
            ["dates", "SegmentNo", "SegmentDuration"], inplace=True
        )

        if "Unnamed: 0" in timeseries.columns:
            del timeseries["Unnamed: 0"]
        if "Segment Step" in timeseries.columns:
            del timeseries["Segment Step"]
        # print(timeseries)

    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterOrder = aggregation.clusterOrder
    clusterCenterIndices = aggregation.clusterCenterIndices

    if segmentation:
        if extremePeriodMethod != "None":
            timeseries = segmentation_extreme_periods(
                timeseries_df, timeseries, extremePeriodMethod
            )

    else:
        if extremePeriodMethod == "new_cluster_center":
            for i in aggregation.extremePeriods.keys():
                clusterCenterIndices.insert(
                    aggregation.extremePeriods[i]["newClusterNo"],
                    aggregation.extremePeriods[i]["stepNo"],
                )

        if extremePeriodMethod == "append":
            for i in aggregation.extremePeriods.keys():
                clusterCenterIndices.insert(
                    aggregation.extremePeriods[i]["clusterNo"],
                    aggregation.extremePeriods[i]["stepNo"],
                )

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

    # get list of representative days
    representative_day = []

    # cluster:medoid des jeweiligen Clusters
    dic_clusterCenterIndices = dict(enumerate(clusterCenterIndices))
    for i in clusterOrder:
        representative_day.append(dic_clusterCenterIndices[i])

    # get list of last and first hour of representative days
    last_hour_datetime = []
    for i in representative_day:
        last_hour = i * hours + hours - 1
        last_hour_datetime.append(timeseries_df.index[last_hour])

    # create a dataframe (index=nr. of day in a year/candidate)
    df_cluster = pd.DataFrame(
        {
            "Cluster": clusterOrder,  # Cluster of the day
            "RepresentativeDay": representative_day,  # representative day of
            # the cluster
            "last_hour_RepresentativeDay": last_hour_datetime,
        }
    )  # last hour of the cluster
    df_cluster.index = df_cluster.index + 1
    df_cluster.index.name = "Candidate"

    # create a dataframe each timeseries (h) and its candiddate day (i) df_i_h
    nr_day = []
    x = len(timeseries_df.index) / hours + 1

    for i in range(1, int(x)):
        j = 1
        while j <= hours:
            nr_day.append(i)
            j = j + 1
    df_i_h = pd.DataFrame(
        {"Timeseries": timeseries_df.index, "Candidate_day": nr_day}
    )
    df_i_h.set_index("Timeseries", inplace=True)

    return df_cluster, cluster_weights, dates, hours, df_i_h, timeseries


def segmentation_extreme_periods(
    timeseries_df, timeseries, extremePeriodMethod
):
    """
    Function to consider extreme snapshots while using segmentation.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        Dataframe wit timeseries to cluster.
    timeseries : pd.DataFrame
        Information on segments after segmentation.
    extremePeriodMethod : {'None','append','new_cluster_center',
        'replace_cluster_center'}, optional method to consider extreme
        snapshots in reduced timeseries. The default is 'None'.

    Raises
    ------
    ValueError
        When calling wrong method to consider extreme values.

    Returns
    -------
    timeseries : pd.DataFrame
        Information on segments including extreme snapshots after segmentation.
    """

    # find maximum / minimum value in residual load
    maxi = timeseries_df["residual_load"].idxmax()
    mini = timeseries_df["residual_load"].idxmin()

    # add timestep if it is not already calculated
    if maxi not in timeseries.index.get_level_values("dates"):
        # identifiy timestep, adapt it to timeseries-df and add it
        max_val = timeseries_df.loc[maxi].copy()
        max_val["SegmentNo"] = len(timeseries)
        max_val["SegmentDuration"] = 1
        max_val["dates"] = max_val.name
        max_val = pd.DataFrame(max_val).transpose()

        if extremePeriodMethod == "append":
            max_val.set_index(
                ["dates", "SegmentNo", "SegmentDuration"], inplace=True
            )
            timeseries = timeseries.append(max_val)
            timeseries = timeseries.sort_values(by="dates")

            # split up segment in which the extreme timestep was added
            i = -1
            for date in timeseries.index.get_level_values("dates"):
                if date < maxi:
                    i = i + 1
                else:
                    timeseries["SegmentDuration_Extreme"] = (
                        timeseries.index.get_level_values("SegmentDuration")
                    )
                    old_row = timeseries.iloc[i].copy()
                    old_row = pd.DataFrame(old_row).transpose()

                    delta_t = (
                        timeseries.index.get_level_values("dates")[i + 1]
                        - timeseries.index.get_level_values("dates")[i]
                    )
                    delta_t = delta_t.total_seconds() / 3600
                    timeseries["SegmentDuration_Extreme"].iloc[i] = delta_t

                    timeseries_df["row_no"] = range(0, len(timeseries_df))
                    new_row = int(timeseries_df.loc[maxi]["row_no"]) + 1
                    new_date = timeseries_df[
                        timeseries_df.row_no == new_row
                    ].index

                    if new_date.isin(
                        timeseries.index.get_level_values("dates")
                    ):
                        timeseries["dates"] = (
                            timeseries.index.get_level_values("dates")
                        )
                        timeseries["SegmentNo"] = (
                            timeseries.index.get_level_values("SegmentNo")
                        )
                        timeseries["SegmentDuration"] = timeseries[
                            "SegmentDuration_Extreme"
                        ]
                        timeseries.drop(
                            "SegmentDuration_Extreme", axis=1, inplace=True
                        )
                        timeseries.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        break
                    else:
                        new_row = timeseries_df.iloc[new_row].copy()
                        new_row.drop("row_no", inplace=True)
                        new_row["SegmentNo"] = len(timeseries)
                        new_row["SegmentDuration"] = (
                            old_row["SegmentDuration_Extreme"][0] - delta_t - 1
                        )
                        new_row["dates"] = new_row.name
                        new_row = pd.DataFrame(new_row).transpose()
                        new_row.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        for col in new_row.columns:
                            new_row[col][0] = old_row[col][0]

                        timeseries["dates"] = (
                            timeseries.index.get_level_values("dates")
                        )
                        timeseries["SegmentNo"] = (
                            timeseries.index.get_level_values("SegmentNo")
                        )
                        timeseries["SegmentDuration"] = timeseries[
                            "SegmentDuration_Extreme"
                        ]
                        timeseries.drop(
                            "SegmentDuration_Extreme", axis=1, inplace=True
                        )
                        timeseries.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        timeseries = timeseries.append(new_row)
                        timeseries = timeseries.sort_values(by="dates")
                        break

        elif extremePeriodMethod == "replace_cluster_center":
            # replace segment in which the extreme timestep was added
            i = -1
            for date in timeseries.index.get_level_values("dates"):
                if date < maxi:
                    i = i + 1
                else:
                    if i == -1:
                        i = 0
                    max_val["SegmentDuration"] = (
                        timeseries.index.get_level_values("SegmentDuration")[i]
                    )
                    max_val.set_index(
                        ["dates", "SegmentNo", "SegmentDuration"], inplace=True
                    )
                    timeseries.drop(timeseries.index[i], inplace=True)
                    timeseries = timeseries.append(max_val)
                    timeseries = timeseries.sort_values(by="dates")
                    break

        else:
            raise ValueError(
                """Choose 'append' or 'replace_cluster_center' for
                 consideration of extreme periods with segmentation method"""
            )

    # add timestep if it is not already calculated
    if mini not in timeseries.index.get_level_values("dates"):
        # identifiy timestep, adapt it to timeseries-df and add it
        min_val = timeseries_df.loc[mini].copy()
        min_val["SegmentNo"] = len(timeseries) + 1
        min_val["SegmentDuration"] = 1
        min_val["dates"] = min_val.name
        min_val = pd.DataFrame(min_val).transpose()

        if extremePeriodMethod == "append":
            min_val.set_index(
                ["dates", "SegmentNo", "SegmentDuration"], inplace=True
            )
            timeseries = timeseries.append(min_val)
            timeseries = timeseries.sort_values(by="dates")

            # split up segment in which the extreme timestep was added
            i = -1
            for date in timeseries.index.get_level_values("dates"):
                if date < mini:
                    i = i + 1
                else:
                    timeseries["SegmentDuration_Extreme"] = (
                        timeseries.index.get_level_values("SegmentDuration")
                    )
                    old_row = timeseries.iloc[i].copy()
                    old_row = pd.DataFrame(old_row).transpose()

                    delta_t = (
                        timeseries.index.get_level_values("dates")[i + 1]
                        - timeseries.index.get_level_values("dates")[i]
                    )
                    delta_t = delta_t.total_seconds() / 3600
                    timeseries["SegmentDuration_Extreme"].iloc[i] = delta_t

                    timeseries_df["row_no"] = range(0, len(timeseries_df))
                    new_row = int(timeseries_df.loc[mini]["row_no"]) + 1
                    new_date = timeseries_df[
                        timeseries_df.row_no == new_row
                    ].index

                    if new_date.isin(
                        timeseries.index.get_level_values("dates")
                    ):
                        timeseries["dates"] = (
                            timeseries.index.get_level_values("dates")
                        )
                        timeseries["SegmentNo"] = (
                            timeseries.index.get_level_values("SegmentNo")
                        )
                        timeseries["SegmentDuration"] = timeseries[
                            "SegmentDuration_Extreme"
                        ]
                        timeseries.drop(
                            "SegmentDuration_Extreme", axis=1, inplace=True
                        )
                        timeseries.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        break
                    else:
                        new_row = timeseries_df.iloc[new_row].copy()
                        new_row.drop("row_no", inplace=True)
                        new_row["SegmentNo"] = len(timeseries) + 1
                        new_row["SegmentDuration"] = (
                            old_row["SegmentDuration_Extreme"][0] - delta_t - 1
                        )
                        new_row["dates"] = new_row.name
                        new_row = pd.DataFrame(new_row).transpose()
                        new_row.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        for col in new_row.columns:
                            new_row[col][0] = old_row[col][0]
                        timeseries["dates"] = (
                            timeseries.index.get_level_values("dates")
                        )
                        timeseries["SegmentNo"] = (
                            timeseries.index.get_level_values("SegmentNo")
                        )
                        timeseries["SegmentDuration"] = timeseries[
                            "SegmentDuration_Extreme"
                        ]
                        timeseries.drop(
                            "SegmentDuration_Extreme", axis=1, inplace=True
                        )
                        timeseries.set_index(
                            ["dates", "SegmentNo", "SegmentDuration"],
                            inplace=True,
                        )
                        timeseries = timeseries.append(new_row)
                        timeseries = timeseries.sort_values(by="dates")
                    break

        elif extremePeriodMethod == "replace_cluster_center":
            # replace segment in which the extreme timestep was added
            i = -1
            for date in timeseries.index.get_level_values("dates"):
                if date < mini:
                    i = i + 1
                else:
                    if i == -1:
                        i = 0
                    min_val["SegmentDuration"] = (
                        timeseries.index.get_level_values("SegmentDuration")[i]
                    )
                    min_val.set_index(
                        ["dates", "SegmentNo", "SegmentDuration"], inplace=True
                    )
                    timeseries.drop(timeseries.index[i], inplace=True)
                    timeseries = timeseries.append(min_val)
                    timeseries = timeseries.sort_values(by="dates")
                    break

        else:
            raise ValueError(
                """Choose 'append' or 'replace_cluster_center' for
                consideration of extreme periods with segmentation method"""
            )

    if "row_no" in timeseries.columns:
        timeseries.drop("row_no", axis=1, inplace=True)

    return timeseries


def run(
    network,
    n_clusters=None,
    how="daily",
    segmented_to=False,
    extreme_periods="None",
):
    """
    Function to call the respecting snapshot clustering function and export the
    result to a csv-file.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    n_clusters : int, optional
        Number of clusters for typical_periods. The default is None.
    how : {'daily', 'weekly', 'monthly'}, optional
        Definition of period for typical_periods. The default is 'daily'.
    segmented_to : int, optional
        Number of segments for segmentation. The default is False.
    extremePeriodMethod : {'None','append','new_cluster_center',
        'replace_cluster_center'}, optional
        Method to consider extreme snapshots in reduced timeseries.
        The default is 'None'.

    Returns
    -------
    network : pypsa.Network object
        Container for all network components.

    """

    if segmented_to is not False:
        segment_no = segmented_to
        segmentation = True

    else:
        segment_no = 24
        segmentation = False

    if not extreme_periods:
        extreme_periods = "None"

    # calculate clusters
    (
        df_cluster,
        cluster_weights,
        dates,
        hours,
        df_i_h,
        timeseries,
    ) = tsam_cluster(
        prepare_pypsa_timeseries(network),
        typical_periods=n_clusters,
        how="daily",
        extremePeriodMethod=extreme_periods,
        segmentation=segmentation,
        segment_no=segment_no,
        segm_hoursperperiod=network.snapshots.size,
    )

    if segmentation:
        pd.DataFrame(
            timeseries.reset_index(),
            columns=["dates", "SegmentNo", "SegmentDuration"],
        ).set_index("SegmentNo").to_csv(
            "timeseries_segmentation=" + str(segment_no) + ".csv"
        )
    else:
        if how == "daily":
            howie = "days"
        elif how == "weekly":
            howie = "weeks"
        elif how == "monthly":
            howie = "months"
        elif how == "hourly":
            howie = "hours"
        df_cluster.to_csv(
            "cluster_typical-periods=" + str(n_clusters) + howie + ".csv"
        )

    network.cluster = df_cluster
    network.cluster_ts = df_i_h

    update_data_frames(
        network, cluster_weights, dates, hours, timeseries, segmentation
    )

    return network


def prepare_pypsa_timeseries(network):
    """
    Prepares timeseries and residual load timeseries for clustering.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    Returns
    -------
    df : pd.DataFrame
        Timeseries to be considered when clustering.

    """

    loads = network.loads_t.p_set.copy()
    loads.columns = "L" + loads.columns

    renewables = network.generators_t.p_max_pu.mul(
        network.generators.p_nom[network.generators_t.p_max_pu.columns], axis=1
    ).copy()
    renewables.columns = "G" + renewables.columns

    residual_load = pd.DataFrame()
    residual_load["residual_load"] = loads.sum(axis=1) - renewables.sum(axis=1)
    df = pd.concat([renewables, loads, residual_load], axis=1)

    return df


def update_data_frames(
    network, cluster_weights, dates, hours, timeseries, segmentation
):
    """
    Updates the snapshots, snapshot weightings and the dataframes based on
    the original data in the network and the medoids created by clustering
    these original data.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    cluster_weights : dict
        Weightings per cluster after clustering to typical periods.
    dates : DatetimeIndex
        Dates of clusters after clustering to typical periods.
    hours : int
        Hours per typical period.
    timeseries : pd.DataFrame
        Information on segments after segmentation.
    segmentation : boolean
        Checks if segmentation of clustering to typical periods has been used.

    Returns
    -------
    network : pypsa.Network object
        Container for all network components.

    """
    if segmentation:
        network.snapshots = timeseries.index.get_level_values(0)
        network.snapshot_weightings["objective"] = pd.Series(
            data=timeseries.index.get_level_values(2).values,
            index=timeseries.index.get_level_values(0),
        )
        network.snapshot_weightings["stores"] = pd.Series(
            data=timeseries.index.get_level_values(2).values,
            index=timeseries.index.get_level_values(0),
        )
        network.snapshot_weightings["generators"] = pd.Series(
            data=timeseries.index.get_level_values(2).values,
            index=timeseries.index.get_level_values(0),
        )

    else:
        network.snapshots = dates
        network.snapshot_weightings = network.snapshot_weightings.loc[dates]

        snapshot_weightings = []
        for i in cluster_weights.values():
            x = 0
            while x < hours:
                snapshot_weightings.append(i)
                x += 1
        for i in range(len(network.snapshot_weightings)):
            network.snapshot_weightings["objective"][i] = snapshot_weightings[
                i
            ]
            network.snapshot_weightings["stores"][i] = snapshot_weightings[i]
            network.snapshot_weightings["generators"][i] = snapshot_weightings[
                i
            ]

        # put the snapshot in the right order
        network.snapshots.sort_values()
        network.snapshot_weightings.sort_index()

    print(network.snapshots)

    return network


def skip_snapshots(self):
    """
    Conducts the downsapling to every n-th snapshot.

    Returns
    -------
    None.

    """

    # save second network for optional dispatch disaggregation
    if (
        self.args["temporal_disaggregation"]["active"]
        and not self.args["snapshot_clustering"]["active"]
    ) or self.args["method"]["market_optimization"]:
        self.network_tsa = self.network.copy()

    n_skip = self.args["skip_snapshots"]

    if n_skip:
        last_weight = (
            int(
                (
                    self.network.snapshots[-1]
                    - self.network.snapshots[::n_skip][-1]
                ).seconds
                / 3600
            )
            + 1
        )

        self.network.snapshots = self.network.snapshots[::n_skip]

        self.network.snapshot_weightings["objective"] = n_skip
        self.network.snapshot_weightings["stores"] = n_skip
        self.network.snapshot_weightings["generators"] = n_skip

        if last_weight < n_skip:
            self.network.snapshot_weightings.loc[
                self.network.snapshot_weightings.index[-1]
            ]["objective"] = last_weight
            self.network.snapshot_weightings.loc[
                self.network.snapshot_weightings.index[-1]
            ]["stores"] = last_weight
            self.network.snapshot_weightings.loc[
                self.network.snapshot_weightings.index[-1]
            ]["generators"] = last_weight
