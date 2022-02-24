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
    if self.args['snapshot_clustering']['active'] == True:

        if self.args['snapshot_clustering']['method'] == 'segmentation' :
            
            self.network = run(network=self.network.copy(),
                      extremePeriodMethod = self.args['snapshot_clustering']['extreme_periods'],
                      n_clusters=1,
                      segmented_to = self.args['snapshot_clustering']['n_segments'],
                      csv_export = self.args['csv_export'] ### can be deleted later, just helpful to see how time steps were clustered
                      )
            
        elif self.args['snapshot_clustering']['method'] == 'typical_periods' :
            
            self.network = run(network=self.network.copy(),
                      extremePeriodMethod = self.args['snapshot_clustering']['extreme_periods'],
                      n_clusters=self.args['snapshot_clustering']['n_clusters'],
                      how=self.args['snapshot_clustering']['how'],
                      csv_export = self.args['csv_export'] ### can be deleted later
                      )
        else :
                 raise ValueError("Type of clustering should be 'typical_periods' or 'segmentation'")


def tsam_cluster(timeseries_df,
                 typical_periods=10,
                 how='daily',
                 extremePeriodMethod = 'None',
                 segmentation = False,
                 segment_no = 10,
                 segm_hoursperperiod = 24):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster
    typical_periods: Number of typical Periods (or clusters)
    how: {'daily', 'weekly'}
    extremePeriodMethod: {'None','append','new_cluster_center',
                           'replace_cluster_center'}, default: 'None'
        Method how to integrate extreme Periods into to the typical period profiles.
        'None': No integration at all.
        'append': append typical Periods to cluster centers
        'new_cluster_center': add the extreme period as additional cluster
            center. It is checked then for all Periods if they fit better
            to the this new center or their original cluster center.
        'replace_cluster_center': replaces the cluster center of the
            cluster where the extreme period belongs to with the periodly
            profile of the extreme period. (Worst case system design)
    segmentation: Is given by the run-function, can be True or False
    segment_no: Only used when segmentation is true, the number of segments
    segm_hoursperperiod: Only used when segmentation is true, defines the length of a cluster period

    Returns
    -------
    df_cluster
    cluster_weights
    dates
    hours
    df_i_h
    timeseries : pd.DataFrame
                Clustered timeseries, only used if segmentation is True
    """
    
    if how == 'daily':
        hours = 24
        period = ' days'
    if how == 'weekly':
        hours = 168
        period = ' weeks'
        timeseries_df = timeseries_df[24:8760]
        ### if clustering to typical weeks, reduce time horizon to 52 weeks = 8736 timesteps
        ### -> calclulate one day less
    if how == 'hourly':
        hours = 1
        period = ' hours'

    if segmentation:
        hoursPerPeriod = segm_hoursperperiod
    else:
        hoursPerPeriod = hours
    
    # define weight for weightDict: 
    # residual load should not impact cluster findings, 
    # but only be the optional parameter to choose an extreme period
    weight = pd.Series(data=1, index=timeseries_df.columns)
    weight['residual_load'] = 0 
    weight = weight.to_dict()
    
    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        extremePeriodMethod = extremePeriodMethod,
        addPeakMin = ['residual_load'],
        addPeakMax = ['residual_load'],
        rescaleClusterPeriods=False,
        hoursPerPeriod=hoursPerPeriod, 
        clusterMethod='hierarchical',
        segmentation = segmentation, 
        noSegments = segment_no,
        weightDict = weight)
    
    if segmentation:
        print('Snapshot Clustering to ' + str(segment_no) + ' Segments' + '\n' +
              'using extreme period method: ' + extremePeriodMethod +' (append only possible method for segmentation)')
    
    else:
        print('Snapshot Clustering to ' + str(typical_periods) + period + '\n' + 
          'using extreme period method: ' + extremePeriodMethod)


    timeseries_creator = aggregation.createTypicalPeriods()
    timeseries = timeseries_creator.copy()

    #If Segmentation is True, insert 'Dates' and 'SegmentNo' column in timeseries
    if segmentation is True:
        
        weights=timeseries.index.get_level_values(2)
        dates_df= timeseries_df.index.get_level_values(0)
        dates=[]
        segmentno=[]
        wcount=0
        count=0
        for weight in weights:
            dates.append(dates_df[wcount])
            wcount = wcount + weight
            segmentno.append(count)
            count = count +1
        timeseries.insert(0, "dates", dates, True)
        timeseries.insert(1, "SegmentNo", segmentno, True)
        timeseries.insert(2, "SegmentDuration", weights, True)
        timeseries.set_index(['dates', 'SegmentNo', 'SegmentDuration'], inplace=True)

        if 'Unnamed: 0' in timeseries.columns:
            del timeseries['Unnamed: 0']
        if 'Segment Step' in timeseries.columns:
            del timeseries['Segment Step']
        #print(timeseries)
        
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterOrder = aggregation.clusterOrder
    clusterCenterIndices= aggregation.clusterCenterIndices
    
    # for segmentation: optional adding of extreme periods
    if segmentation == True and extremePeriodMethod != 'None':
        
        # find maximum / minimum value in residual load
        maxi = timeseries_df['residual_load'].idxmax()
        mini = timeseries_df['residual_load'].idxmin()
        
        # add timestep if it is not already calculated
        if maxi not in timeseries.index.get_level_values('dates'):
            
            # identifiy timestep, adapt it to timeseries-df and add it
            max_val = timeseries_df.loc[maxi].copy()
            max_val['SegmentNo'] = len(timeseries)
            max_val['SegmentDuration'] = 1
            max_val['dates'] = max_val.name
            max_val = pd.DataFrame(max_val).transpose()
            max_val.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
            timeseries = timeseries.append(max_val)
            timeseries = timeseries.sort_values(by='dates')
            
            # split up segment in which the extreme timestep was added 
            i=-1
            for date in timeseries.index.get_level_values('dates'):
                if date < maxi:
                    i = i+1
                else:
                    timeseries['SegmentDuration_Extreme']=timeseries.index.get_level_values('SegmentDuration')
                    old_row = timeseries.iloc[i].copy()
                    old_row = pd.DataFrame(old_row).transpose()
                    
                    delta_t = timeseries.index.get_level_values('dates')[i+1]-timeseries.index.get_level_values('dates')[i]
                    delta_t = delta_t.total_seconds()/3600
                    timeseries['SegmentDuration_Extreme'].iloc[i]=delta_t
                    
                    timeseries_df['row_no']=range(0,len(timeseries_df))
                    new_row = int(timeseries_df.loc[maxi]['row_no'])+1
                    new_date = timeseries_df[timeseries_df.row_no==new_row].index
                    
                    if new_date.isin(timeseries.index.get_level_values('dates')):
                        timeseries['dates'] = timeseries.index.get_level_values('dates')
                        timeseries['SegmentNo'] = timeseries.index.get_level_values('SegmentNo')
                        timeseries['SegmentDuration'] = timeseries['SegmentDuration_Extreme']
                        timeseries.drop('SegmentDuration_Extreme', axis=1, inplace=True)
                        timeseries.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        break
                    else:                     
                        new_row = timeseries_df.iloc[new_row].copy()
                        new_row.drop('row_no', inplace=True)
                        new_row['SegmentNo'] = len(timeseries)
                        new_row['SegmentDuration'] = old_row['SegmentDuration_Extreme'][0] - delta_t - 1
                        new_row['dates'] = new_row.name
                        new_row = pd.DataFrame(new_row).transpose()
                        new_row.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        for col in new_row.columns:
                            new_row[col][0] = old_row[col][0]
                        
                        timeseries['dates'] = timeseries.index.get_level_values('dates')
                        timeseries['SegmentNo'] = timeseries.index.get_level_values('SegmentNo')
                        timeseries['SegmentDuration'] = timeseries['SegmentDuration_Extreme']
                        timeseries.drop('SegmentDuration_Extreme', axis=1, inplace=True)
                        timeseries.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        timeseries = timeseries.append(new_row)
                        timeseries = timeseries.sort_values(by='dates')
                        break
    
        # add timestep if it is not already calculated
        if mini not in timeseries.index.get_level_values('dates'):
            
            # identifiy timestep, adapt it to timeseries-df and add it
            min_val = timeseries_df.loc[mini].copy()
            min_val['SegmentNo'] = len(timeseries)+1
            min_val['SegmentDuration'] = 1
            min_val['dates'] = min_val.name
            min_val = pd.DataFrame(min_val).transpose()
            min_val.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
            timeseries = timeseries.append(min_val)
            timeseries = timeseries.sort_values(by='dates')
            
            # split up segment in which the extreme timestep was added 
            i=-1
            for date in timeseries.index.get_level_values('dates'):
                if date < mini:
                    i = i+1
                else:
                    timeseries['SegmentDuration_Extreme']=timeseries.index.get_level_values('SegmentDuration')
                    old_row = timeseries.iloc[i].copy()
                    old_row = pd.DataFrame(old_row).transpose()
                    
                    delta_t = timeseries.index.get_level_values('dates')[i+1]-timeseries.index.get_level_values('dates')[i]
                    delta_t = delta_t.total_seconds()/3600
                    timeseries['SegmentDuration_Extreme'].iloc[i]=delta_t

                    timeseries_df['row_no']=range(0,len(timeseries_df))
                    new_row = int(timeseries_df.loc[mini]['row_no'])+1
                    new_date = timeseries_df[timeseries_df.row_no==new_row].index
                    
                    if new_date.isin(timeseries.index.get_level_values('dates')):
                        timeseries['dates'] = timeseries.index.get_level_values('dates')
                        timeseries['SegmentNo'] = timeseries.index.get_level_values('SegmentNo')
                        timeseries['SegmentDuration'] = timeseries['SegmentDuration_Extreme']
                        timeseries.drop('SegmentDuration_Extreme', axis=1, inplace=True)
                        timeseries.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        break
                    else: 
                        new_row = timeseries_df.iloc[new_row].copy()
                        new_row.drop('row_no', inplace=True)
                        new_row['SegmentNo'] = len(timeseries)+1
                        new_row['SegmentDuration'] = old_row['SegmentDuration_Extreme'][0] - delta_t - 1
                        new_row['dates'] = new_row.name
                        new_row = pd.DataFrame(new_row).transpose()
                        new_row.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        for col in new_row.columns:
                            new_row[col][0] = old_row[col][0]
                        timeseries['dates'] = timeseries.index.get_level_values('dates')
                        timeseries['SegmentNo'] = timeseries.index.get_level_values('SegmentNo')
                        timeseries['SegmentDuration'] = timeseries['SegmentDuration_Extreme']
                        timeseries.drop('SegmentDuration_Extreme', axis=1, inplace=True)
                        timeseries.set_index(['dates', 'SegmentNo', 'SegmentDuration'],inplace=True)
                        timeseries = timeseries.append(new_row)
                        timeseries = timeseries.sort_values(by='dates')
                    break
    
    if segmentation != True:

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
    representative_period=[]

    #cluster:medoid des jeweiligen Clusters
    dic_clusterCenterIndices = dict(enumerate(clusterCenterIndices))
    for i in clusterOrder:
        representative_period.append(dic_clusterCenterIndices[i])
        
    if how == 'hourly':
        
        hour_datetime = []
        
        for i in representative_period:
            hour_datetime.append(timeseries_df.index[i])
    
        #create a dataframe (index=nr. of day in a year/candidate)
        df_cluster =  pd.DataFrame({
                            'Cluster': clusterOrder, #Cluster of the day
                            'RepresentativeHour': representative_period, #representative day of the cluster
                            'Hour': hour_datetime})
        df_cluster.index = df_cluster.index + 1
        df_cluster.index.name = 'Candidate'
        
    else:

    	#get list of last hour of representative days
    	last_hour_datetime=[]
    	for i in representative_period:
        	last_hour = i * hours + hours - 1
        	last_hour_datetime.append(timeseries_df.index[last_hour])

    	#create a dataframe (index=nr. of day in a year/candidate)
    	df_cluster =  pd.DataFrame({
                        'Cluster': clusterOrder, #Cluster of the day
                        'RepresentativeDay': representative_period, #representative day of the cluster
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

    return df_cluster, cluster_weights, dates, hours, df_i_h, timeseries


def run(network, extremePeriodMethod='None', n_clusters=None, how='daily', segmented_to=False, csv_export=False):
    """
    """
    if segmented_to is not False:
        segment_no = segmented_to
        segmentation = True
        csv_export = csv_export
    else:
        segment_no = 24
        segmentation = False 
        
    timeseries_df_original = prepare_pypsa_timeseries(network)
    
    # calculate clusters
    df_cluster, cluster_weights, dates, hours, df_i_h, timeseries = tsam_cluster(
                timeseries_df_original,
                typical_periods = n_clusters,
                how=how,
                extremePeriodMethod = extremePeriodMethod,
                segmentation = segmentation,
                segment_no = segment_no,
                segm_hoursperperiod = network.snapshots.size)         
         
    ###### can be deleted later, just helpful to see how time steps were clustered ###
    if csv_export is not False:
        if not os.path.exists(csv_export):
            os.makedirs(csv_export, exist_ok=True)
        if segmentation != False:
            timeseries.to_csv('segmentation/timeseries_segmentation=' + str(segment_no) + '.csv') 
        else:
            if how=='daily':
                howie='days'
                path='typical_days'
            elif how=='weekly':
                howie='weeks'
                path='typical_weeks'
            elif how=='hourly':
                howie='hours'
                path='typical_hours'
            df_cluster.to_csv(path+'/cluster_typical-periods=' + str(n_clusters) + howie + '.csv')
    ########################

    network.cluster = df_cluster
    network.cluster_ts = df_i_h

    update_data_frames(network, cluster_weights, dates, hours, timeseries, segmentation, how, timeseries_df_original)

    return network


def prepare_pypsa_timeseries(network):
    """
    """
    
    loads = network.loads_t.p_set.copy()
    el_loads = network.loads[network.loads.carrier=='AC']
    el_loads = loads[list(el_loads.index)]
    loads.columns = 'L' + loads.columns
                                     
    renewables = network.generators_t.p_max_pu.mul(
                network.generators.p_nom[
                network.generators_t.p_max_pu.columns], axis = 1).copy()
    el_renewables = network.generators[network.generators.carrier == 'wind'] # TODO: für Minibsp, aber Erweiterung in etrago 
    el_renewables = renewables[list(el_renewables.index)]
    renewables.columns = 'G' + renewables.columns
    
    residual_load=pd.DataFrame()
    residual_load['residual_load']=el_loads.sum(axis=1)-el_renewables.sum(axis=1)
    df = pd.concat([renewables, loads, residual_load], axis=1)

    return df


def update_data_frames(network, cluster_weights, dates, hours, timeseries, segmentation, how, timeseries_df_original):
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
    
    if segmentation is True:
        network.snapshot_weightings = pd.Series(data = timeseries.index.get_level_values(2).values,
            index = timeseries.index.get_level_values(0))
        network.snapshots = timeseries.index.get_level_values(0)

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
            network.snapshot_weightings['objective'][i] = snapshot_weightings[i]
            network.snapshot_weightings['stores'][i] = snapshot_weightings[i]
            network.snapshot_weightings['generators'][i] = snapshot_weightings[i]

        # put the snapshot in the right order
        network.snapshots.sort_values()
        network.snapshot_weightings.sort_index()
        print(network.snapshots)
        
    return network


def skip_snapshots(self):
    n_skip = self.args['skip_snapshots']

    if n_skip:
        self.network.snapshots = self.network.snapshots[::n_skip]

        self.network.snapshot_weightings['objective'] = n_skip
        self.network.snapshot_weightings['stores'] = n_skip
        self.network.snapshot_weightings['generators'] = n_skip

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
