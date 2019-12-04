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


def snapshot_clustering(network, args, how='daily', extremePeriodMethod = 'None', clusterMethod='hierarchical'):
    """
    """

    network, df_cluster = run(network, args=args, n_clusters=args['snapshot_clustering'], 
            how=how, normed=False, extremePeriodMethod = extremePeriodMethod, clusterMethod=clusterMethod)

    return network



def tsam_cluster(timeseries_df,
                 typical_periods=10,
                 how='hourly',
                 extremePeriodMethod = 'none',
                 clusterMethod='hierarchical'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timeseries to cluster
    extremePeriodMethod: {'None','append','new_cluster_center',
                           'replace_cluster_center'}, optional, default: 'None'
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
    if how == 'weekly':
        hours = 168
    if how == 'hourly':
        hours = 1
    print(extremePeriodMethod)
    aggregation = tsam.TimeSeriesAggregation(
        timeseries_df,
        noTypicalPeriods=typical_periods,
        extremePeriodMethod = extremePeriodMethod,
        addPeakMin = ['residual_load'],
        addPeakMax = ['residual_load'],
        rescaleClusterPeriods=False,
        hoursPerPeriod=hours,
        clusterMethod=clusterMethod)
   # import pdb; pdb.set_trace()
    timeseries = aggregation.createTypicalPeriods()
    cluster_weights = aggregation.clusterPeriodNoOccur
    clusterOrder =aggregation.clusterOrder
    clusterCenterIndices= aggregation.clusterCenterIndices 
    #import pdb; pdb.set_trace()
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
    start=[]  
    # get the first hour of the clusterCenterIndices (days start with 0)
    for i in clusterCenterIndices:
        start.append(i*hours)
        
    # get a list with all hours belonging to the clusterCenterIndices
    nrhours=[]  
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

def disaggregate_soc_results(network):
    """
    Disaggregate snapshot clustered results. 
    Set soc_intra from cluster-days, soc_inter for each day and 
    soc as sum of soc_intra and soc_inter.
    """
    
    
    network.storage_units_t['state_of_charge_intra'] = pd.DataFrame(
                index = network.storage_units_t.state_of_charge.index, 
                columns = network.storage_units.index)
    network.storage_units_t['state_of_charge_inter'] = pd.DataFrame(
                index = network.storage_units_t.state_of_charge.index, 
                columns = network.storage_units.index)
        
    d = network.model.state_of_charge_intra.extract_values()
    for k in d.keys():
            network.storage_units_t['state_of_charge_intra'][k[0]][k[1]] = d[k]
    inter = network.model.state_of_charge_inter.extract_values() 
        
    for s in network.cluster.index: 
        snapshots = network.snapshots[
                network.snapshots.dayofyear-1  ==
                network.cluster['RepresentativeDay'][s]]

        day = pd.date_range(
                start = pd.to_datetime(s-1, 
                                       unit='d',
                                       origin=pd.Timestamp('2011-01-01')), 
                                       periods=24, freq = 'h')

        for su in network.storage_units.index:
            network.storage_units_t['state_of_charge_inter'][su][day] = \
                    inter[(su, s)]
            if not (day == snapshots).all():
                network.storage_units_t['state_of_charge_intra'][su][day]=\
                network.storage_units_t['state_of_charge_intra'][su][snapshots]
                    
                network.storage_units_t['state_of_charge_inter'][su][day] = \
                    inter[(su, s)]
                
                network.storage_units_t['state_of_charge'][su][day] = \
                    network.storage_units_t['state_of_charge_intra'][su][day] \
                    + network.storage_units_t['state_of_charge_inter'][su][day]

def disaggregate_hourly_soc_results(network):
    soc_all = network.model.state_of_charge_all.extract_values()
    network.storage_units_t['state_of_charge_all'] = pd.DataFrame(
                index = network.storage_units_t.state_of_charge.index, 
                columns = network.storage_units.index)
    for s in network.cluster.index-1: 
        for su in network.storage_units.index:
            network.storage_units_t['state_of_charge_all'][su][network.storage_units_t['state_of_charge_all'].index[s]] = \
                    soc_all[(su,s)]
    
def run(network, args, n_clusters=None, how='hourly',
        normed=False, extremePeriodMethod = 'None',
        clusterMethod='hierarchical'):
    """
    """
    
    if n_clusters is not None:

        # calculate clusters
        df_cluster, cluster_weights, dates, hours, df_i_h= tsam_cluster(
                prepare_pypsa_timeseries(network),
                typical_periods=n_clusters,
                how=how,
                extremePeriodMethod = extremePeriodMethod,
                clusterMethod=clusterMethod)
        network.cluster = df_cluster
        network.cluster_ts = df_i_h

        update_data_frames(network, cluster_weights, dates, hours)
#        if args['sc_settings']['constraint'] == 'daily_bounds':
#            extra_functionality = daily_bounds
#        elif args['sc_settings']['constraint'] == 'storage_soc':
#            extra_functionality = snapshot_cluster_constraints
#        elif args['sc_settings']['constraint'] == 'soc_hourly':
#            extra_functionality = houry_storage_constraints
#        else:
#            extra_functionality = None

    return network, df_cluster

def prepare_pypsa_timeseries(network, normed=False):
    """
    """

    if normed:
        normed_loads = network.loads_t.p_set / network.loads_t.p_set.max()
        normed_loads.columns = 'L' + normed_loads.columns
        normed_renewables = network.generators_t.p_max_pu
        normed_renewables.columns = 'G' + normed_renewables.columns

        sum_renew=pd.DataFrame()
        sum_renew['sum_renew']= normed_renewables.sum(axis=1)
        sum_loads=pd.DataFrame()
        sum_loads['sum_loads']= normed_loads.sum(axis=1)
        residual_load=pd.DataFrame()
        residual_load['residual_load']=normed_loads.sum(axis=1)-normed_renewables.sum(axis=1)
        df = pd.concat([normed_renewables, normed_loads, residual_load, sum_loads, sum_renew], axis=1)

    else:
        loads = network.loads_t.p_set.copy()
        loads.columns = 'L' + loads.columns
        renewables =network.generators_t.p_max_pu.mul(network.generators.p_nom[network.generators_t.p_max_pu.columns], axis = 1).copy()
        renewables.columns = 'G' + renewables.columns
        sum_renew=pd.DataFrame()
        sum_renew['sum_renew']= renewables.sum(axis=1)
        sum_loads=pd.DataFrame()
        sum_loads['sum_loads']= loads.sum(axis=1)
        residual_load=pd.DataFrame()
        residual_load['residual_load']=loads.sum(axis=1)-renewables.sum(axis=1)
        df = pd.concat([renewables, loads, residual_load, sum_loads, sum_renew], axis=1)

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
    
    #put the snapshot in the right order
    network.snapshots=network.snapshots.sort_values()
    network.snapshot_weightings=network.snapshot_weightings.sort_index()

     
    return network

def houry_storage_constraints(network, snapshots):
    network.model.del_component('state_of_charge_all')
    network.model.del_component('state_of_charge_all_index')
    network.model.del_component('state_of_charge_all_index_0')
    network.model.del_component('state_of_charge_all_index_1')
    network.model.del_component('state_of_charge_constraint')
    network.model.del_component('state_of_charge_constraint_index')
    network.model.del_component('state_of_charge_constraint_index_0')
    network.model.del_component('state_of_charge_constraint_index_1')
    
    candidates = network.cluster.index.get_level_values(0).unique()
    #import pdb; pdb.set_trace()
#    network.model.state_of_charge = po.Var(list(network.storage_units.index), snapshots,
#                                        domain=po.NonNegativeReals, bounds=(0,None))
    
    network.model.state_of_charge_all = po.Var(
            network.storage_units.index, candidates-1, 
            within=po.NonNegativeReals)
    network.model.storages = network.storage_units.index

    def set_soc_all(m,s,h):
       #import pdb; pdb.set_trace()
       if h == 0:
           prev = network.cluster.index.get_level_values(0)[-1]-1
          # import pdb; pdb.set_trace()

       else: 
           prev = h - 1

       cluster_hour = network.cluster['last_hour_RepresentativeDay'][h+1]
       
       expr = (m.state_of_charge_all[s, h] == 
                m.state_of_charge_all[s, prev] 
            * (1 - network.storage_units.at[s, 'standing_loss'])
            -(m.storage_p_dispatch[s,cluster_hour]/
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,cluster_hour]))
       return expr

    network.model.soc_all = po.Constraint(
            network.model.storages, candidates-1, rule = set_soc_all)
    
    def soc_equals_soc_all(m,s,h):
       # import pdb; pdb.set_trace()
        hour = (h.dayofyear -1)*24 + h.hour
        return (m.state_of_charge_all[s,hour] == 
                m.state_of_charge[s,h])
    
    network.model.soc_equals_soc_all = po.Constraint(
            network.model.storages, network.snapshots, 
            rule = soc_equals_soc_all)
    network.model.del_component('state_of_charge_upper')
    network.model.del_component('state_of_charge_upper_index')
    network.model.del_component('state_of_charge_upper_index_0')
    network.model.del_component('state_of_charge_upper_index_1')


    def state_of_charge_upper(m,s,h):

            
        if network.storage_units.p_nom_extendable[s]:
            p_nom = m.storage_p_nom[s]
        else:
            p_nom = network.storage_units.p_nom[s]

        return (m.state_of_charge_all[s,h] 
                    <= p_nom * network.storage_units.at[s,'max_hours']) 
              
         
    network.model.state_of_charge_upper = po.Constraint(
             network.storage_units.index,  candidates-1,
             rule = state_of_charge_upper)
    
def snapshot_cluster_constraints(network, snapshots):
    """  
    Sets snapshot cluster constraints for storage units according to :
    L. Kotzur et al: 'Time series aggregation for energy system design: 
    Modeling seasonal storage', 2018.
    
    Parameters
    -----------
    network : :class:`pypsa.Network
        Overall container of PyPSA

    snapshots: pd.DateTimeSeries
        List of snapshots 
    """
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index
    
    if True:
        candidates = network.cluster.index.get_level_values(0).unique()

        # create set for inter-temp constraints and variables
        network.model.candidates = po.Set(initialize=candidates,
                                          ordered=True)

        # create intra soc variable for each storage and each hour
        network.model.state_of_charge_intra = po.Var(
            sus.index, network.snapshots)

        def intra_soc_rule(m, s, h):
            """
            Sets soc_inter of first hour of every day to 0. Other hours are set
            by total_soc_contraint and pypsa's state_of_charge_constraint
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018, equation no. 18
            """
            if h.hour ==  0:
                expr = (m.state_of_charge_intra[s, h] == 0)
            else:
#                expr = po.Constraint.Skip
                expr = (
                    m.state_of_charge_intra[s, h] ==
                    m.state_of_charge_intra[s, h-pd.DateOffset(hours=1)] 
                    * (1 - network.storage_units.at[s, 'standing_loss'])
                    -(m.storage_p_dispatch[s,h-pd.DateOffset(hours=1)]/
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,h-pd.DateOffset(hours=1)]))
            return expr

        network.model.soc_intra_all = po.Constraint(
            network.model.storages, network.snapshots, rule = intra_soc_rule)       
        
        # create inter soc variable for each storage and each candidate
        network.model.state_of_charge_inter = po.Var(
            sus.index, network.model.candidates, 
            within=po.NonNegativeReals)

        def inter_storage_soc_rule(m, s, i):
            """
            Define the state_of_charge_inter as the state_of_charge_inter of
            the day before minus the storage losses plus the state_of_charge_intra
            of one hour after the last hour of the representative day.
            For the last reperesentive day, the soc_inter is the same as 
            the first day due to cyclic soc condition 
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018, equation no. 19
            """
            
            if i == network.model.candidates[-1]:
                last_hour = network.cluster["last_hour_RepresentativeDay"][i]
               # print(last_hour)
                expr = po.Constraint.Skip
#                expr = (
#                m.state_of_charge_inter[s, network.model.candidates[1] ] ==
#               m.state_of_charge_inter[s, i] 
#                * (1 - network.storage_units.at[s, 'standing_loss'])**24
#                + m.state_of_charge_intra[s, last_hour]\
#                        * (1 - network.storage_units.at[s, 'standing_loss'])
#                        -(m.storage_p_dispatch[s, last_hour]/\
#                        network.storage_units.at[s, 'efficiency_dispatch'] -
#                        network.storage_units.at[s, 'efficiency_store'] * 
#                        m.storage_p_store[s,last_hour]))

            else:
                last_hour = network.cluster["last_hour_RepresentativeDay"][i+1]#-pd.DateOffset(hours=23)
                last_hour_prev = network.cluster["last_hour_RepresentativeDay"][i]
                expr = (
                m.state_of_charge_inter[s, i+1 ] ==
                m.state_of_charge_inter[s, i] 
               # * (1 - network.storage_units.at[s, 'standing_loss'])**24
                + m.state_of_charge_intra[s, last_hour_prev]\
                - m.storage_p_dispatch[s, last_hour_prev]\
                + m.storage_p_store[s,last_hour_prev])
                """* (1 - network.storage_units.at[s, 'standing_loss'])\
                        -(m.storage_p_dispatch[s, last_hour_prev]/\
                        network.storage_units.at[s, 'efficiency_dispatch'] -
                        network.storage_units.at[s, 'efficiency_store'] * 
                        m.storage_p_store[s,last_hour_prev]))"""
        
            return expr

        network.model.inter_storage_soc_constraint = po.Constraint(
            sus.index, network.model.candidates,
            rule=inter_storage_soc_rule)

                #new definition of the state_of_charge used in pypsa
        network.model.del_component('state_of_charge_constraint')
        network.model.del_component('state_of_charge_constraint_index')
        network.model.del_component('state_of_charge_constraint_index_0')
        network.model.del_component('state_of_charge_constraint_index_1')
        
        def total_state_of_charge(m,s,h):
            """
            Define the state_of_charge as the sum of state_of_charge_inter 
            and state_of_charge_intra
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018
            """

            return(m.state_of_charge[s,h] ==
                   m.state_of_charge_intra[s,h] + m.state_of_charge_inter[
                           s,network.cluster_ts['Candidate_day'][h]])                

        network.model.total_storage_constraint = po.Constraint(
                sus.index, network.snapshots, rule = total_state_of_charge)

        def state_of_charge_lower(m,s,h):
            """
            Define the state_of_charge as the sum of state_of_charge_inter 
            and state_of_charge_intra
            
            According to:
            L. Kotzur et al: 'Time series aggregation for energy system design: 
            Modeling seasonal storage', 2018
            """
             
          #  import pdb; pdb.set_trace()
          # Choose datetime of representive day
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]

            hour = str(h).split(' ')[1]
            
            intra_hour = pd.to_datetime(date + ' ' + hour)

            return(m.state_of_charge_intra[s,intra_hour] + 
                   m.state_of_charge_inter[s,network.cluster_ts['Candidate_day'][h]]
                  # * (1 - network.storage_units.at[s, 'standing_loss'])**24
                   >= 0)                

        network.model.state_of_charge_lower = po.Constraint(
                sus.index, network.cluster_ts.index, rule = state_of_charge_lower)
        
        
        network.model.del_component('state_of_charge_upper')
        network.model.del_component('state_of_charge_upper_index')
        network.model.del_component('state_of_charge_upper_index_0')
        network.model.del_component('state_of_charge_upper_index_1')


        def state_of_charge_upper(m,s,h):
            date = str(network.snapshots[
                network.snapshots.dayofyear -1 ==
                network.cluster['RepresentativeDay'][h.dayofyear]][0]).split(' ')[0]
            
            hour = str(h).split(' ')[1]
            
            intra_hour = pd.to_datetime(date + ' ' + hour)
            
            if network.storage_units.p_nom_extendable[s]:
                p_nom = m.storage_p_nom[s]
            else:
                p_nom = network.storage_units.p_nom[s]

            return (m.state_of_charge_intra[s,intra_hour] + 
                    m.state_of_charge_inter[s,network.cluster_ts['Candidate_day'][h]] 
                   # * (1 - network.storage_units.at[s, 'standing_loss'])**24
                    <= p_nom * network.storage_units.at[s,'max_hours']) 
              
         
        network.model.state_of_charge_upper = po.Constraint(
             sus.index, network.cluster_ts.index,
             rule = state_of_charge_upper)


        def cyclic_state_of_charge(m,s):
            """
            Defines cyclic condition like pypsas 'state_of_charge_contraint'.
            There are small differences to original results.
            """
            last_day = network.cluster.index[-1]
            
            last_calc_hour = network.cluster['last_hour_RepresentativeDay'][last_day]
            
            last_inter = m.state_of_charge_inter[s, last_day]
            
            last_intra = m.state_of_charge_intra[s, last_calc_hour]
            
            first_day =  network.cluster.index[0]
            
            first_calc_hour = network.cluster['last_hour_RepresentativeDay'][first_day] - pd.DateOffset(hours=23)
            
            first_inter = m.state_of_charge_inter[s, first_day]
            
            first_intra = m.state_of_charge_intra[s, first_calc_hour]

            return  (first_intra + first_inter == \
                   ((last_intra + last_inter)
                   * (1 - network.storage_units.at[s, 'standing_loss'])
                   -m.storage_p_dispatch[s,last_calc_hour]/ 
                           network.storage_units.at[s, 'efficiency_dispatch']
                   +m.storage_p_store[s,last_calc_hour] * 
                           network.storage_units.at[s, 'efficiency_store'])) 

        network.model.cyclic_storage_constraint = po.Constraint(
                sus.index,  rule = cyclic_state_of_charge)
        
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
