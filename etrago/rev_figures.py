import json
import pypsa
from etrago import Etrago

import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

plt.rc('font', size=20)


# modularity

mod = pd.read_csv("Calcs/modularity.csv")
mod.rename(columns={'Unnamed: 0':"number of nodes"}, inplace= True)
mod.set_index("number of nodes", inplace = True)
mod.drop(columns=["kmedoids-dijkstra_x"], inplace= True)

mod.at[400, "k-means Clustering"] = 0.825
mod.at[500, "k-means Clustering"] = 0.802
mod.at[600, "k-means Clustering"] = 0.798

mod.at[400, "k-medoids Dijkstra Clustering"] = 0.87
mod.at[500, "k-medoids Dijkstra Clustering"] = 0.856
mod.at[600, "k-medoids Dijkstra Clustering"] = 0.847

mod.plot(ylabel="modularity")


# optimization time 

mod = pd.read_csv("Calcs/calc-time.csv")
mod['number of nodes'] = mod["spatial resolution"]
mod.drop("spatial resolution", axis=1, inplace=True)
mod.set_index("number of nodes", inplace = True)

mod.plot(ylabel="optimization time in minutes")


# plot system costs as bar diagram - NEW


# export

def dc_export(network):
    
    de_buses = network.buses[network.buses.country == "DE"]
    for_buses = network.buses[network.buses.country != "DE"]
    exp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus0.isin(de_buses.index))
        & (network.links.bus1.isin(for_buses.index))
    ]
    imp = network.links[
        (network.links.carrier == "DC")
        & (network.links.bus1.isin(de_buses.index))
        & (network.links.bus0.isin(for_buses.index))
    ]
    return (
        network.links_t.p0[exp.index].sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
        + network.links_t.p1[imp.index].sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
    )

# investment costs in DE

def investment_Germany(network, sto=True):
    
    de_buses = network.buses[network.buses.country == "DE"]
    
    de_lines = network.lines[(network.lines.bus0.isin(de_buses.index))&
                             (network.lines.bus1.isin(de_buses.index))&
                             network.lines.s_nom_extendable==True]
    
    lines = ((de_lines.s_nom_opt-de_lines.s_nom_min
                         )*de_lines.capital_cost).sum()
    
    if sto:
    
        de_sto = network.storage_units[(network.storage_units.carrier== 'battery')&
                                       network.storage_units.bus.isin(de_buses.index)]
        
        sto = ((de_sto.p_nom_opt-de_sto.p_nom_min)*de_sto.capital_cost).sum()
    
    else:
        sto= 0
    
    return lines + sto

def foreign_sto_expansion(network):
    
    bus = network.buses[network.buses.country!='DE']
    
    sto = network.stores[network.stores.bus.isin(bus.index)]
    sto_u = network.storage_units[network.storage_units.bus.isin(bus.index)]
    
    sto = ((sto.e_nom_opt -
            sto.e_nom_min
            )[sto.e_nom_extendable]*sto.capital_cost).sum()
    
    sto_u = ((sto_u.p_nom_opt - sto_u.p_nom_min
            )[sto_u.p_nom_extendable]\
                *sto_u.capital_cost).sum()
                
    return sto + sto_u


spatial_resolution = [50, 100, 150, 200, 250, 300, 400, 500, 600] #20

spatial_method = ['kmeans', 'kmedoids-dijkstra']

# dataframe for system costs = marginal + investment costs
costs1 = pd.DataFrame(columns=['marginal costs', 'battery investment costs', 'electrical grid investment costs'])# 'electrical grid investment costs in Germany', 'electrical grid investment costs in foreign countries'], index=spatial_resolution)
costs2 = costs1.copy()

# dataframe for investment costs = store + grid investment in Germany and foreign countries
inv1 = pd.DataFrame(columns=['store investment costs in foreign countries', 'electrical grid investment costs in foreign countries', 'store investment costs in Germany', 'electrical grid investment costs in Germany'])
inv2 = inv1.copy()

# Germany
inv1_DE = pd.DataFrame(columns=['store investment costs', 'electrical grid investment costs'])
inv2_DE = inv1_DE.copy()

# to plot some relative costs 
# related to the result of the 500 nodes k-medoids Dijkstra clustered network

etrago = Etrago(csv_folder_name="Calcs/kmedoids-dijkstra/500")
etrago.calc_results()

relac_expansion =  pd.DataFrame(index=spatial_resolution, columns=spatial_method)
base_relac = etrago.results.loc['annual ac grid investment costs'].value

new_relac = pd.Series(index=spatial_resolution)
new_ac = pd.Series(index=spatial_resolution)
ac_mean = pd.Series(index=spatial_resolution)
new_reldc = pd.Series(index=spatial_resolution)
new_relsto = pd.Series(index=spatial_resolution)
new_relmarg = pd.Series(index=spatial_resolution)
new_relsys = pd.Series(index=spatial_resolution)

reldc_expansion =  pd.DataFrame(index=spatial_resolution, columns=spatial_method)
base_reldc = etrago.results.loc['annual dc grid investment costs'].value

relsto_expansion =  pd.DataFrame(index=spatial_resolution, columns=spatial_method)
base_relsto = etrago.results.loc['annual storage+store investment costs'].value

rel_marginal =  pd.DataFrame(index=spatial_resolution, columns=spatial_method)
base_relmar = etrago.results.loc['annual marginal costs'].value

rel_objective = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
base_objective = etrago.results.loc['annual system costs'].value

for i in range (0, len(spatial_method)):

    if spatial_method[i] == 'kmeans':
        costs = costs1
        inv = inv1
        inv_DE = inv1_DE
    else:
        costs = costs2
        inv = inv2
        inv_DE = inv2_DE
        
    col = spatial_method[i]

    for j in range(0, len(spatial_resolution)):
        
            idx = spatial_resolution[j]
    
            etrago = Etrago(csv_folder_name="Calcs/"+col+"/"+str(idx))
            
            etrago.calc_results()
            
            costs.at[idx, 'marginal costs'] = etrago.results.loc['annual marginal costs'].value
            costs.at[idx, 'electrical grid investment costs'] = etrago.results.loc['annual electrical grid investment costs'].value
            #costs.at[idx, 'electrical grid investment costs in Germany'] = etrago.results.loc['annual ac grid investment costs'].value
            #costs.at[idx, 'electrical grid investment costs in foreign countries'] = etrago.results.loc['annual ac grid investment costs'].value
            costs.at[idx, 'battery investment costs'] = etrago.results.loc['annual storage+store investment costs'].value
            
            foreign_store = foreign_sto_expansion(etrago.network)
            inv.at[idx, 'store investment costs in foreign countries'] = foreign_store
            inv.at[idx, 'electrical grid investment costs in foreign countries'] = etrago.results.loc['annual dc grid investment costs'].value
            inv.at[idx, 'store investment costs in Germany'] = etrago.results.loc['annual storage+store investment costs'].value - foreign_store
            inv.at[idx,'electrical grid investment costs in Germany'] = etrago.results.loc['annual ac grid investment costs'].value

            inv_DE.at[idx, 'store investment costs'] = etrago.results.loc['annual storage+store investment costs'].value - foreign_store
            inv_DE.at[idx,'electrical grid investment costs'] = etrago.results.loc['annual ac grid investment costs'].value

            relac_expansion.at[idx, col] = etrago.results.loc['annual ac grid investment costs'].value / base_relac
            reldc_expansion.at[idx, col] = etrago.results.loc['annual dc grid investment costs'].value / base_reldc
            relsto_expansion.at[idx, col] = etrago.results.loc['annual storage+store investment costs'].value / base_relsto
            rel_marginal.at[idx, col] = etrago.results.loc['annual marginal costs'].value / base_relmar
            rel_objective.at[idx, col] = etrago.results.loc['annual system costs'].value / base_objective
            
            if col=='kmedoids-dijkstra':

                base = Etrago(csv_folder_name="Calcs/kmeans/"+str(idx))
                base.calc_results()
                
                base_ac = base.results.loc['annual ac grid investment costs'].value
                new_relac.loc[idx] = ((etrago.results.loc['annual ac grid investment costs'].value / base_ac)*100) -100
                new_ac.loc[idx] = etrago.results.loc['annual ac grid investment costs'].value
                ac_mean.loc[idx] = base_ac 
                
                base_dc = base.results.loc['annual dc grid investment costs'].value
                new_reldc.loc[idx] = ((etrago.results.loc['annual dc grid investment costs'].value / base_dc)*100) -100
                
                base_sto = base.results.loc['annual storage+store investment costs'].value
                new_relsto.loc[idx] = ((etrago.results.loc['annual storage+store investment costs'].value / base_sto)*100) -100
                
                base_marg = base.results.loc['annual marginal costs'].value
                new_relmarg.loc[idx] = ((etrago.results.loc['annual marginal costs'].value / base_marg)*100) -100
                
                base_sys = base.results.loc['annual system costs'].value
                new_relsys.loc[idx] = ((etrago.results.loc['annual system costs'].value / base_sys)*100) -100
                

costs1 = costs1 / 1000000000
costs2 = costs2 / 1000000000

fig = plt.figure()#figsize=(30,20))
ax = fig.add_subplot(111)

index1 = spatial_resolution.copy()
index2 = spatial_resolution.copy()

for i in range(0, len(index1)):
    index1[i] = index1[i] - 5
    index2[i] = index2[i] + 5

bottom1 = [0, 0, 0, 0, 0, 0, 0, 0, 0] #0
bottom2=bottom1.copy()
i = -1

#color1 = ['darkslategrey', 'darkcyan', 'skyblue']#, 'skyblue'] # kmeans
#color2 = ['maroon', 'tomato', 'lightsalmon']#, 'lightsalmon'] # kmedoids-djkstra
color1 = ['skyblue', 'darkcyan', 'darkslategrey']
color2 = ['lightsalmon', 'tomato', 'maroon']

#hatch = [False, False, False, "/"]

for col in costs1.columns:
    
    if i >= 0:
        bottom1 = bottom1 + costs1[costs1.columns[i]].values
        bottom2 = bottom2 + costs2[costs2.columns[i]].values
    i = i+1
    
    #if hatch[i]:
    ax.bar(index1, costs1[col].values, width = 4, bottom = bottom1, color=color1[i], label='k-means Clustering: '+col) #, hatch=hatch[i]
    ax.bar(index2, costs2[col].values, width = 4, bottom = bottom2, color=color2[i], label='k-medoids Dijkstra Clustering: '+col)
    #else:   
        #plt.bar(index1, costs1[col].values, width = 4, bottom = bottom1, color=color1[i], label='k-means Clustering: '+col)
        #plt.bar(index2, costs2[col].values, width = 4, bottom = bottom2, color=color2[i], label='k-medoids Dijkstra Clustering: '+col)

#plt.xticks(spatial_resolution)
#plt.legend(loc='lower right')
#plt.ylabel('costs in billion Euro')
#plt.xlabel('number of nodes')
#plt.title('System Costs depending on Spatial Resolution')

ax2 = ax.twinx()

new_relsys.plot(linestyle='-', marker='x', ax = ax2, label='system costs')
new_relmarg.plot(linestyle='-', marker='x', ax = ax2, label='operational costs')
new_reldc.plot(linestyle='-', marker='x', ax = ax2, label='foreign grid expansion costs')
new_relac.plot(linestyle='-', marker='x', ax = ax2, label='inner-German grid expansion costs')
new_relsto.plot(linestyle='-', marker='x', ax = ax2, label='battery expansion costs')

ax.set_xticks(spatial_resolution)

ax.set_ylim([0,40])
ax2.set_ylim([-10, 400])

ax.legend(loc='upper left', fontsize=20)
ax2.legend(loc='upper right', fontsize=20)

ax.set_ylabel('costs in billion Euro')
ax2.set_ylabel('rel. difference in %')
ax.set_xlabel('number of nodes')


# network and storage expansion - label storage capacity in k-medoids Dijkstra Clustering

etrago = Etrago(csv_folder_name="Calcs/kmedoids-dijkstra/300")

etrago.plot_grid(line_colors='expansion_abs', bus_colors='storage_expansion', bus_sizes=0.000001, ext_min=0.01, legend_entries=[], boundaries=[0, 20000])

etrago = Etrago(csv_folder_name="Calcs/kmeans/300")

etrago.plot_grid(line_colors='expansion_abs', bus_colors='storage_expansion', bus_sizes=0.000001, ext_min=0.01, legend_entries=[], boundaries=[0, 20000])