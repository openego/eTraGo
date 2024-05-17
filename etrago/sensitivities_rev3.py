import json
import pypsa
from etrago import Etrago

import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

###############################################################################

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

def TVA_km(network):
    
    de_buses = network.buses[network.buses.country == "DE"]
    de_lines = network.lines[(network.lines.bus0.isin(de_buses.index))&(network.lines.bus1.isin(de_buses.index))&network.lines.s_nom_extendable==True]
    
    lines = (((de_lines.s_nom_opt-de_lines.s_nom_min) /1000000) * de_lines.length).sum()
    
    return lines

spatial_method = ['kmeans', 'kmedoids-dijkstra']

# spatial_resolution = [300, 300, 300, 300, 300, 300, 300]

percentage = [10, 25, 50, 100, 150, 175, 200]

##############################lines############################################

costs_kmeans = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
costs_dijkstra = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
rel_kmeans = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
rel_dijkstra = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
load_shedding = pd.DataFrame(index=percentage, columns=spatial_method)

for i in range (0, len(spatial_method)):

    if spatial_method[i] == 'kmeans':
        costs = costs_kmeans
        rel = rel_kmeans
        base = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_lines/300_100")

    else:
        costs = costs_dijkstra
        rel = rel_dijkstra
        base = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_lines/300_100")
        
    base.calc_results()
    foreign_store_base = foreign_sto_expansion(base.network)
    inner_store_base = base.results.loc['annual storage+store investment costs'].value - foreign_store_base

    for j in range(0, len(percentage)):
        
        etrago = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_lines/300_"+str(percentage[j]))
        
        etrago.network.lines.capital_cost = etrago.network.lines.capital_cost * (100/percentage[j])
        
        loads = etrago.network.generators[etrago.network.generators.carrier=='load shedding']
        load_shedding.at[percentage[j], spatial_method[i]] = etrago.network.generators_t.p.transpose().loc[loads.index].sum().sum()
        load_shedding_lines = load_shedding.copy()
        
        etrago.calc_results()
        
        costs.at[percentage[j], 'system costs'] = etrago.results.loc['annual system costs'].value
        costs.at[percentage[j], 'marginal costs'] = etrago.results.loc['annual marginal costs'].value
        costs.at[percentage[j], 'battery investment costs'] = etrago.results.loc['annual electrical storage investment costs'].value
        costs.at[percentage[j], 'electrical grid investment costs'] = etrago.results.loc['annual electrical grid investment costs'].value
        foreign_store = foreign_sto_expansion(etrago.network)
        costs.at[percentage[j], 'inner-German battery investment costs'] = etrago.results.loc['annual storage+store investment costs'].value - foreign_store
        costs.at[percentage[j], 'inner-German grid investment costs'] = etrago.results.loc['annual ac grid investment costs'].value
        costs.at[percentage[j], 'inner-German grid expansion (TW*km)'] = TVA_km(etrago.network)
        
        rel.at[percentage[j], 'system costs'] = ((base.results.loc['annual system costs'].value - etrago.results.loc['annual system costs'].value) / base.results.loc['annual system costs'].value)*100
        rel.at[percentage[j], 'marginal costs'] = ((base.results.loc['annual marginal costs'].value - etrago.results.loc['annual marginal costs'].value) / base.results.loc['annual marginal costs'].value)*100
        rel.at[percentage[j], 'battery investment costs'] = ((base.results.loc['annual electrical storage investment costs'].value - etrago.results.loc['annual electrical storage investment costs'].value) / base.results.loc['annual electrical storage investment costs'].value) *100
        rel.at[percentage[j], 'electrical grid investment costs'] = ((base.results.loc['annual electrical grid investment costs'].value - etrago.results.loc['annual electrical grid investment costs'].value) / base.results.loc['annual electrical grid investment costs'].value)*100
        rel.at[percentage[j], 'inner-German battery investment costs'] = ((inner_store_base - costs.loc[percentage[j], 'inner-German battery investment costs'] ) / inner_store_base) *100
        rel.at[percentage[j], 'inner-German grid investment costs'] = ((base.results.loc['annual ac grid investment costs'].value - etrago.results.loc['annual ac grid investment costs'].value) / base.results.loc['annual ac grid investment costs'].value) *100
        rel.at[percentage[j], 'inner-German grid expansion (TW*km)'] = ((TVA_km(base.network)-TVA_km(etrago.network))/TVA_km(base.network))*100
        
rel_diff = ((costs_dijkstra - costs_kmeans) / costs_dijkstra) *100
        
plot_abs = pd.DataFrame(index=percentage, columns=['k-medoids Dijkstra Clustering', 'k-means Clustering'])
plot_abs['k-medoids Dijkstra Clustering'] = costs_dijkstra['inner-German grid investment costs']
plot_abs['k-means Clustering'] = costs_kmeans['inner-German grid investment costs']
#plot_abs.plot()
plot_abs_lines = plot_abs.copy()

plot_exp = pd.DataFrame(index=percentage, columns=['k-medoids Dijkstra Clustering', 'k-means Clustering'])
plot_exp['k-medoids Dijkstra Clustering'] = costs_dijkstra['inner-German grid expansion (TW*km)']
plot_exp['k-means Clustering'] = costs_kmeans['inner-German grid expansion (TW*km)']
#plot_exp.plot()
plot_lines = plot_exp.copy()

plot_expdiff = pd.DataFrame(index=percentage, data=rel_diff['inner-German grid expansion (TW*km)'].values)
# plot_expdiff.plot()

##############################batteries#########################################

costs_kmeans = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
costs_dijkstra = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
rel_kmeans = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
rel_dijkstra = pd.DataFrame(index=percentage, columns=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])
load_shedding = pd.DataFrame(index=percentage, columns=spatial_method)

for i in range (0, len(spatial_method)):

    if spatial_method[i] == 'kmeans':
        costs = costs_kmeans
        rel = rel_kmeans
        base = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_sto/300_100")

    else:
        costs = costs_dijkstra
        rel = rel_dijkstra
        base = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_sto/300_100")
        
    base.calc_results()
    foreign_store_base = foreign_sto_expansion(base.network)
    inner_store_base = base.results.loc['annual storage+store investment costs'].value - foreign_store_base

    for j in range(0, len(percentage)):
        
        etrago = Etrago(csv_folder_name="Calcs/"+str(spatial_method[i])+"_sto/300_"+str(percentage[j]))
        
        loads = etrago.network.generators[etrago.network.generators.carrier=='load shedding']
        load_shedding.at[percentage[j], spatial_method[i]] = etrago.network.generators_t.p.transpose().loc[loads.index].sum().sum()
        
        etrago.calc_results()
        
        costs.at[percentage[j], 'system costs'] = etrago.results.loc['annual system costs'].value
        costs.at[percentage[j], 'marginal costs'] = etrago.results.loc['annual marginal costs'].value
        costs.at[percentage[j], 'battery investment costs'] = etrago.results.loc['annual electrical storage investment costs'].value
        costs.at[percentage[j], 'electrical grid investment costs'] = etrago.results.loc['annual electrical grid investment costs'].value
        foreign_store = foreign_sto_expansion(etrago.network)
        costs.at[percentage[j], 'inner-German battery investment costs'] = etrago.results.loc['annual storage+store investment costs'].value - foreign_store
        costs.at[percentage[j], 'inner-German grid investment costs'] = etrago.results.loc['annual ac grid investment costs'].value
        costs.at[percentage[j], 'inner-German grid expansion (TW*km)'] = TVA_km(etrago.network)
        
        rel.at[percentage[j], 'system costs'] = ((base.results.loc['annual system costs'].value - etrago.results.loc['annual system costs'].value) / base.results.loc['annual system costs'].value)*100
        rel.at[percentage[j], 'marginal costs'] = ((base.results.loc['annual marginal costs'].value - etrago.results.loc['annual marginal costs'].value) / base.results.loc['annual marginal costs'].value)*100
        rel.at[percentage[j], 'battery investment costs'] = ((base.results.loc['annual electrical storage investment costs'].value - etrago.results.loc['annual electrical storage investment costs'].value) / base.results.loc['annual electrical storage investment costs'].value) *100
        rel.at[percentage[j], 'electrical grid investment costs'] = ((base.results.loc['annual electrical grid investment costs'].value - etrago.results.loc['annual electrical grid investment costs'].value) / base.results.loc['annual electrical grid investment costs'].value)*100
        rel.at[percentage[j], 'inner-German battery investment costs'] = ((inner_store_base - costs.loc[percentage[j], 'inner-German battery investment costs'] ) / inner_store_base) *100
        rel.at[percentage[j], 'inner-German grid investment costs'] = ((base.results.loc['annual ac grid investment costs'].value - etrago.results.loc['annual ac grid investment costs'].value) / base.results.loc['annual ac grid investment costs'].value) *100
        rel.at[percentage[j], 'inner-German grid expansion (TW*km)'] = ((TVA_km(base.network)-TVA_km(etrago.network))/TVA_km(base.network))*100
        
rel_diff_sto = ((costs_dijkstra - costs_kmeans) / costs_dijkstra) *100
        
plot_abs = pd.DataFrame(index=percentage, columns=['k-medoids Dijkstra Clustering', 'k-means Clustering'])
plot_abs['k-medoids Dijkstra Clustering'] = costs_dijkstra['inner-German grid investment costs']
plot_abs['k-means Clustering'] = costs_kmeans['inner-German grid investment costs']
#plot_abs.plot()
plot_abs_sto = plot_abs.copy()

plot_exp = pd.DataFrame(index=percentage, columns=['k-medoids Dijkstra Clustering', 'k-means Clustering'])
plot_exp['k-medoids Dijkstra Clustering'] = costs_dijkstra['inner-German grid expansion (TW*km)']
plot_exp['k-means Clustering'] = costs_kmeans['inner-German grid expansion (TW*km)']
#plot_exp.plot()
plot_sto = plot_exp.copy()

plot_expdiff = pd.DataFrame(index=percentage, data=rel_diff['inner-German grid expansion (TW*km)'].values)
# plot_expdiff.plot()

##############################resolution#######################################

spatial_resolution = [50, 100, 150, 200, 250, 300, 400, 500, 600]

spatial_method = ['kmeans', 'kmedoids-dijkstra']

df = pd.DataFrame(index=spatial_resolution, columns=['k-medoids Dijkstra Clustering', 'k-means Clustering', 'rel. difference'])
costs = df.copy()

for i in range (0, len(spatial_method)):

    fold = spatial_method[i]
    
    if spatial_method[i] == 'kmeans':
        col = 'k-means Clustering'
    else:
        col = 'k-medoids Dijkstra Clustering'

    for j in range(0, len(spatial_resolution)):
        
            idx = spatial_resolution[j]

            etrago = Etrago(csv_folder_name="Calcs/"+fold+"/"+str(idx))
            etrago.calc_results()

            df.at[idx, col] = TVA_km(etrago.network)
            costs.at[idx, col] = etrago.results.loc['annual ac grid investment costs'].value

df['rel. difference'] = ((df['k-medoids Dijkstra Clustering'] - df['k-means Clustering']) / df['k-medoids Dijkstra Clustering'])*100
costs['rel. difference'] = ((costs['k-medoids Dijkstra Clustering'] - costs['k-means Clustering']) / costs['k-medoids Dijkstra Clustering'])*100

###############################################################################

# sensitivity analysis as one figure for paper

plt.rc('font', size=22)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

index1 = spatial_resolution.copy()
index11 = spatial_resolution.copy()
index12 = spatial_resolution.copy()

per = percentage.copy()
for i in range(0,len(percentage)):
    per[i] = percentage[i]/100
    
plot_lines.index = per.copy()
plot_sto.index = per.copy()

index2 = per.copy()
index21 = per.copy()
index22 = per.copy()


for i in range(0, len(index1)):
    index11[i] = index1[i] - 5
    index12[i] = index1[i] + 5
    
for i in range (0, len(index2)):
    index21[i] = index2[i] - 0.02
    index22[i] = index2[i] + 0.02

ax12 = ax1.twinx()
ax1.bar(index11, df['k-medoids Dijkstra Clustering'], label='k-medoids Dijkstra Clustering', width=5)
ax1.bar(index12, df['k-means Clustering'], label='k-means Clustering', width=5)
ax12.plot(df.index, df['rel. difference'], color='darkslategrey', label='rel. difference')

ax1.set_xticks([100, 200, 300, 400, 500, 600])
ax1.set_xlabel('number of nodes')
ax12.set_ylabel('rel. difference in %', size = 25)
ax1.legend(loc='upper left', fontsize=22)
ax12.legend(loc='upper right', fontsize=22)
ax1.set_ylim([0,15])
ax12.set_ylim([0,100])

ax22 = ax2.twinx()
ax2.bar(index21, plot_lines['k-medoids Dijkstra Clustering'], width=0.02, label='k-medoids Dijkstra Clustering')
ax2.bar(index22, plot_lines['k-means Clustering'], width=0.02, label='k-means Clustering')
ax22.plot(plot_lines.index, rel_diff['inner-German grid expansion (TW*km)'],color='darkslategrey', label='rel. difference')

ax2.set_xticks([0.1, 0.25, 0.5, 1, 1.5, 1.75, 2])
ax2.set_xlabel('share of original capital costs for lines')
ax2.set_ylabel('inner-German grid expansion in TVA*km', size = 25)
#ax22.set_ylabel('rel. difference in %', size = 25)
ax2.set_ylim([0,15])
ax22.set_ylim([0,50])

ax32 = ax3.twinx()
ax3.bar(index21, plot_sto['k-medoids Dijkstra Clustering'], width=0.02, label='k-medoids Dijkstra Clustering')
ax3.bar(index22, plot_sto['k-means Clustering'], width=0.02, label='k-means Clustering')
ax32.plot(plot_lines.index, rel_diff_sto['inner-German grid expansion (TW*km)'], color='darkslategrey', label='rel. difference')

ax3.set_xticks([0.1, 0.25, 0.5, 1, 1.5, 1.75, 2])
ax3.set_xlabel('share of original capital costs for batteries')
#ax3.set_ylabel('inner-German grid expansion in TVA*km')
#ax32.set_ylabel('rel. difference in %')
ax3.legend(loc='upper left', fontsize=22)
ax32.legend(loc='upper right', fontsize=22)
ax3.set_ylim([0,15])
ax32.set_ylim([0,50])

plt.subplots_adjust(hspace=0.3)

# sensitivity analysis as two figures for paper

# lines and storage units

plot_abs_lines = plot_abs_lines / 1000000
plot_abs_sto = plot_abs_sto / 1000000

plt.rc('font', size=22)

fig, (ax2, ax3) = plt.subplots(2, 1, sharey=True)

index1 = spatial_resolution.copy()
index11 = spatial_resolution.copy()
index12 = spatial_resolution.copy()

per = percentage.copy()
for i in range(0,len(percentage)):
    per[i] = percentage[i]/100
    
plot_lines.index = per.copy()
plot_sto.index = per.copy()

index2 = per.copy()
index21 = per.copy()
index22 = per.copy()


for i in range(0, len(index1)):
    index11[i] = index1[i] - 5
    index12[i] = index1[i] + 5
    
for i in range (0, len(index2)):
    index21[i] = index2[i] - 0.02
    index22[i] = index2[i] + 0.02

ax22 = ax2.twinx()
ax2.bar(index21, plot_abs_lines['k-medoids Dijkstra Clustering'], width=0.02, label='k-medoids Dijkstra Clustering')
ax2.bar(index22, plot_abs_lines['k-means Clustering'], width=0.02, label='k-means Clustering')
# ax22.plot(plot_lines.index, rel_diff['inner-German grid expansion (TW*km)'],color='darkslategrey', label='rel. difference')
ax22.plot(plot_lines.index, rel_diff['inner-German grid investment costs'],color='darkslategrey', label='rel. difference', linestyle='-', marker='x',)

ax2.set_xticks([0.1, 0.25, 0.5, 1, 1.5, 1.75, 2])
ax2.set_xlabel('share of the base specific capital costs of lines in p.u.')
#ax2.set_ylim([0,15])
ax22.set_ylim([0,50])

ax32 = ax3.twinx()
ax3.bar(index21, plot_abs_sto['k-medoids Dijkstra Clustering'], width=0.02, label='k-medoids Dijkstra Clustering')
ax3.bar(index22, plot_abs_sto['k-means Clustering'], width=0.02, label='k-means Clustering')
#ax32.plot(plot_lines.index, rel_diff_sto['inner-German grid expansion (TW*km)'], color='darkslategrey', label='rel. difference')
ax32.plot(plot_lines.index, rel_diff_sto['inner-German grid investment costs'],color='darkslategrey', label='rel. difference', linestyle='-', marker='x',)

ax3.set_xticks([0.1, 0.25, 0.5, 1, 1.5, 1.75, 2])
ax3.set_xlabel('share of the base specific capital costs of batteries in p.u.')
#ax3.set_ylim([0,15])
ax32.set_ylim([0,50])

#ax3.legend(loc='upper left')
#ax32.legend(loc='upper right')
ax3.legend(bbox_to_anchor=(0.3, 2.6))
ax32.legend(bbox_to_anchor=(1, 2.5))

fig.text(0.09, 0.22, 'inner-German grid investments in million €', ha='center', rotation='vertical')
fig.text(0.93, 0.38, 'rel. difference in %', ha='center',  rotation=90)

#plt.ylabel('inner-German grid expansion in TVA*km')
#ax32.set_ylabel('rel. difference in %')

plt.subplots_adjust(hspace=0.3)

# resolution

plt.rcParams["figure.figsize"] = [17,6]
plt.rc('font', size=25)

fig, ax1 = plt.subplots(1, 1)

index1 = spatial_resolution.copy()
index11 = spatial_resolution.copy()
index12 = spatial_resolution.copy()

per = percentage.copy()
for i in range(0,len(percentage)):
    per[i] = percentage[i]/100
    
plot_lines.index = per.copy()
plot_sto.index = per.copy()

index2 = per.copy()
index21 = per.copy()
index22 = per.copy()


for i in range(0, len(index1)):
    index11[i] = index1[i] - 5
    index12[i] = index1[i] + 5
    
for i in range (0, len(index2)):
    index21[i] = index2[i] - 0.02
    index22[i] = index2[i] + 0.02

ax12 = ax1.twinx()
ax1.bar(index11, df['k-medoids Dijkstra Clustering'], label='k-medoids Dijkstra Clustering', width=5)
ax1.bar(index12, df['k-means Clustering'], label='k-means Clustering', width=5)
ax12.plot(df.index, df['rel. difference'], color='darkslategrey', label='rel. difference')

ax1.set_xticks([100, 200, 300, 400, 500, 600])
ax1.set_xlabel('number of nodes')
ax1.set_ylabel('inner-German grid expansion in TVA*km')
ax12.set_ylabel('rel. difference in %')
ax1.legend(loc='upper left')
ax12.legend(loc='upper right')
#ax1.set_ylim([0,15])
ax12.set_ylim([0,100])

###############################################################################

# ehv clustering

kmeans = Etrago(csv_folder_name="Calcs/kmeans_ehv/300")
kmeans.calc_results()

dijkstra = Etrago(csv_folder_name="Calcs/kmedoids-dijkstra_ehv/300")
dijkstra.calc_results()

df= pd.DataFrame(columns=['k-means Clustering', 'k-medoids Dijkstra Clustering'], index=['system costs', 'marginal costs', 'battery investment costs', 'electrical grid investment costs', 'inner-German battery investment costs', 'inner-German grid investment costs'])

df.at['system costs', 'k-means Clustering'] = kmeans.results.loc['annual system costs'].value
df.at['marginal costs', 'k-means Clustering'] = kmeans.results.loc['annual marginal costs'].value
df.at['battery investment costs', 'k-means Clustering'] = kmeans.results.loc['annual electrical storage investment costs'].value
df.at['electrical grid investment costs', 'k-means Clustering'] = kmeans.results.loc['annual electrical grid investment costs'].value
foreign_store = foreign_sto_expansion(kmeans.network)
df.at['inner-German battery investment costs', 'k-means Clustering'] = kmeans.results.loc['annual storage+store investment costs'].value - foreign_store
df.at['inner-German grid investment costs', 'k-means Clustering'] = kmeans.results.loc['annual ac grid investment costs'].value
df.at['inner-German grid expansion (TW*km)', 'k-means Clustering'] = TVA_km(kmeans.network)

df.at['system costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual system costs'].value
df.at['marginal costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual marginal costs'].value
df.at['battery investment costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual electrical storage investment costs'].value
df.at['electrical grid investment costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual electrical grid investment costs'].value
foreign_store = foreign_sto_expansion(dijkstra.network)
df.at['inner-German battery investment costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual storage+store investment costs'].value - foreign_store
df.at['inner-German grid investment costs', 'k-medoids Dijkstra Clustering'] = dijkstra.results.loc['annual ac grid investment costs'].value
df.at['inner-German grid expansion (TW*km)', 'k-medoids Dijkstra Clustering'] = TVA_km(dijkstra.network)

df.at['system costs', 'rel. difference'] = ((dijkstra.results.loc['annual system costs'].value - kmeans.results.loc['annual system costs'].value) / dijkstra.results.loc['annual system costs'].value) *100
df.at['marginal costs', 'rel. difference'] = ((dijkstra.results.loc['annual marginal costs'].value - kmeans.results.loc['annual marginal costs'].value) /dijkstra.results.loc['annual marginal costs'].value)*100
df.at['battery investment costs', 'rel. difference'] = ((dijkstra.results.loc['annual electrical storage investment costs'].value - kmeans.results.loc['annual electrical storage investment costs'].value) / dijkstra.results.loc['annual electrical storage investment costs'].value)*100
df.at['electrical grid investment costs', 'rel. difference'] = ((dijkstra.results.loc['annual electrical grid investment costs'].value - kmeans.results.loc['annual electrical grid investment costs'].value) / dijkstra.results.loc['annual electrical grid investment costs'].value)*100
df.at['inner-German battery investment costs', 'rel. difference'] = ((df.loc['inner-German battery investment costs', 'k-medoids Dijkstra Clustering'] - df.loc['inner-German battery investment costs', 'k-means Clustering']) / df.loc['inner-German battery investment costs', 'k-medoids Dijkstra Clustering'])*100
df.at['inner-German grid investment costs', 'rel. difference'] = ((dijkstra.results.loc['annual ac grid investment costs'].value-kmeans.results.loc['annual ac grid investment costs'].value)/dijkstra.results.loc['annual ac grid investment costs'].value)*100
df.at['inner-German grid expansion (TW*km)', 'rel. difference'] = ((TVA_km(dijkstra.network)-TVA_km(kmeans.network))/TVA_km(dijkstra.network))*100

loads = dijkstra.network.generators[dijkstra.network.generators.carrier=='load shedding']
ls=dijkstra.network.generators_t.p.transpose().loc[loads.index].sum().sum()
ls_costs = ls * 10000



                                                
     
        
        

