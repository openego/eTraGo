import json
import pypsa
from etrago import Etrago

import pandas as pd
import networkx as nx

### analysis functions ########################################################

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
    
###############################################################################

spatial_resolution = [20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500]

spatial_method = ['kmeans', 'kmedoids-dijkstra']

load_shedding = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
export = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
modularity = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
system_costs = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
investment = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
investment_DE = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
investment_acDE = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
marginal = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
network_expansion = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
network_ac = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
network_dc = pd.DataFrame(index=spatial_resolution, columns=spatial_method)
storage_expansion = pd.DataFrame(index=spatial_resolution, columns=spatial_method)

for i in range (0, len(spatial_method)):

    col = spatial_method[i]

    for j in range(0, len(spatial_resolution)):
        
            idx = spatial_resolution[j]

            etrago = Etrago(csv_folder_name="Calcs/"+col+"/"+str(idx))

            network = etrago.network
            
            # load_shedding
            
            loads = etrago.network.generators[etrago.network.generators.carrier=='load shedding']
            load_shedding.at[idx, col] = etrago.network.generators_t.p.transpose().loc[loads.index].sum().sum()
            
            # export
            
            export.at[idx, col] = dc_export(network)
            
            # modularity
            from etrago.tools.utilities import modular_weight
            modularity.at[idx, col] = modular_weight(etrago.busmap['orig_network'],etrago.busmap['busmap'])
            
            # system costs
            
            etrago.calc_results()
            
            system_costs.at[idx, col] = etrago.results.loc['annual system costs'].value
            investment.at[idx, col] = etrago.results.loc['annual investment costs'].value
            marginal.at[idx, col] = etrago.results.loc['annual marginal costs'].value
            
            investment_DE.at[idx, col] = investment_Germany(network)
            investment_acDE.at[idx, col] = investment_Germany(network, sto=False)
            
            # network expansion
            
            network_expansion.at[idx, col] = etrago.results.loc['abs. electrical grid expansion'].value
            network_ac.at[idx, col] = etrago.results.loc['abs. electrical ac grid expansion'].value
            network_dc.at[idx, col] = etrago.results.loc['abs. electrical dc grid expansion'].value
            
            # storage expansion
            
            storage_expansion.at[idx, col] = etrago.results.loc['battery storage expansion'].value
            
# calculation time

calc_time = pd.read_csv("Calcs/calc-time.csv")
calc_time.set_index("spatial resolution", inplace=True)

'''# plots

import matplotlib.pyplot as plt

calc_time.plot(legend=True)

load_shedding.plot(legend=True)
export.plot(legend=True)

modularity.plot(legend=True)

system_costs.plot(legend=True)
investment.plot(legend=True)
marginal.plot(legend=True)

investment_DE.plot(legend=True)
investment_acDE.plot(legend=True)

network_expansion.plot(legend=True)
network_ac.plot(legend=True)
network_dc.plot(legend=True)

storage_expansion.plot(legend=True)'''

etrago = Etrago(csv_folder_name="Calcs/kmeans/500")

etrago.plot_grid(line_colors='expansion_abs', bus_colors='storage_expansion', bus_sizes=0.000001, ext_min=0.01, legend_entries=[], boundaries=[0, 20000])

etrago = Etrago(csv_folder_name="Calcs/kmedoids-dijkstra/500")

etrago.plot_grid(line_colors='expansion_abs', bus_colors='storage_expansion', bus_sizes=0.000001, ext_min=0.01, legend_entries=[], boundaries=[0, 20000])

###############################################################################

# analysis of system costs

import json
import pypsa
from etrago import Etrago

import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

spatial_resolution = [20, 50, 100, 150, 200, 250, 300, 400, 500]

spatial_method = ['kmeans', 'kmedoids-dijkstra']

# dataframe for system costs = marginal + investment costs
costs1 = pd.DataFrame(columns=['marginal costs', 'store investment costs', 'electrical grid investment costs'])# 'electrical grid investment costs in Germany', 'electrical grid investment costs in foreign countries'], index=spatial_resolution)
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
            costs.at[idx, 'store investment costs'] = etrago.results.loc['annual storage+store investment costs'].value
            
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

#relac_expansion.plot(legend=True)
#reldc_expansion.plot(legend=True)
#relsto_expansion.plot(legend=True)
#rel_marginal.plot(legend=True)
#rel_objective.plot(legend=True)

# plot system costs as bar diagram

costs1 = costs1 / 1000000000
costs2 = costs2 / 1000000000

fig = plt.figure()
ax = fig.add_subplot(111)

index1 = spatial_resolution.copy()
index2 = spatial_resolution.copy()

for i in range(0, len(index1)):
    index1[i] = index1[i] - 5
    index2[i] = index2[i] + 5

bottom1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bottom2=bottom1.copy()
i = -1

color1 = ['darkslategrey', 'darkcyan', 'skyblue']#, 'skyblue'] # kmeans
color2 = ['maroon', 'tomato', 'lightsalmon']#, 'lightsalmon'] # kmedoids-djkstra

hatch = [False, False, False, "/"]

for col in costs.columns:
    
    if i >= 0:
        bottom1 = bottom1 + costs1[costs1.columns[i]].values
        bottom2 = bottom2 + costs2[costs2.columns[i]].values
    i = i+1
    
    if hatch[i]:
        plt.bar(index1, costs1[col].values, width = 4, bottom = bottom1, color=color1[i], hatch=hatch[i], label='k-means Clustring: '+col)
        plt.bar(index2, costs2[col].values, width = 4, bottom = bottom2, color=color2[i], hatch=hatch[i], label='k-medoids Dijkstra Clustering: '+col)
    else:   
        plt.bar(index1, costs1[col].values, width = 4, bottom = bottom1, color=color1[i], label='k-means Clustring: '+col)
        plt.bar(index2, costs2[col].values, width = 4, bottom = bottom2, color=color2[i], label='k-medoids Dijkstra Clustering: '+col)

plt.xticks(spatial_resolution)

plt.legend(loc='lower right')
plt.ylabel('costs in billion Euro')
plt.xlabel('number of nodes')
plt.title('System Costs depending on Spatial Resolution')

# plot investment costs as bar diagram

inv1 = inv1 / 1000000000
inv2 = inv2 / 1000000000

fig = plt.figure()
ax = fig.add_subplot(111)

index1 = spatial_resolution.copy()
index2 = spatial_resolution.copy()

for i in range(0, len(index1)):
    index1[i] = index1[i] - 5
    index2[i] = index2[i] + 5

bottom1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bottom2=bottom1.copy()
i = -1

color1 = ['darkslategrey', 'darkcyan', 'skyblue', 'lavender'] # kmeans
color2 = ['maroon', 'tomato', 'lightsalmon', 'peachpuff'] # kmedoids-djkstra
    
for col in inv.columns:
    
    if i >= 0:
        bottom1 = bottom1 + inv1[inv1.columns[i]].values
        bottom2 = bottom2 + inv2[inv2.columns[i]].values
    i = i+1

    plt.bar(index1, inv1[col].values, width = 4, bottom = bottom1, color=color1[i], label='k-means Clustring: '+col)
    plt.bar(index2, inv2[col].values, width = 4, bottom = bottom2, color=color2[i], label='k-medoids Dijkstra Clustering: '+col)
    
plt.legend(loc='lower right')
plt.ylabel('costs in billion Euro')
plt.xlabel('number of nodes')
plt.title('Investment Costs depending on Spatial Resolution')
    
# plot investment costs in Germany

inv1_DE = inv1_DE / 1000000000
inv2_DE = inv2_DE / 1000000000

fig = plt.figure()
ax = fig.add_subplot(111)

index1 = spatial_resolution.copy()
index2 = spatial_resolution.copy()

for i in range(0, len(index1)):
    index1[i] = index1[i] - 5
    index2[i] = index2[i] + 5

bottom1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bottom2=bottom1.copy()
i = -1

color1 = ['darkslategrey', 'skyblue'] # kmeans
color2 = ['maroon', 'lightsalmon'] # kmedoids-djkstra
    
for col in inv_DE.columns:
    
    if i >= 0:
        bottom1 = bottom1 + inv1_DE[inv1_DE.columns[i]].values
        bottom2 = bottom2 + inv2_DE[inv2_DE.columns[i]].values
    i = i+1

    plt.bar(index1, inv1_DE[col].values, width = 4, bottom = bottom1, color=color1[i], label='k-means Clustring: '+col)
    plt.bar(index2, inv2_DE[col].values, width = 4, bottom = bottom2, color=color2[i], label='k-medoids Dijkstra Clustering: '+col)
    
plt.legend(loc='lower right')
plt.ylabel('costs in billion Euro')
plt.xlabel('number of nodes')
plt.title('Investment Costs in Germany depending on Spatial Resolution')
    
def compare_sector_coupled(path_dijkstra, path_kmeans):
    from matplotlib import pyplot as plt
    from matplotlib import pylab
    from etrago import Etrago

    ed = Etrago(csv_folder_name=path_dijkstra)
    ek = Etrago(csv_folder_name=path_kmeans)

    def plot_diff(df, file):
        fig, ax = plt.subplots()
        plt.xticks(rotation=90)
        plt.tight_layout()
        ax.bar(df.index, df["diff[%]"])
        plt.ylabel("kmeans - dijkstra [%]")

        filename = f"dijkstra-paper/{file}"
        pylab.savefig(filename, dpi=300, bbox_inches="tight")

        return


    ed.calc_results()
    ek.calc_results()

    results = ed.results.rename(columns={"value":"dijkstra"})
    results["kmeans"] = ek.results["value"]
    results["diff"] = results["kmeans"] - results["dijkstra"]
    results["diff[%]"] = (results["kmeans"] - results["dijkstra"])/results["kmeans"] * 100

    # Group the costs
    costs = results[results.index.str.contains("costs")]

    # group expansion
    expansion = results[results.index.str.contains("expansion")]
    expansion = expansion[expansion.unit != "electrical grid expansion"]
    expansion = expansion[expansion.index != "p.u."]
    expansion.dropna(inplace=True)

    plot_diff(costs, "cost_diff")
    plot_diff(expansion, "expansion_diff")

    # group investment costs
    inv_cost = results[results.index.str.contains("investment costs")]
    inv_cost = inv_cost[inv_cost.index != "annual investment costs"]
    inv_cost = inv_cost[inv_cost.index != "annual electrical grid investment costs"]
    inv_cost = inv_cost[inv_cost.index != "annual storage+store investment costs"]

    fig, ax = plt.subplots()
    plt.title("Absolute diff in investment costs")
    ax.pie(inv_cost["diff"].abs(), labels=inv_cost.index)

    filename = "dijkstra-paper/Absolute diff in investment costs"
    pylab.savefig(filename, dpi=300, bbox_inches="tight")

compare_sector_coupled("/home/student/git/eTraGo/etrago/dijkstra-paper/calculations/eGon2035",
                       "/home/student/git/eTraGo/etrago/dijkstra-paper/calculations/eGon2035_kmeans")
    
    



            
        
            


