import pypsa
import json
from etrago import Etrago
from etrago.tools.utilities import modular_weight
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', size=22)

def split_parallel_lines(network):
    print("Splitting parallel lines...")

    parallel_lines = network.lines[network.lines.num_parallel > 1]
    s_max_pu = network.lines_t.s_max_pu[
        parallel_lines[
            parallel_lines.index.isin(network.lines_t.s_max_pu.columns)].index]

    new_lines = pd.DataFrame(columns=network.lines.columns)
    new_lines_t = pd.DataFrame(index=network.snapshots)
    for i in parallel_lines.index:
        data_new = parallel_lines[parallel_lines.index == i]
        for col in ["b", "g", "s_nom", "s_nom_min", "s_nom_max", "s_nom_opt"]:
            data_new[col] = data_new[col] / data_new.num_parallel
        for col in ["x", "r"]:
            data_new[col] = data_new[col] * data_new.num_parallel
        data_new.cables = 3
        data_new.num_parallel = 1
        num = parallel_lines.num_parallel[i]
        for n in range(int(num)):
            data_new.index = [str(i) + "_" + str(int(n + 1))]
            new_lines = pd.concat(
                [
                    new_lines,
                    data_new,
                ],
            )
            if i in s_max_pu.columns:
                new_lines_t.loc[:, data_new.index] =  s_max_pu[i]

    network.mremove("Line", parallel_lines.index)

    network.import_components_from_dataframe(new_lines, "Line")

    if not new_lines_t.empty:
        network.import_series_from_dataframe(new_lines_t, "Line", "s_max_pu")

    for i in network.lines.index[
            ~network.lines.index.isin(network.lines_t.s_max_pu.columns)]:
        network.lines_t.s_max_pu[i] = network.lines.s_max_pu[i]
    return network


network_orig = pypsa.Network(import_name="orig_network.csv")

vorher = network_orig.lines.num_parallel

bus = network_orig.buses[network_orig.buses.v_nom==380.]
li = network_orig.lines[network_orig.lines.bus0.isin(bus.index)&network_orig.lines.bus1.isin(bus.index)]

network_orig.lines.loc[(li.s_nom%1790==0).index, "num_parallel"] = li.s_nom/1790
network_orig.lines.loc[(li.s_nom%925==0).index, "num_parallel"] = li.s_nom/925

bus = network_orig.buses[network_orig.buses.v_nom==220.]
li = network_orig.lines[network_orig.lines.bus0.isin(bus.index)&network_orig.lines.bus1.isin(bus.index)]

network_orig.lines.loc[(li.s_nom%520==0).index, "num_parallel"] = li.s_nom/520
network_orig.lines.loc[(li.s_nom%550==0).index, "num_parallel"] = li.s_nom/550

bus = network_orig.buses[network_orig.buses.v_nom==110.]
li = network_orig.lines[network_orig.lines.bus0.isin(bus.index)&network_orig.lines.bus1.isin(bus.index)]

network_orig.lines.loc[(li.s_nom%260==0).index, "num_parallel"] = li.s_nom/260
network_orig.lines.loc[(li.s_nom%280==0).index, "num_parallel"] = li.s_nom/280

#network_orig.lines.at['28283', 'num_parallel'] = 4
network_orig = split_parallel_lines(network_orig)

buses = network_orig.buses.copy()
lines = network_orig.lines.copy()
buses_foreign = buses[buses.country != "DE"].index

network_orig.buses = buses[~buses.index.isin(buses_foreign)]
network_orig.lines = lines[(lines.bus0.isin(buses.index)) & (lines.bus1.isin(buses.index))]

# Take all lines to the same voltage level
network_orig.lines["v_nom"] = network_orig.lines.bus0.map(network_orig.buses.v_nom)
lines_v_nom_b = network_orig.lines.v_nom != 380

voltage_factor = (network_orig.lines.loc[lines_v_nom_b, "v_nom"] / 380.0) ** 2

network_orig.lines.loc[lines_v_nom_b, "x"] *= 1 / voltage_factor


for i in range(0,3):
    
    if i==0:
        weight = 1/network_orig.lines.x
        exclude = None
    elif i==1:
        weight = 1/network_orig.lines.x
        exclude = 0.25
    elif i==2:
        weight = 1
        excude = None

    df = pd.DataFrame(columns=["dijkstra","kmeans"])

    folders_dij = glob.glob("Calcs/kmedoids-dijkstra/*")
    for folder in folders_dij:
        n_clusters = int(folder.split("/")[-1])
        busmap_path = folder + "/clustering/busmap.json"
    
        with open(busmap_path) as json_data:
            busmap = json.load(json_data)
            json_data.close()
        df.at[n_clusters, "dijkstra"] = modular_weight(network_orig, busmap, weight, exclude)
    
    folders_k = glob.glob("Calcs/kmeans/*")
    for folder in folders_k:
        n_clusters = int(folder.split("/")[-1])
        busmap_path = folder + "/clustering/busmap.json"
    
        with open(busmap_path) as json_data:
            busmap = json.load(json_data)
            json_data.close()
        df.at[n_clusters, "kmeans"] = modular_weight(network_orig, busmap, weight, exclude)
        
    df.sort_index(inplace=True)
    
    if i==0:
        #print('x')
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: weighting 1/X', color='C0', linestyle='dashed')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: weighting 1/X', color='orange', linestyle='dashed')
    elif i==1:
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: weighting 1/X w/o quartile shortest lines', color='C0', linestyle='dotted')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: weighting 1/X w/o quartile shortest lines', color='orange' , linestyle='dotted')
    elif i==2:
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: non-weighted')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: non-weighted')
        
    plt.xlabel('number of nodes')
    plt.ylabel("modularity")
    plt.legend(bbox_to_anchor=(0.5, 1.45), loc='upper center')
    plt.subplots_adjust(top=0.7)
    #plt.tight_layout()
    plt.show()
    #plt.legend(loc='lower center')
    #bbox_to_anchor = [0.4, 0.9])
    
    #df["dijkstra"].plot(label="dijkstra")
    #df["kmeans"].plot(label="kmeans")


'''percentile_25 = network_orig.lines['x'].quantile(0.25)
lines = network_orig.lines[network_orig.lines['x'] > percentile_25]
#lines = network_orig.lines #.loc[network_orig.lines['length']<50]
scatter = pd.Series(lines.x.values, index=range(0,len(lines)))
km = scatter[scatter<1]
m = scatter[scatter<0.5]

#plt.scatter(scatter.index, scatter.values, s=1, label='other lines')
#plt.scatter(km.index, km.values, s=1, c='r', label='lines<1km')
#plt.scatter(m.index, m.values, s=1, c='g', label='lines<0.5km')
#plt.legend()

import scipy.stats as stats
import numpy as np
plt.rcParams["hist.bins"] = 1000
scatter = scatter.sort_values()
fit = stats.norm.pdf(scatter, np.mean(scatter), np.std(scatter))  #this is a fitting indeed
plt.plot(scatter,fit,'-o')
plt.hist(scatter,density=True) 
plt.vlines(0.5, ymin=0, ymax=fit.max(), colors='red')
plt.vlines(1, ymin=0, ymax=fit.max(), colors='green')'''

