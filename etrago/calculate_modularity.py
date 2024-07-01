import pypsa
import json
from etrago import Etrago
from etrago.tools.utilities import modular_weight
import glob
import pandas as pd
import matplotlib.pyplot as plt


network_orig = pypsa.Network(import_name="orig_network.csv")

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
        weight = 1
        exclude=None
    elif i==1:
        weight = 1/network_orig.lines.x
        exclude=None
    elif i==2:
        weight = network_orig.lines.s_nom
        exclude=None

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
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: weighting 1')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: weighting 1')
    elif i==1:
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: weighting 1/jx')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: weighting 1/jx')
    elif i==2:
        plt.plot(df.index, df["dijkstra"], label='k-medoids Dijkstra Clustering: weighting s_nom', linestyle='-')
        plt.plot(df.index, df["kmeans"], label='k-means Clustering: weighting s_nom', linestyle='-')

    plt.legend()
    
    #df["dijkstra"].plot(label="dijkstra")
    #df["kmeans"].plot(label="kmeans")


'''scatter = pd.Series(network_orig.lines.length.values, index=range(0,len(network_orig.lines)))
km = scatter[scatter<1]
m = scatter[scatter<0.5]

#plt.scatter(scatter.index, scatter.values, s=1, label='other lines')
#plt.scatter(km.index, km.values, s=1, c='r', label='lines<1km')
#plt.scatter(m.index, m.values, s=1, c='g', label='lines<0.5km')
#plt.legend()

import scipy.stats as stats
import numpy as np
plt.rcParams["hist.bins"] = 100
scatter = scatter.sort_values()
fit = stats.norm.pdf(scatter, np.mean(scatter), np.std(scatter))  #this is a fitting indeed
plt.plot(scatter,fit,'-o')
plt.hist(scatter,density=True) 
plt.vlines(0.5, ymin=0, ymax=fit.max(), colors='red')
plt.vlines(1, ymin=0, ymax=fit.max(), colors='green')'''

