#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:33:11 2023

@author: clara
"""
import cartopy
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes

import geopandas as gpd
from etrago import Etrago
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point, LineString

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 8})

# #etrago: Dijkstra
etrago_d = Etrago(
    csv_folder_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/calculations/kmedoids-dijkstra/300",
)
#etrago: kmeans
etrago_k = Etrago(
    csv_folder_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/calculations/kmeans/300",
)
# etrago: original
etrago_o = Etrago(
    csv_folder_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/calculations/eGon2035_orig",
)

# Import DDR geodata
ddr_shape_path = "/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/gis/DDR.geojson"
ddr = gpd.read_file(ddr_shape_path)
ddr = ddr.drop(index=[8])
ddr_u = gpd.GeoDataFrame({"geometry": ddr.unary_union}, crs="EPSG:4326", index=["ddr"])
ddr_bound = gpd.GeoDataFrame({"geometry": ddr_u.boundary}, crs="EPSG:4326", index=["ddr"])

def plot_topo(etrago, ddr=ddr_u, file_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/gis/DDR.png"):

    def order(l):
        prio = 0
        if ((l["to_ddr"])&(l["v_nom"]==110)):
            prio = 3
        elif l["v_nom"]>110:
            prio = 2
        return prio
        
    
    link_w = pd.Series(index = etrago.network.links.index, data =0)
    
    link_w[etrago.network.links[etrago.network.links.carrier.isin(['DC'])].index] = 1
    
    link_c = pd.Series(index = etrago.network.links.index, data ='darkblue')
    
    link_c[etrago.network.links[etrago.network.links.carrier.isin(['CH4'])].index] = 'orange'
    
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    ddr_u.plot(ax=ax, color= "thistle")
    
    # Mark coss DDR lines
    buses = etrago.network.buses
    buses['x'] = buses['x'].apply(float)
    buses['y'] = buses['y'].apply(float)
    buses['geom'] = buses.apply(lambda x: Point(x.x, x.y), axis= 1)
    buses = gpd.GeoDataFrame(buses, geometry= "geom", crs= 4326)

    lines = etrago.network.lines
    lines['geom'] = lines.apply(lambda x: LineString([buses.at[x.bus0, 'geom'],
                                                      buses.at[x.bus1, 'geom']]),
                                axis= 1)
    lines = gpd.GeoDataFrame(lines, geometry= "geom", crs= 4326)
    lines["v_nom"] = (lines["bus0"].map(buses.v_nom)).apply(int)
    buses["in_ddr"] = buses.intersects(ddr.unary_union)
    
    voltage = {110: "skyblue",
               220: "darkblue",
               380: "darkblue"}
    
    voltage_ddr =  {110: "magenta",
                    220: "orange",
                    380: "orange"}
    
    
    
    lines["bus0_in_ddr"] = lines["bus0"].map(buses["in_ddr"])
    lines["bus1_in_ddr"] = lines["bus1"].map(buses["in_ddr"])
    lines["to_ddr"] = lines["bus0_in_ddr"] ^ lines["bus1_in_ddr"]
    lines["order"] = lines.apply(order, axis=1)
    etrago.network.lines["order"] = etrago.network.lines.apply(order, axis=1)
    etrago.network.lines.sort_values(by="order", inplace=True)
    lines.sort_values(by="order", inplace= True)
    lines["color"] = lines.apply(lambda x: voltage_ddr[x["v_nom"]] if x["to_ddr"] else voltage[x["v_nom"]], axis=1)
    lines["width"] = lines.to_ddr.apply(lambda x: 2 if x else 1)
    
    
    
    
    etrago.network.plot(link_widths=link_w,
                        link_colors = link_c,
                        bus_sizes = 0,
                        line_colors = lines["color"],
                        line_widths = lines["width"],
                        ax=ax,
                        geomap=True,
                        projection=ccrs.PlateCarree(),
                        color_geomap=True,
                        boundaries=[5, 16, 46.8, 58],)

    legend_elements = [Line2D([0], [0], color='darkblue', lw=2, label='eHV line'),
                       Line2D([0], [0], color='orange', lw=2, label='former cross-border eHV line'),
                       #Line2D([0], [0], color='skyblue', lw=2, label='HV line'),
                       #Line2D([0], [0], color='magenta', lw=2, label='former cross-border HV line'),
                       ]

    ax.legend(handles=legend_elements, loc='upper left')
    
    from matplotlib import pylab
    
    pylab.savefig(file_name, dpi=600, bbox_inches="tight")

plot_topo(etrago_d, file_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/gis/topo_dijkstra.png")
plot_topo(etrago_k, file_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/gis/topo_k-means.png")
plot_topo(etrago_o, file_name="/home/carlos/git/eTraGo_old/etrago/dijkstra-paper/gis/topo_orig.png")
