#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import clustered_path, original_path, plot_path
from os import path, listdir
from etrago.tools.plot import network_extension_diff 
import pandas as pd
import pypsa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

abs_err = {}
rel_err = {}
rel_err_mean_lines = {}
rel_err_mean_storages = {}

k = [10, 20]

for i in k:
    lines = pd.read_csv(path.join(original_path, str(i), 'lines.csv'))
    lines_c = pd.read_csv(path.join(clustered_path, str(i), 'lines.csv'))

    rel_err_mean_lines[str(i)] = ((((lines_c['s_nom_opt'] -\
        lines['s_nom_opt'])) / lines['s_nom_opt'])*100).mean()
    
    storages = pd.read_csv(path.join(original_path, str(i), 
         'storage_units.csv'))
    storages_c = pd.read_csv(path.join(clustered_path, str(i),
        'storage_units.csv'))
    
    rel_err_mean_storages[str(i)] = ((((storages_c['p_nom_opt']
        [storages_c.p_nom_extendable == True] -\
        storages['p_nom_opt'][storages.p_nom_extendable == True])) /\
        storages['p_nom_opt'][storages.p_nom_extendable == True])*100).mean()
    
    networkA = pypsa.Network(csv_folder_name=path.join(original_path, str(i)))
    networkB = pypsa.Network(csv_folder_name=path.join(clustered_path, str(i)))
    
    network_extension_diff(networkA, networkB, filename=path.join(plot_path,
        (str(i) + 'extension_diff_network.eps') ))
    

results = pd.DataFrame({'rel_err_mean_lines': rel_err_mean_lines,
                       'rel_err_mean_storages': rel_err_mean_storages})



ax =results['rel_err_mean_lines'].plot(style='*--', label = 'lines')
ax2 = ax.twinx()
ax.set_title('Comparison (BM - Method)')
ax.set_ylabel('Relative mean line extension deviation in %')
ax.set_xlabel('Number of buses')
results['rel_err_mean_storages'].plot(ax=ax2, style='*--', color='red')
ax.plot(np.nan, '-r', label = 'storages')
ax2.set_ylabel('Relative mean strorage expansion deviation in %')
ax.legend(loc=0)
fig = ax.get_figure()
fig.savefig(path.join(plot_path, 'comparison_extension.eps'))


def plot_heatmap(buses, difference, minmax=(None,None)):
    """
    """
    x = np.array(buses['x'])
    y = np.array(buses['y'])

    alpha = np.array([i for i in difference.values()])

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    cmap = plt.cm.jet

    plt.hexbin(x, y, C=alpha, cmap=cmap, gridsize=20, vmin=minmax[0],
        vmax=minmax[1])

    cb = plt.colorbar()

    # plt.clim(-100, 100) normalized colors
    cb.set_label('Absolute difference of Installed Storage Capacities in MW')
    
    return(fig)


# TODO: check why are there multiple storages for e.g. bus with name 2 and 9


# store all storage results
clustered = {}
# nested dict with absolute difference per cluster and bus
abs_diff = {}

c_df = pd.DataFrame()

for i in k:
    original = pd.read_csv(path.join(original_path, str(i), 'storage_units.csv'))
    buses = pd.read_csv(path.join(original_path, str(i), 'buses.csv'))
    abs_diff[str(i)] = {}
    df = pd.read_csv(path.join(clustered_path, str(i), 'storage_units.csv'))
    df['k'] = int(str(i))
    df['abs_diff'] = df.p_nom_opt - original.p_nom_opt
    abs_diff[str(i)] = {
            b: sum(df.loc[df['bus'] == b]['abs_diff'])
            for b in buses['name']}
    c_df = pd.concat([c_df, df])
    clustered[str(i)] = df

    minmax = (min(abs_diff[str(i)].values()),
          max(abs_diff[str(i)].values()))

    with PdfPages(path.join(plot_path, (str(i) + 'storage-capacities.pdf')))\
        as pdf:
            fig = plot_heatmap(buses, abs_diff[str(i)], minmax=minmax)
            pdf.savefig(fig)
