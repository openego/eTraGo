#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:17 2018

@author: clara
"""

from config import clustered_path, original_path, plot_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

from etrago.tools.plot import network_extension_diff 
from matplotlib import pyplot as plt
import pandas as pd
import pypsa
abs_err = {}
rel_err = {}
rel_err_mean_lines = {}

rel_err_mean_storages = {}
k = [10, 20]

for i in k:
    lines = pd.read_csv(path.join(original_path, str(i), 'lines.csv'))
    lines_c = pd.read_csv(path.join(clustered_path, str(i), 'lines.csv'))

    rel_err_mean_lines[str(i)] = (((abs(lines_c['s_nom_opt'] -\
                               lines['s_nom_opt'])) / lines['s_nom_opt'])*100).mean()
    
    storages = pd.read_csv(path.join(original_path, str(i), 'storage_units.csv'))
    storages_c = pd.read_csv(path.join(clustered_path, str(i), 'storage_units.csv'))
    
    rel_err_mean_storages[str(i)] = (((abs(storages_c['p_nom_opt'][storages_c.p_nom_extendable == True] -\
                               storages['p_nom_opt'][storages.p_nom_extendable == True])) /\
                                storages['p_nom_opt'][storages.p_nom_extendable == True])*100).mean()
    
    networkA = pypsa.Network(csv_folder_name=path.join(original_path, str(i)))
    networkB = pypsa.Network(csv_folder_name=path.join(clustered_path, str(i)))
    
    network_extension_diff(networkA, networkB, filename=path.join(plot_path, (str(i) + 'extension_diff_network.png') ))
    

results = pd.DataFrame({'rel_err_mean_lines': rel_err_mean_lines,
                       'rel_err_mean_storages': rel_err_mean_storages})



ax =results['rel_err_mean_lines'].plot(style='--*')
ax2 = ax.twinx()
ax.set_title('Comparison')
ax.set_ylabel('Relative mean line extension deviation in %')
ax.set_xlabel('Number of buses')
results['rel_err_mean_storages'].plot(ax=ax2, style='*--', color='red')
ax2.set_ylabel('Relative mean strorage expansion deviation in %')
fig = ax.get_figure()
fig.savefig(path.join(plot_path, 'comparison_extension.eps'))