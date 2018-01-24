# -*- coding: utf-8 -*-
"""
"""

from os import path, listdir

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# TODO: check why are there multiple storages for e.g. bus with name 2 and 9

results_dir = 'snapshot-clustering-results-k10-noDailyBounds'

# get all directories from the result directory
daily_path = path.join('/home/simnh/pf_results', results_dir, 'daily')
original_path = path.join('/home/simnh/pf_results', results_dir, 'original')

original = pd.read_csv(path.join(original_path, 'storage_units.csv'))
buses = pd.read_csv(path.join(original_path, 'buses.csv'))

# store all storage results
clustered = {}
# nested dict with absolute difference per cluster and bus
abs_diff = {}
for d in listdir(daily_path):
    # TODO: excluse distance matrix, move this file to the root directory?
    if d != 'Z.csv':
        abs_diff[d] = {}
        clustered[d] = pd.read_csv(
            path.join(daily_path, d, 'storage_units.csv'))
        # add absolute storage capacity difference to dataframe
        clustered[d]['abs_diff'] = (
            original.p_nom_opt -
            clustered[d].p_nom_opt)
        abs_diff[d] = {
            b: sum(clustered[d].loc[clustered[d]['bus'] == b]['abs_diff'])
            for b in buses['name']}


def plot_heatmap(buses, difference):
    """
    """
    x = np.array(buses['x'])
    y = np.array(buses['y'])

    alpha = np.array([i for i in difference.values()])

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    cmap = plt.cm.jet

    plt.hexbin(x, y, C=alpha, cmap=cmap, gridsize=20)
    cb = plt.colorbar()

    cb.set_label('Absolute difference of Installed Storage Capacities in MW')
    plt.show()
    fig.savefig('clustered.png')

plot_heatmap(buses, abs_diff['10'], boundaries=[])
