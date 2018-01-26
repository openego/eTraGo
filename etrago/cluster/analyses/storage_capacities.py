# -*- coding: utf-8 -*-
"""
"""
from config import clustered_path, original_path, plot_path

from os import path, listdir

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

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

original = pd.read_csv(path.join(original_path, 'storage_units.csv'))
buses = pd.read_csv(path.join(original_path, 'buses.csv'))

# store all storage results
clustered = {}
# nested dict with absolute difference per cluster and bus
abs_diff = {}

c_df = pd.DataFrame()
for d in listdir(clustered_path):
    # TODO: excluse distance matrix, move this file to the root directory?
    if d != 'Z.csv':
        abs_diff[d] = {}
        df = pd.read_csv(path.join(clustered_path, d, 'storage_units.csv'))
        df['cdays'] = int(d)
        df['abs_diff'] = df.p_nom_opt - original.p_nom_opt
        abs_diff[d] = {
            b: sum(df.loc[df['bus'] == b]['abs_diff'])
            for b in buses['name']}
        c_df = pd.concat([c_df, df])
        clustered[d] = df

# plot
minmax = (min([min([j for j in i.values()]) for i in abs_diff.values()]),
          max([max([j for j in i.values()]) for i in abs_diff.values()]))

with PdfPages(path.join(plot_path, 'storage-capacities.pdf')) as pdf:
    for d in listdir(clustered_path):
        if d != 'Z.csv':
            fig = plot_heatmap(buses, abs_diff[d], minmax=minmax)
            pdf.savefig(fig)

# experimental stuff
# c_df.set_index(['cdays', 'name'], inplace=True)
# c_df.sort_index()
