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
    return(fig)


# TODO: check why are there multiple storages for e.g. bus with name 2 and 9

original = pd.read_csv(path.join(original_path, 'storage_units.csv'))
buses = pd.read_csv(path.join(original_path, 'buses.csv'))

# store all storage results
clustered = {}
# nested dict with absolute difference per cluster and bus
abs_diff = {}

with PdfPages(path.join(plot_path, 'storage-capacities.pdf')) as pdf:
    for d in listdir(clustered_path):
        # TODO: excluse distance matrix, move this file to the root directory?
        if d != 'Z.csv':
            abs_diff[d] = {}
            clustered[d] = pd.read_csv(
                path.join(clustered_path, d, 'storage_units.csv'))
            # add absolute storage capacity difference to dataframe
            clustered[d]['abs_diff'] = (
                original.p_nom_opt -
                clustered[d].p_nom_opt)
            abs_diff[d] = {
                b: sum(clustered[d].loc[clustered[d]['bus'] == b]['abs_diff'])
                for b in buses['name']}
            fig = plot_heatmap(buses, abs_diff[d])
            pdf.savefig(fig)
