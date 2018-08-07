#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import clustered_path, original_path, plot_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

abs_err = {}
rel_err = {}
abs_time = {}
rel_time = {}

k = [10,20,30,50]
for i in k:
    network = pd.read_csv(path.join(original_path, str(i), 'network.csv'))
    network_c = pd.read_csv(path.join(clustered_path, str(i), 'network.csv'))
    abs_err[str(i)] = (abs(network_c['objective'].values[0] -\
                               network['objective'].values[0])) 
    rel_err[str(i)]= abs_err[str(i)] / network['objective'].values[0]* 100
    abs_time[str(i)] = float(network_c['time'])
    rel_time[str(i)] = (float(network_c['time']) / float(network['time']) * 100)

results = pd.DataFrame({
    'abs_err': abs_err,
    'rel_err': rel_err,
    'abs_time': abs_time,
    'rel_time': rel_time})
results.index = [int(i) for i in results.index]
results.sort_index(inplace=True)

# plotting 2 axis plot
ax = results['rel_err'].plot(style='--*', label = 'error')
ax2 = ax.twinx()
ax.set_title('Comparison')
ax.set_ylabel('Relative objective function deviation in %')
ax.set_xlabel('Number of buses')
results['rel_time'].plot(ax=ax2, style='*--', color='red', label = 'time')
ax2.set_ylabel('Relative run-time deviation in %')
fig = ax.get_figure()
fig.savefig(path.join(plot_path, 'comparison_obj_time.eps'))