# -*- coding: utf-8 -*-
"""
"""
from os import path, listdir

import matplotlib
import pandas as pd

results_dir = 'snapshot-clustering-results-k10-noDailyBounds'
clustered_path = path.join('/home/simnh/pf_results', results_dir, 'daily')
original_path = path.join('/home/simnh/pf_results', results_dir, 'original')

network = pd.read_csv(path.join(original_path, 'network.csv'))

abs_err = {}
rel_err = {}
abs_time = {}
rel_time = {}

for c in listdir(clustered_path):
    if c != 'Z.csv':
        network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))
        abs_err[str(c)] = (abs(network_c['objective'].values[0] -
                               network['objective'].values[0])) * 100
        rel_err[str(c)] = abs_err[str(c)] / network['objective'].values[0]
        abs_time[str(c)] = float(network_c['time'])
        rel_time[str(c)] = (float(network_c['time']) /
                            float(network['time']) * 100)

results = pd.DataFrame({
    'abs_err': abs_err,
    'rel_err': rel_err,
    'abs_time': abs_time,
    'rel_time': rel_time})
results.index = [int(i) for i in results.index]
results.sort_index(inplace=True)

# plotting 2 axis plot
ax = results['rel_err'].plot(style='--*')
ax2 = ax.twinx()
ax.set_title('Comparison')
ax.set_ylabel('Relative objective function deviation in %')
ax.set_xlabel('Clustered Days')
results['rel_time'].plot(ax=ax2, style='*--', color='red')
ax2.set_ylabel('Relative run-time deviation in %')
fig = ax.get_figure()
# fig.savefig("comparison_obj_time.eps")
