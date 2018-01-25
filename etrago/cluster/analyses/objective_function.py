# -*- coding: utf-8 -*-
"""
"""
from config import clustered_path, original_path #, plot_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

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
plt.show()
#fig.savefig(path.join(plot_path, 'comparison_obj_time.eps'))
