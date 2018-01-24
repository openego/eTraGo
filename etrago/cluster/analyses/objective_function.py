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
for c in listdir(clustered_path):
    if c != 'Z.csv':
        network_c = pd.read_csv(path.join(clustered_path, c, 'network.csv'))
        abs_err[str(c)] = (abs(network_c['objective'].values[0] -
                               network['objective'].values[0]))
        rel_err[str(c)] = abs_err[str(c)] / network['objective'].values[0]

errors = pd.DataFrame({'abs_err': abs_err, 'rel_err': rel_err})
errors.index = [int(i) for i in errors.index]
errors.sort_index(inplace=True)

errors['rel_err'].plot(style='--*')
