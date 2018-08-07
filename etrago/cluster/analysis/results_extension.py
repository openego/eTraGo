#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:06:17 2018

@author: clara
"""

from config import clustered_path, original_path, plot_path
# TODO: Make plot_path import work...don't know why it does not
from os import path, listdir

from matplotlib import pyplot as plt
import pandas as pd
abs_err = {}
rel_err = {}
k = [10]
for i in k:
    lines = pd.read_csv(path.join(original_path, str(i), 'lines.csv'))
    lines_c = pd.read_csv(path.join(clustered_path, str(i), 'lines.csv'))
    abs_err[str(i)] = (abs(lines_c['s_nom_opt'] -\
                               lines['s_nom_opt']))
    rel_err[str(i)]= (abs_err[str(i)] / lines['s_nom_opt'])*100
    rel_err_mean = (((abs(lines_c['s_nom_opt'] -\
                               lines['s_nom_opt'])) / lines['s_nom_opt'])*100).mean()
    
