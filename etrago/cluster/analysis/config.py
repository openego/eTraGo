# -*- coding: utf-8 -*-
"""
"""
from os import path


root_path = path.join(path.expanduser('~'),'pf_results/')

#path for results of individual simulations
sim_results_path = path.join(root_path, 'simulation_results/')

#path for results exported in ResultsOverview.py 
total_results_path = path.join(root_path, 'total_results/')

plot_path = total_results_path


