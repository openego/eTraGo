#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:26:19 2022

@author: student
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from etrago import Etrago

def import_network():

    network = pypsa.Network()
    network.set_snapshots(pd.date_range(
    start='10/06/2021-00:00', freq='H', periods=2))

### Electrical Sector###

    network.add(
        'Bus',
        name= 'AC 1',
        carrier= 'AC',
        v_nm = 380,
        x = 10.6414678044968,
        y = 53.9191626912878)

    network.add(
        "Generator",
        name = "wind turbine",
        carrier = "wind",
        bus = "AC 1",
        p_nom = 100,
        control= 'PV',
        p_max_pu = [0.85, 0.8])
    
    network.add(
        "Generator",
        name = "gas turbine",
        carrier = "gas",
        bus = "AC 1",
        p_nom = 300,
        p_min_pu = 0,
        control= 'PV',
        marginal_cost=450)

    network.add(
        "Load",
        name = "AC load 1",
        carrier = "AC",
        bus = "AC 1",
        p_set = [100, 125])
        
    
    return network

args = {
    # Setup and Configuration:
    'db': 'egon-data',  # database session
    'gridversion': None,  # None for model_draft or Version number
    'method': { # Choose method and settings for optimization
        'type': 'lopf', # type of optimization, currently only 'lopf'
        'n_iter': 2, # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        'pyomo': True}, # set if pyomo is used for model building
    'pf_post_lopf': {
        'active': False, # choose if perform a pf after a lopf simulation
        'add_foreign_lopf': True, # keep results of lopf for foreign DC-links
        'q_allocation': 'p_nom'}, # allocate reactive power via 'p_nom' or 'p'
    'start_snapshot': 1,
    'end_snapshot': 3,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {},
    'model_formulation': 'kirchhoff', # angles or kirchhoff
    'scn_name': 'eGon2035',  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    'scn_extension': None,  # None or array of extension scenarios
    'scn_decommissioning': None,  # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'csv_export': 'results',  # save results as csv: False or /path/tofolder
    # Settings:
    'extendable': [],  # Array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'extra_functionality':{},  # Choose function name or {}
    # Clustering:
    'network_clustering_kmeans': {
        'active': False, # choose if clustering is activated
        'n_clusters': 10, # number of resulting nodes
        'kmeans_busmap': False, # False or path/to/busmap.csv
        'line_length_factor': 1, #
        'remove_stubs': False, # remove stubs bevore kmeans clustering
        'use_reduced_coordinates': False, #
        'bus_weight_tocsv': None, # None or path/to/bus_weight.csv
        'bus_weight_fromcsv': None, # None or path/to/bus_weight.csv
        'n_init': 10, # affects clustering algorithm, only change when neccesary
        'max_iter': 100, # affects clustering algorithm, only change when neccesary
        'tol': 1e-6, # affects clustering algorithm, only change when neccesary
        'n_jobs': -1}, # affects clustering algorithm, only change when neccesary
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': None,  # None, 'mini' or 'uniform'
    'snapshot_clustering': {
        'active': False, # choose if clustering is activated
        'n_clusters': 2, # number of periods
        'how': 'daily', # type of period, currently only 'daily'
        'storage_constraints': 'soc_constraints'}, # additional constraints for storages
    # Simplifications:
    'skip_snapshots': False, # False or number of snapshots to skip
    'branch_capacity_factor': {'HV':1, 'eHV': 1},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'DC', # 'DC' for modeling foreign lines as links
                      'capacity': 'osmTGmod'}, # 'osmTGmod', 'ntc_acer' or 'thermal_acer'
    'comments': None}

etrago = Etrago(args, json_path=None)

etrago.network = import_network()

etrago.network.lopf(solver_name='gurobi')

#etrago.calc_results()

##results
gen = etrago.network.generators
gen_p = etrago.network.generators_t.p
gen_p_max_pu= etrago.network.generators_t.p_max_pu

gen_wind =gen.filter(regex='wind turbine', axis=0)
gen_wind_p =gen_p.filter(regex='wind turbine', axis=1)
gen_wind_p = gen_wind_p.T
gen_wind_p_max_pu = gen_p_max_pu.filter(regex='wind turbine', axis=1)
gen_wind_p_max_pu = gen_wind_p_max_pu.T

gen_turb = gen.filter(regex='gas turbine', axis=0)
gen_turb_p = gen_p.filter(regex='gas turbine', axis=1)
gen_turb_p = gen_turb_p.T
gen_turb_p_max_pu= gen_p_max_pu.filter(regex='gas turbine', axis=1)

load = etrago.network.loads
load_p = etrago.network.loads_t.p
load_p = load_p.T

#price = gen_p*gen['85ginal_cost']
gen.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen.csv' )
gen_p.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen_p.csv' )
gen_wind_p.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen_wind_p.csv' )
gen_turb_p.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen_turb_p.csv')
load_p.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/load_p.csv')
#price.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/price_m.csv')
#gen_turb_p_max_pu.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen_turb_p_max_pu.csv')
#gen_wind_p_max_pu.to_csv('/home/student/Documents/Masterthesis/Minimalbeispiel/Szenario_1/gen_wind_p_max_pu.csv')









