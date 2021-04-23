"""
This file can be used to check if eTraGo is installed succesfully. 
"""

from etrago.appl import run_etrago

args = {
    # Setup and Configuration:
    'db': 'local',  # database session
    'gridversion': 'v0.4.6',  # None for model_draft or Version number
    'method': { # Choose method and settings for optimization
        'type': 'lopf', # type of optimization, currently only 'lopf'
        'n_iter': 1, # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        'pyomo': True}, # set if pyomo is used for model building
    'pf_post_lopf': {
        'active': False, # choose if perform a pf after a lopf simulation
        'add_foreign_lopf': True, # keep results of lopf for foreign DC-links
        'q_allocation': 'p_nom'}, # allocate reactive power via 'p_nom' or 'p'
    'start_snapshot': 1,
    'end_snapshot': 2,
    'solver': 'glpk',  # glpk, cplex or gurobi
    'solver_options': {},
    'model_formulation': 'kirchhoff', # angles or kirchhoff
    'scn_name': 'NEP 2035',  # a scenario: Status Quo, NEP 2035, eGo 100
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
        'active': True, # choose if clustering is activated
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
    'branch_capacity_factor': {'HV': 0.5, 'eHV': 0.7},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'AC', # 'DC' for modeling foreign lines as links
                      'capacity': 'osmTGmod'}, # 'osmTGmod', 'ntc_acer' or 'thermal_acer'
    'comments': None}

print("eTraGo started")
etrago = run_etrago(args, json_path=None)
print("eTraGo finished")

etrago.plot_grid('line_loading', bus_colors='gen_dist', bus_sizes=0.00002)

etrago.session.close()
