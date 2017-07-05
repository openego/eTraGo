"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.
Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line."""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "tba"

import numpy as np
np.random.seed()
from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario
import time
from egopowerflow.tools.plot import plot_line_loading, plot_stacked_gen, add_coordinates, curtailment, gen_dist, storage_distribution
from extras.utilities import load_shedding, data_manipulation_sh, results_to_csv
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage

args = {'network_clustering':False,
        'db': 'oedb', # db session
        'gridversion':None, #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'start_h': 2301,
        'end_h' : 2302,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv' 
        'outfile': '/path', # state if and where you want to save pyomo's lp file
        'results': '/path', # state if and where you want to save results as csv
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':True,
        'load_shedding':True,
        'generator_noise':False}


session = oedb_session(args['db'])

# additional arguments cfgpath, version, prefix
scenario = NetworkScenario(session,
                           version=args['gridversion'],
                           prefix=args['ormcls_prefix'],
                           method=args['method'],
                           start_h=args['start_h'],
                           end_h=args['end_h'],
                           scn_name=args['scn_name'])

network = scenario.build_network()

# add coordinates
network = add_coordinates(network)
  
if args['branch_capacity_factor']:
    network.lines.s_nom = network.lines.s_nom*args['branch_capacity_factor']
    network.transformers.s_nom = network.transformers.s_nom*args['branch_capacity_factor']


if args['generator_noise']:
    # add random noise to all generators with marginal_cost of 0. 
    network.generators.marginal_cost[ network.generators.marginal_cost == 0] = abs(np.random.normal(0,0.00001,sum(network.generators.marginal_cost == 0)))

if args['storage_extendable']:
    # set virtual storages to be extendable
    network.storage_units.p_nom_extendable = True
    # set virtual storage costs with regards to snapshot length 
    network.storage_units.capital_cost = network.storage_units.capital_cost / (8760//(args['end_h']-args['start_h']+1))

    
# for SH scenario run do data preperation:
if args['scn_name'] == 'SH Status Quo':
    data_manipulation_sh(network)

#load shedding in order to hunt infeasibilities
if args['load_shedding']:
	load_shedding(network)

# network clustering
if args['network_clustering']:
    network.generators.control="PV"
    busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
    network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

# start powerflow calculations
x = time.time()
network.lopf(scenario.timeindex, solver_name=args['solver'])
y = time.time()
z = (y - x) / 60 # z is time for lopf in minutes

# write results
network.model.write(args['outfile'], io_options={'symbolic_solver_labels':
                                                     True})
results_to_csv(network, args['results'])

# plots

# make a line loading plot
plot_line_loading(network)

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# plot to show extendable storages
storage_distribution(network)

# close session
session.close()
