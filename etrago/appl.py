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
from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario
import time
from egopowerflow.tools.plot import plot_line_loading, plot_stacked_gen, add_coordinates, curtailment 
from extras.utilities import load_shedding, data_manipulation_sh
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage

args = {'network_clustering':True,
        'db': 'oedb',
        'gridversion':'v0.2.10',
        'method': 'lopf',
        'start_h': 1,
        'end_h' : 2,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoPfHv',
        'outfile': '/home/ulf/file.lp',
        'solver': 'gurobi',
	'branch_capacity_factor': 1,
	'storage_extendable':False,
	'load_shedding':False
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
z = (y - x) / 60

network.model.write(args['outfile'], io_options={'symbolic_solver_labels':
                                                     True})

# make a line loading plot
plot_line_loading(network)

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# close session
session.close()
