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
from egopowerflow.tools.plot import (plot_line_loading, plot_stacked_gen,
                                     add_coordinates, curtailment, gen_dist,
                                     storage_distribution)
from etrago.extras.utilities import load_shedding, data_manipulation_sh, results_to_csv, parallelisation
from etrago.cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage

args = {'network_clustering':False,
        'db': 'oedb', # db session
        'gridversion':None, #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'pf_post_lopf': True, #state whether you want to perform a pf after a lopf simulation
        'start_h': 2320,
        'end_h' : 2324,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoGridPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv'
        'lpfile': False, # state if and where you want to save pyomo's lp file: False or '/path/tofolder'
        'results': False , # state if and where you want to save results as csv: False or '/path/tofolder'
        'solver': 'gurobi', #glpk, cplex or gurobi
        'branch_capacity_factor': 1, #to globally extend or lower branch capacities
        'storage_extendable':True,
        'load_shedding':True,
        'generator_noise':False,
        'parallelisation':False}

def etrago(args):
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
        network.generators.marginal_cost[ network.generators.marginal_cost
         == 0] = abs(np.random.normal(0,0.00001,sum(network.generators.marginal_cost == 0)))

    if args['storage_extendable']:
        # set virtual storages to be extendable
        network.storage_units.p_nom_extendable = True
        # set virtual storage costs with regards to snapshot length
        network.storage_units.capital_cost = (network.storage_units.capital_cost /
        (8760//(args['end_h']-args['start_h']+1)))


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

    # parallisation
    if args['parallelisation']:
        parallelisation(network, start_h=args['start_h'], end_h=args['end_h'],group_size=1, solver_name=args['solver'])
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(scenario.timeindex, solver_name=args['solver'])
        y = time.time()
        z = (y - x) / 60 # z is time for lopf in minutes
    # start non-linear powerflow simulation
    elif args['method'] == 'pf':
        network.pf(scenario.timeindex)
    
    if args['pf_post_lopf']:
        network.pf(scenario.timeindex, use_seed=True)

    # write lpfile to path
    if not args['lpfile'] == False:
        network.model.write(args['lpfile'], io_options={'symbolic_solver_labels':
                                                     True})
    # write PyPSA results to csv to path
    if not args['results'] == False:
        results_to_csv(network, args['results'])

    return network


# execute etrago function
network = etrago(args)

# plots

# make a line loading plot
plot_line_loading(network)

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# plot to show extendable storages
storage_distribution(network)
prepf = network
pups = network

#For the PF, set the P to the optimised P
pups.generators_t.p_set = pups.generators_t.p_set.reindex(columns=pups.generators.index)
pups.generators_t.p_set = pups.generators_t.p

    #set all buses to PV, since we don't know what Q set points are
    #network.generators.control = "PV"

    #Need some PQ buses so that Jacobian doesn't break
    #f = network.generators[network.generators.bus == "24220"]
    #network.generators.loc[f.index,"control"] = "PQ"

    #Troubleshooting
contingency_factor=2
#network.loads_t.p_set['28314'] = network.loads_t.p_set['28314']*0.5
#network.loads_t.q_set['28314'] = network.loads_t.q_set['28314']*0.5
#network.transformers.x=network.transformers.x['22596']*0.1
pups.lines.s_nom = contingency_factor*pups.lines.s_nom
pups.transformers.s_nom = pups.transformers.s_nom*contingency_factor

    #network.generators_t.p_set = network.generators_t.p_set*0.9
    #network.loads_t.p_set = network.loads_t.p_set*0.9

    #network_pf.pf(network.snapshots)

    #calculate p line losses

pups.pf(pups.snapshots, use_seed=True)

import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,1)
fig.set_size_inches(6,6)
load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
load_distribution_q = network.loads_t.q_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
network.plot(bus_sizes=load_distribution,ax=ax,title="Load distribution")
network.plot(bus_colors='r',bus_sizes=load_distribution_q,ax=ax,title="q Load distribution")

plt.show()

network.buses_t.v_mag_pu.min().plot()
network.buses_t.v_mag_pu.max().plot()
plt.show()

# close session
#session.close()
