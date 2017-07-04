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
import sys
sys.path.append('/home/mario-arbeit/Dokumente/Open-eGo/eTraGo/etrago/')

from math import sqrt    

from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import NetworkScenario
import time
from egopowerflow.tools.plot import plot_line_loading, plot_stacked_gen, add_coordinates, curtailment, gen_dist 
from extras.utilities import load_shedding, data_manipulation_sh
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage
import progressbar
from oemof.db import cfg

# only load config file
cfg.load_config('/home/mario-arbeit/Dokumente/config.ini')

args = {'network_clustering':False,
        'db': 'oedb', # db session
        'gridversion': 'v0.2.10', #None for model_draft or Version number (e.g. v0.2.10) for grid schema
        'method': 'lopf', # lopf or pf
        'start_h': 2009,
        'end_h' : 2010,
        'scn_name': 'SH Status Quo',
        'ormcls_prefix': 'EgoPfHv', #if gridversion:'version-number' then 'EgoPfHv', if gridversion:None then 'EgoGridPfHv' 
        'outfile': '/home/mario-arbeit/Dokumente/lopf/file.lp', # state if and where you want to safe pyomo's lp file
        'solver': 'gurobi', #glpk or gurobi
	  'branch_capacity_factor': 0.1, #to globally extend or lower branch capacities
	  'storage_extendable':False,
        'line_extendable':True,
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

if args['line_extendable']:
    first_scenario = NetworkScenario(session,
                           version=args['gridversion'],
                           prefix=args['ormcls_prefix'],
                           method=args['method'],
                           start_h=args['start_h'],
                           end_h=args['end_h'],
                           scn_name=args['scn_name'])
    
    first_network = first_scenario.build_network()
    first_network = add_coordinates(first_network)
      
    first_network.lines.s_nom = first_network.lines.s_nom * args['branch_capacity_factor']
    # add random noise to all generators with marginal_cost of 0. 
    first_network.generators.marginal_cost[first_network.generators.marginal_cost == 0] = abs(np.random.normal(0,0.00001,sum(first_network.generators.marginal_cost == 0)))
    
    network.generators.marginal_cost = first_network.generators.marginal_cost
    
    if args['scn_name'] == 'SH Status Quo':
        data_manipulation_sh(first_network)
    
    load_shedding(first_network)    
    
    if args['network_clustering']:
        first_network.generators.control="PV"
        first_busmap = busmap_from_psql(first_network, session, scn_name=args['scn_name'])
        first_network = cluster_on_extra_high_voltage(first_network, first_busmap, with_time=True)
    
    # start powerflow calculations
    x = time.time()
    first_network.lopf(first_scenario.timeindex, solver_name=args['solver'])
    y = time.time()
    z = (y - x) / 60  
    
   # network = first_network
    bar = progressbar.ProgressBar()
    cap_cost = []
    line_keys = []
    timestep=0
    i = 0
    p = first_network.lines_t.p0.loc[first_network.snapshots[timestep]]
    q = first_network.lines_t.q0.loc[first_network.snapshots[timestep]]
    keys = first_network.lines_t.p0.keys()
    gesamt = len(p)*(args['end_h']-args['start_h'])
    k = 0
    while(i<len(p)):
        bar.update(((k)/gesamt)*100)
        timestep=0
        percent_max = ['NaN',0]
        while(timestep < (args['end_h']-args['start_h'])):
            bar.update(((k)/gesamt)*100)
            p = first_network.lines_t.p0.loc[first_network.snapshots[timestep]]
            q = first_network.lines_t.q0.loc[first_network.snapshots[timestep]]
            if(i<len(q)):
                percent = (sqrt(p[i]**2 + q[i]**2)/(first_network.lines.s_nom[i])) * 100 
            else:
                percent = (sqrt(p[i]**2 + 0**2)/(first_network.lines.s_nom[i])) * 100
            
            if(percent>percent_max[1]):
                percent_max[0] = keys[i]
                percent_max[1] = percent
            
            k +=1
            timestep+=1
            
            
        if(percent_max[1] > 80):
            network.lines.s_nom_extendable[percent_max[0]] = True
            network.lines.s_nom_min= network.lines.s_nom[percent_max[0]] 
            network.lines.s_nom_max= network.lines.s_nom[percent_max[0]]*5
            
            U = network.buses.v_nom[network.lines.bus0[percent_max[0]]]
            l = network.lines.length[percent_max[0]]
            x = network.lines.x[percent_max[0]]
            r = network.lines.r[percent_max[0]]
            Z = sqrt(x**2 + r**2)
            S = network.lines.s_nom[percent_max[0]]
            
            cap_cost.append((((((U*1e3)**2)*l)/(Z*(S*1e6+1e6))-l)*50))        
                        
            
            network.lines.capital_cost[percent_max[0]] = 0#cap_cost[len(cap_cost)-1]
            line_keys.append(percent_max[0])
            
        
        i+=1
    
    timestep = 0
    
    # make a line loading plot
    plot_line_loading(first_network, filename = 'vorher.png')
    
    # plot stacked sum of nominal power for each generator type and timestep
    plot_stacked_gen(first_network, resolution="MW")
    
    
    print("line_extendable finished")    
    
    # set virtual lines to be extendable 
#    from pyomo.environ import Constraint    
#    timestep = 0
    
#    def extra_functionality(network,snapshots):
#        network.lines.s_nom = \
#        Constraint(((network.lines.p0 ** 2 + network.lines.q0 ** 2).apply(sqrt)) \
#        <= (0.7 * (network.lines.s_nom)))
#
#    network.lines.s_nom_extendable = True 
#    network.lines.s_nom_min = network.lines.s_nom
#    network.lines.s_nom_max = network.lines.s_nom * 1.5
#else:
#    extra_functionality = None
    
network.lines.s_nom = network.lines.s_nom * args['branch_capacity_factor']    
    
# start powerflow calculations
x = time.time()
network.lopf(scenario.timeindex, solver_name=args['solver'])
y = time.time()
z = (y - x) / 60

network.model.write(args['outfile'], io_options={'symbolic_solver_labels':
                                                     True})



# make a line loading plot
plot_line_loading(network, filename = 'nachher.png')

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")


# close session
session.close()