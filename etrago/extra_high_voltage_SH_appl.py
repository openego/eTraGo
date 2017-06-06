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
from egopowerflow.tools.io import get_timerange, import_components, import_pq_sets,\
    add_source_types, create_powerflow_problem
from egopowerflow.tools.plot import add_coordinates, plot_line_loading,\
     plot_stacked_gen, curtailment, gen_dist
from egoio.db_tables.model_draft import EgoGridPfHvBus as Bus, EgoGridPfHvLine as Line, EgoGridPfHvGenerator as Generator, EgoGridPfHvLoad as Load,\
    EgoGridPfHvTransformer as Transformer, EgoGridPfHvTempResolution as TempResolution, EgoGridPfHvGeneratorPqSet as GeneratorPqSet,\
    EgoGridPfHvLoadPqSet as LoadPqSet, EgoGridPfHvSource as Source, EgoGridPfHvStorage as Storage
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage
from extras.utilities import load_shedding


session = oedb_session()

scenario = 'SH Status Quo'

# define relevant tables of generator table
pq_set_cols_1 = ['p_set']
pq_set_cols_2 = ['q_set']
p_max_pu = ['p_max_pu']

# choose relevant parameters used in pf
temp_id_set = 1
start_h = 1
end_h = 168

# define investigated time range
timerange = get_timerange(session, temp_id_set, TempResolution, start_h, end_h)

# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer, Storage]

# get components from database tables
components = import_components(tables, session, scenario)

# create PyPSA powerflow problem
network, snapshots = create_powerflow_problem(timerange, components)

# import pq-set tables to pypsa network (p_set for generators and loads)
pq_object = [GeneratorPqSet, LoadPqSet]
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=pq_object,
                         timerange=timerange,
                         scenario=scenario,
                         columns=pq_set_cols_1,
                         start_h=start_h,
                         end_h=end_h)

# import pq-set table to pypsa network (q_set for loads)
network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=[LoadPqSet],
                         timerange=timerange,
                         scenario=scenario,
                         columns=pq_set_cols_2,
                         start_h=start_h,
                         end_h=end_h)

network = import_pq_sets(session=session,
                         network=network,
                         pq_tables=[GeneratorPqSet],
                         timerange=timerange,
                         scenario=scenario,
                         columns=p_max_pu,
                         start_h=start_h,
                         end_h=end_h)

## import time data for storages:
#network = import_pq_sets(session=session,
#                         network=network,
#                         pq_tables=[StoragePqSet],
#                         timerange=timerange,
#                         scenario=scenario,
#                         columns=storage_sets,
#                         start_h=start_h,
#                         end_h=end_h)

# add coordinates to network nodes and make ready for map plotting
network = add_coordinates(network)

# add source names to generators
add_source_types(session, network, table=Source)

#add connection from Luebeck to Siems
network.add("Bus", "Siems220",carrier='AC', v_nom=220, x=10.760835, y=53.909745)
network.add("Transformer", "Siems220_380", bus0="25536", bus1="Siems220", x=1.29960, tap_ratio=1)
network.add("Line","LuebeckSiems", bus0="26387",bus1="Siems220", x=0.0001, s_nom=1600)


#network.lines.s_nom = network.lines.s_nom*1.5
#network.transformers.s_nom = network.transformers.s_nom*1.5

# set virtual storages to be extendable
network.storage_units.p_nom_extendable = True

# set virtual storage costs with regards to snapshot length
network.storage_units.capital_cost = network.storage_units.capital_cost / 52

network.generators.control="PV"

busmap = busmap_from_psql(network, session, scn_name=scenario)

network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

# add random noise to all generators with marginal_cost of 0. 
network.generators.marginal_cost[ network.generators.marginal_cost == 0] = abs(np.random.normal(0,0.00001,sum(network.generators.marginal_cost == 0)))

#load shedding in order to hunt infeasibilities
#load_shedding(network)

# start powerflow calculations
network.lopf(snapshots, solver_name='gurobi')

network.model.write('/home/ulf/file.lp', io_options={'symbolic_solver_labels':True})

# make a line loading plot
plot_line_loading(network)

#plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# same as before, limited to one specific bus
plot_stacked_gen(network, bus='24560', resolution='MW')

# close session
session.close()
