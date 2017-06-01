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


from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import get_timerange, import_components, import_pq_sets,\
    add_source_types, create_powerflow_problem
from egopowerflow.tools.plot import add_coordinates, plot_line_loading,\
     plot_stacked_gen
from egoio.db_tables.model_draft import EgoGridPfHvBus as Bus, EgoGridPfHvLine as Line, EgoGridPfHvGenerator as Generator, EgoGridPfHvLoad as Load,\
    EgoGridPfHvTransformer as Transformer, EgoGridPfHvTempResolution as TempResolution, EgoGridPfHvGeneratorPqSet as GeneratorPqSet,\
    EgoGridPfHvLoadPqSet as LoadPqSet, EgoGridPfHvSource as Source
from cluster.networkclustering import busmap_from_psql, cluster_on_extra_high_voltage
from pypsa.networkclustering import busmap_by_kmeans, get_clustering_from_busmap
import pandas as pd
import numpy as np

session = oedb_session('grids_test')

scenario = 'SH Status Quo'

# define relevant tables of generator table
pq_set_cols_1 = ['p_set']
pq_set_cols_2 = ['q_set']
p_max_pu = ['p_max_pu']

# choose relevant parameters used in pf
temp_id_set = 1
start_h = 2300
end_h = 2301

# define investigated time range
timerange = get_timerange(session, temp_id_set, TempResolution, start_h, end_h)

# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer]

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
network.add("Transformer", "Siems220_380", bus0="25536", bus1="Siems220", x=1.29960, tap_ratio=1, s_nom=1600)
network.add("Line","LuebeckSiems", bus0="26387",bus1="Siems220", x=0.0001, s_nom=1600)


#network.lines.s_nom = network.lines.s_nom*1.5
#network.transformers.s_nom = network.transformers.s_nom*3

network.generators.control="PV"

network.buses['v_nom'] = 380.

# TODO adjust the x of the lines which are not 380. problem our lines have no v_nom. this is implicitly defined by the connected buses. Generally it should look something like the following:
#lines_v_nom_b = network.lines.v_nom != 380
#network.lines.loc[lines_v_nom_b, 'x'] *= (380./network.lines.loc[lines_v_nom_b, 'v_nom'])**2
#network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.


trafo_index = network.transformers.index

network.import_components_from_dataframe(
    network.transformers.loc[:,['bus0','bus1','x','s_nom']]
    .assign(x=0.1*380**2/2000)
    .set_index('T' + trafo_index),
    'Line'
)

network.transformers.drop(trafo_index, inplace=True)
for attr in network.transformers_t:
    network.transformers_t[attr] = network.transformers_t[attr].reindex(columns=[])

busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1, len(network.buses)), index=network.buses.index) , n_clusters= 10) 


clustering = get_clustering_from_busmap(network, busmap)
network = clustering.network
#network = cluster_on_extra_high_voltage(network, busmap, with_time=True)


# start powerflow calculat#ions
network.lopf(snapshots, solver_name='gurobi')

#network.model.write('/home/ulf/file.lp', io_options={'symbolic_solver_labels':True})

# make a line loading plot
plot_line_loading(network)

#plot stacked sum of nominal power for each generator type and timestep
#plot_stacked_gen(network, resolution="MW")


# close session
#session.close()
