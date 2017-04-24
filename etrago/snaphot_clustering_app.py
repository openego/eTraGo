"""
"""

__copyright__ = "tba"
__license__ = "tba"
__author__ = "Simon Hilpert"

import os
import pandas as pd


from egopowerflow.tools.tools import oedb_session
from egopowerflow.tools.io import get_timerange, import_components, import_pq_sets,\
    add_source_types, create_powerflow_problem
from egopowerflow.tools.plot import add_coordinates

from egoio.db_tables.model_draft import EgoGridPfHvBus as Bus, EgoGridPfHvLine as Line, EgoGridPfHvGenerator as Generator, EgoGridPfHvLoad as Load,\
    EgoGridPfHvTransformer as Transformer, EgoGridPfHvTempResolution as TempResolution, EgoGridPfHvGeneratorPqSet as GeneratorPqSet,\
    EgoGridPfHvLoadPqSet as LoadPqSet, EgoGridPfHvSource as Source, EgoGridPfHvStorage as StorageUnit
    #, EgoGridPfHvStoragePqSet as Storage

from cluster.snapshot import update_data_frames, prepare_network, \
    linkage, fcluster, get_medoids
from pypsa.opf import network_lopf
import pyomo.environ as po
###############################################################################
def daily_bounds(network, snapshots):
    """ This will bound the storage level to 0.5 max_level every 24th hour.
    """
    if network.cluster:

        sus = network.storage_units

        network.model.period_ends = pd.DatetimeIndex(
                [i for i in network.snapshot_weightings.index[0::24]] +
                [network.snapshot_weightings.index[-1]])


        network.model.storages = sus.index
        def week_rule(m, s, p):
            return m.state_of_charge[s, p] == (sus.at[s, 'max_hours'] *
                                               0.5 * m.storage_p_nom[s])
        network.model.period_bound = po.Constraint(network.model.storages,
                                                   network.model.period_ends,
                                                   rule=week_rule)

######################## Functions to plot / store results ####################
def results_to_csv(network, path):
    """
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    network.export_to_csv_folder(path)
    data = pd.read_csv(os.path.join(path, 'network.csv'))
    data['time'] = network.results['Solver'].Time
    data.to_csv(os.path.join(path, 'network.csv'))

    if hasattr(network, 'Z'):
        file = [i for i in os.listdir(path.strip('0123456789')) if i=='Z.csv']
        if file:
           print('Z already calculated')
        else:
           network.Z.to_csv(path.strip('0123456789')+'/Z.csv', index=False)


def manipulate_storage_invest(network, costs=None, wacc=0.05, lifetime=15):
    # default: 4500 € / MW, high 300 €/MW
    crf = (1 / wacc) - (wacc / ((1 + wacc) ** lifetime))
    network.storage_units.capital_cost = costs / crf

def write_lpfile(network=None, path=None):
    network.model.write(path,
                        io_options={'symbolic_solver_labels':True})

def fix_storage_capacity(resultspath, n_clusters):
    path = resultspath.strip('daily')
    values = pd.read_csv(path + 'storage_capacity.csv')[n_clusters].values
    network.storage_units.p_nom_max = values
    network.storage_units.p_nom_min = values
    resultspath = 'compare-'+resultspath

    return resultspath

def run(network, path, write_results=False, n_clusters=None, how='daily',
        normed=False):
    """
    """
    # reduce storage costs due to clusters

    if n_clusters is not None:
        path = os.path.join(path, str(n_clusters))

        network.cluster = True

        # calculate clusters
        df, n_groups = prepare_network(network, how=how, normed=normed)

        Z = linkage(df, n_groups)

        network.Z = pd.DataFrame(Z)

        clusters = fcluster(df, Z, n_groups, n_clusters)

        medoids = get_medoids(clusters)

        update_data_frames(network, medoids)

        snapshots = network.snapshots

    else:
        network.cluster = False
        path = os.path.join(path, 'original')

    snapshots = network.snapshots

    # add coordinates to network nodes and make ready for map plotting
    network = add_coordinates(network)

    # add source names to generators
    add_source_types(session, network, table=Source)

    # start powerflow calculations
    network_lopf(network, snapshots, extra_functionality=daily_bounds,
                 solver_name='gurobi')

    #write_lpfile(network, path=os.path.join(path, "file.lp"))

    # write results to csv
    if write_results:
        results_to_csv(network, path)

    return network

###############################################################################
session = oedb_session('open_ego')

scenario = 'SH Status Quo'

# define relevant tables of generator table
pq_set_cols_1 = ['p_set']
pq_set_cols_2 = ['q_set']
p_max_pu = ['p_max_pu']
storage_sets = ['inflow'] # or: p_set, q_set, p_min_pu, p_max_pu, soc_set, inflow

# choose relevant parameters used in pf
temp_id_set = 1
start_h = 1
end_h = 5
# define investigated time range
timerange = get_timerange(session, temp_id_set, TempResolution, start_h, end_h)

# define relevant tables
tables = [Bus, Line, Generator, Load, Transformer, StorageUnit]

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

network.storage_units.p_nom_extendable = True
network.storage_units.p_min_pu_fixed = -1
network.storage_units.p_nom = 0
network.storage_units.cyclic_state_of_charge = True

###############################################################################
# Run scenarios .....
###############################################################################

how = 'daily'
clusters = [5] #[7] +  [i*7*2 for i in range(1,7)]
write_results = True

home = os.path.expanduser("~")
resultspath = os.path.join(home, 'snapshot-clustering-results', scenario)

# This will calculate the original problem
run(network=network.copy(), path=resultspath,
    write_results=write_results, n_clusters=None)

# This will claculate the aggregated problems
for c in clusters:
    path = os.path.join(resultspath, how)

    run(network=network.copy(), path=path,
        write_results=write_results, n_clusters=c,
        how=how, normed=False)

session.close()

