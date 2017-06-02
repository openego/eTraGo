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
from egopowerflow.tools.io import NetworkScenario
import time
from egopowerflow.tools.plot import plot_line_loading, plot_stacked_gen, add_coordinates

args = {'network_clustering':False,
        'db': 'oedb',
        'gridversion': None,
        'method': 'lopf',
        'start_h': 1,
        'end_h' : 2,
        'scn_name': 'Status Quo',
        'ormcls_prefix': 'EgoGridPfHv',
        'outfile': '/home/ulf/file.lp',
        'solver': 'gurobi'}


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

# sh scenario
if args['scn_name'] == 'SH Status Quo':
    prep_sh_scenario(network)

# network clustering
if args['network_clustering']:
    network.generators.control="PV"
    busmap = busmap_from_psql(network, session, scn_name=scenario)
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

# same as before, limited to one specific bus
plot_stacked_gen(network, bus='24560', resolution='MW')

# close session
session.close()
