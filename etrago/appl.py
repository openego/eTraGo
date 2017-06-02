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

session = oedb_session('oedb')
# additional arguments cfgpath, version, prefix
scenario = NetworkScenario(session,version='v0.2.10', prefix='EgoPfHv', method='lopf', start_h=1, end_h=2,
                           scn_name='Status Quo')
network = scenario.build_network()

# add coordinates
network = add_coordinates(network)

# data preparation
network.storage_units.p_nom_extendable = True
network.lines.s_nom = network.lines.s_nom*1.5
network.transformers.s_nom = network.transformers.s_nom*1.5

# start powerflow calculations
x = time.time()
network.lopf(scenario.timeindex, solver_name='gurobi')
y = time.time()
z = (y - x) / 60

network.model.write('/home/ulf/file.lp', io_options={'symbolic_solver_labels':
                                                     True})

# make a line loading plot
plot_line_loading(network)

# plot stacked sum of nominal power for each generator type and timestep
plot_stacked_gen(network, resolution="MW")

# same as before, limited to one specific bus
plot_stacked_gen(network, bus='24560', resolution='MW')

# close session
session.close()
