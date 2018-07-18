# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
This is the application file for the tool eTraGo.
Define your connection parameters and power flow settings before executing
the function etrago.
"""

import numpy as np
from numpy import genfromtxt
import time
import datetime
import os

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"


if 'READTHEDOCS' not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly
    from etrago.tools.io import (
        NetworkScenario,
        results_to_oedb,
        extension,
        decommissioning)
    from etrago.tools.plot import (
        plot_line_loading,
        plot_stacked_gen,
        add_coordinates,
        curtailment,
        gen_dist,
        storage_distribution,
        storage_expansion,
        extension_overlay_network,
        nodal_gen_dispatch)

    from etrago.tools.utilities import (
        load_shedding,
        data_manipulation_sh,
        convert_capital_costs,
        results_to_csv,
        parallelisation,
        pf_post_lopf,
        loading_minimization,
        calc_line_losses,
        group_parallel_lines)
    from etrago.tools.extendable import extendable
    from etrago.cluster.networkclustering import (
        busmap_from_psql, cluster_on_extra_high_voltage, kmean_clustering)
    from etrago.cluster.snapshot import snapshot_clustering, daily_bounds
    from egoio.tools import db
    from sqlalchemy.orm import sessionmaker

args = {  # Setup and Configuration:
    'db': 'oedb',  # database session
    'gridversion': 'v0.4.2',  # None for model_draft or Version number
    'method': 'lopf',  # lopf or pf
    'pf_post_lopf': False,  # perform a pf after a lopf simulation
    'start_snapshot': 1,
    'end_snapshot': 2,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {},  # {} for default or dict of solver options
    'scn_name': 'NEP 2035',  # a scenario: Status Quo, NEP 2035, eGo100
    # Scenario variations:
    'scn_extension': None,  # None or extension scenario
    'scn_decommissioning': None,  # None or decommissioning scenario
    'add_Belgium_Norway': False,  # add Belgium and Norway
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'results': False,  # save results as csv: False or /path/tofolder
    'export': False,  # export the results back to the oedb
    # Settings:
    'extendable': None,  # None or array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'minimize_loading': False,
    # Clustering:
    'network_clustering_kmeans': False,  # False or the value k for clustering
    'load_cluster': False,  # False or predefined busmap for k-means
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'snapshot_clustering': False,  # False or the number of 'periods'
    # Simplifications:
    'parallelisation': False,  # run snapshots parallely.
    'skip_snapshots': False,
    'line_grouping': False,  # group lines parallel lines
    'branch_capacity_factor': 0.7,  # factor to change branch capacities
    'load_shedding': False,  # meet the demand at very high cost
    'comments': None}


def etrago(args):
    """The etrago function works with following arguments:


    Parameters
    ----------

    db : str
        ``'oedb'``,
        Name of Database session setting stored in *config.ini* of *.egoio*

    gridversion : NoneType or str
        ``'v0.2.11'``,
        Name of the data version number of oedb: state ``'None'`` for
        model_draft (sand-box) or an explicit version number
        (e.g. 'v0.2.10') for the grid schema.

    method : str
        ``'lopf'``,
        Choose between a non-linear power flow ('pf') or
        a linear optimal power flow ('lopf').

    pf_post_lopf : bool
        False,
        Option to run a non-linear power flow (pf) directly after the
        linear optimal power flow (and thus the dispatch) has finished.

    start_snapshot : int
        1,
        Start hour of the scenario year to be calculated.

    end_snapshot : int
        2,
        End hour of the scenario year to be calculated.

    solver : str
        'glpk',
        Choose your preferred solver. Current options: 'glpk' (open-source),
        'cplex' or 'gurobi'.

    scn_name : str
        'Status Quo',
        Choose your scenario. Currently, there are three different
        scenarios: 'Status Quo', 'NEP 2035', 'eGo100'. If you do not
        want to use the full German dataset, you can use the excerpt of
        Schleswig-Holstein by adding the acronym SH to the scenario
        name (e.g. 'SH Status Quo').

   scn_extension : str
       None,
       Choose an extension-scenario which will be added to the existing
       network container. Data of the extension scenarios are located in
       extension-tables (e.g. model_draft.ego_grid_pf_hv_extension_bus)
       with the prefix 'extension_'.
       Currently there are two overlay networks:
           'nep2035_confirmed' includes all planed new lines confirmed by the
           Bundesnetzagentur
           'nep2035_b2' includes all new lines planned by the
           Netzentwicklungsplan 2025 in scenario 2035 B2

    scn_decommissioning : str
        None,
        Choose an extra scenario which includes lines you want to decommise
        from the existing network. Data of the decommissioning scenarios are
        located in extension-tables
        (e.g. model_draft.ego_grid_pf_hv_extension_bus) with the prefix
        'decommissioning_'.
        Currently, there are two decommissioning_scenarios which are linked to
        extension-scenarios:
            'nep2035_confirmed' includes all lines that will be replaced in
            confirmed projects
            'nep2035_b2' includes all lines that will be replaced in
            NEP-scenario 2035 B2

    add_Belgium_Norway : bool
        False,
        State if you want to add Belgium and Norway as electrical neighbours.
        Currently, generation and load always refer to scenario 'NEP 2035'.

    lpfile : obj
        False,
        State if and where you want to save pyomo's lp file. Options:
        False or '/path/tofolder'.import numpy as np

    results : obj
        False,
        State if and where you want to save results as csv files.Options:
        False or '/path/tofolder'.

    export : bool
        False,
        State if you want to export the results of your calculation
        back to the database.

    extendable : NoneType or list
        ['network', 'storages'],
        Choose None or which components you want to optimize.
        Settings can be added in /tools/extendable.py.
        The most important possibilities:
            'network': set all lines, links and transformers extendable
            'transformers': set all transformers extendable
            'overlay_network': set all components of the 'scn_extension'
                               extendable
            'storages': allow to install extendable storages
                        (unlimited in size) at each grid node in order to meet
                        the flexibility demand.


    generator_noise : bool or int
        State if you want to apply a small random noise to the marginal costs
        of each generator in order to prevent an optima plateau. To reproduce
        a noise, choose the same integer (seed number).

    minimize_loading : bool
        False,
        ...

    network_clustering_kmeans : bool or int
        False,
        State if you want to apply a clustering of all network buses down to
        only ``'k'`` buses. The weighting takes place considering generation
        and load
        at each node. If so, state the number of k you want to apply. Otherwise
        put False. This function doesn't work together with
        ``'line_grouping = True'``.

    load_cluster : bool or obj
        state if you want to load cluster coordinates from a previous run:
        False or /path/tofile (filename similar to ./cluster_coord_k_n_result).

    network_clustering_ehv : bool
        False,
        Choose if you want to cluster the full HV/EHV dataset down to only the
        EHV buses. In that case, all HV buses are assigned to their closest EHV
        sub-station, taking into account the shortest distance on power lines.

    snapshot_clustering : bool or int
        False,
        State if you want to cluster the snapshots and run the optimization
        only on a subset of snapshot periods. The int value defines the number
        of periods (i.e. days) which will be clustered to.
        Move to PyPSA branch:features/snapshot_clustering

    parallelisation : bool
        False,
        Choose if you want to calculate a certain number of snapshots in
        parallel. If yes, define the respective amount in the if-clause
        execution below. Otherwise state False here.

    line_grouping : bool
        True,
        State if you want to group lines that connect the same two buses
        into one system.

    branch_capacity_factor : numeric
        1,
        Add a factor here if you want to globally change line capacities
        (e.g. to "consider" an (n-1) criterion or for debugging purposes).

    load_shedding : bool
        False,
        State here if you want to make use of the load shedding function which
        is helpful when debugging: a very expensive generator is set to each
        bus and meets the demand when regular
        generators cannot do so.

    comments : str
        None

    Returns
    -------
    network : `pandas.DataFrame<dataframe>`
        eTraGo result network based on `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_


    """
    conn = db.connection(section=args['db'])
    Session = sessionmaker(bind=conn)
    session = Session()

    # additional arguments cfgpath, version, prefix
    if args['gridversion'] is None:
        args['ormcls_prefix'] = 'EgoGridPfHv'
    else:
        args['ormcls_prefix'] = 'EgoPfHv'

    scenario = NetworkScenario(session,
                               version=args['gridversion'],
                               prefix=args['ormcls_prefix'],
                               method=args['method'],
                               start_snapshot=args['start_snapshot'],
                               end_snapshot=args['end_snapshot'],
                               scn_name=args['scn_name'])

    network = scenario.build_network()

    # add coordinates
    network = add_coordinates(network)

    # TEMPORARY vague adjustment due to transformer bug in data processing
    if args['gridversion'] == 'v0.2.11':
        network.transformers.x = network.transformers.x * 0.0001

    # set SOC at the beginning and end of the period to equal values
    network.storage_units.cyclic_state_of_charge = True

    # set extra_functionality to default
    extra_functionality = None

    if args['generator_noise'] is not False:
        # add random noise to all generators
        s = np.random.RandomState(args['generator_noise'])
        network.generators.marginal_cost += \
            abs(s.normal(0, 0.001, len(network.generators.marginal_cost)))

    # for SH scenario run do data preperation:
    if (args['scn_name'] == 'SH Status Quo' or
            args['scn_name'] == 'SH NEP 2035'):
        data_manipulation_sh(network)

    # grouping of parallel lines
    if args['line_grouping']:
        group_parallel_lines(network)

    # network clustering
    if args['network_clustering_ehv']:
        network.generators.control = "PV"
        busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
        network = cluster_on_extra_high_voltage(
            network, busmap, with_time=True)

    # k-mean clustering
    if not args['network_clustering_kmeans'] is False:
        network = kmean_clustering(
            network,
            n_clusters=args['network_clustering_kmeans'],
            load_cluster=args['load_cluster'],
            line_length_factor=1,
            remove_stubs=False,
            use_reduced_coordinates=False,
            bus_weight_tocsv=None,
            bus_weight_fromcsv=None)

    # Branch loading minimization
    if args['minimize_loading']:
        extra_functionality = loading_minimization

    if args['skip_snapshots']:
        network.snapshots = network.snapshots[::args['skip_snapshots']]
        network.snapshot_weightings = network.snapshot_weightings[
            ::args['skip_snapshots']] * args['skip_snapshots']

    if args['scn_extension'] is not None:
        network = extension(
            network,
            session,
            scn_extension=args['scn_extension'],
            start_snapshot=args['start_snapshot'],
            end_snapshot=args['end_snapshot'],
            k_mean_clustering=args['network_clustering_kmeans'])

    if args['scn_decommissioning'] is not None:
        network = decommissioning(
            network,
            session,
            scn_decommissioning=args['scn_decommissioning'],
            k_mean_clustering=args['network_clustering_kmeans'])

    if args['add_Belgium_Norway']:
        network = extension(
            network,
            session,
            scn_extension='BE_NO_NEP 2035',
            start_snapshot=args['start_snapshot'],
            end_snapshot=args['end_snapshot'],
            k_mean_clustering=args['network_clustering_kmeans'])

    if args['extendable'] is not None:
        network = extendable(
            network,
            args['extendable'],
            args['scn_extension'])
        network = convert_capital_costs(
            network, args['start_snapshot'], args['end_snapshot'])

    if args['branch_capacity_factor']:
        network.lines.s_nom = network.lines.s_nom * \
            args['branch_capacity_factor']
        network.transformers.s_nom = network.transformers.s_nom * \
            args['branch_capacity_factor']

    # load shedding in order to hunt infeasibilities
    if args['load_shedding']:
        load_shedding(network)

    # snapshot clustering
    if not args['snapshot_clustering'] is False:
        network = snapshot_clustering(
            network, how='daily', clusters=args['snapshot_clustering'])
        extra_functionality = daily_bounds  # daily_bounds or other constraint

    # parallisation
    if args['parallelisation']:
        parallelisation(
            network,
            start_snapshot=args['start_snapshot'],
            end_snapshot=args['end_snapshot'],
            group_size=1,
            solver_name=args['solver'],
            solver_options=args['solver_options'],
            extra_functionality=extra_functionality)
    # start linear optimal powerflow calculations
    elif args['method'] == 'lopf':
        x = time.time()
        network.lopf(
            network.snapshots,
            solver_name=args['solver'],
            solver_options=args['solver_options'],
            extra_functionality=extra_functionality)
        y = time.time()
        z = (y - x) / 60
        # z is time for lopf in minutes
        print("Time for LOPF [min]:", round(z, 2))

        # start non-linear powerflow simulation
    elif args['method'] is 'pf':
        network.pf(scenario.timeindex)
        # calc_line_losses(network)

    if args['pf_post_lopf']:
        pf_post_lopf(network, scenario)
        calc_line_losses(network)

    # provide storage installation costs
    if sum(network.storage_units.p_nom_opt) != 0:
        installed_storages = \
            network.storage_units[network.storage_units.p_nom_opt != 0]
        storage_costs = sum(
            installed_storages.capital_cost *
            installed_storages.p_nom_opt)
        print(
            "Investment costs for all storages in selected snapshots [EUR]:",
            round(
                storage_costs,
                2))

    # write lpfile to path
    if not args['lpfile'] is False:
        network.model.write(
            args['lpfile'], io_options={
                'symbolic_solver_labels': True})
    # write PyPSA results back to database
    if args['export']:
        username = str(conn.url).split('//')[1].split(':')[0]
        args['user_name'] = username
        safe_results = False  # default is False.
        # If it is set to 'True' the result set will be safed
        # to the versioned grid schema eventually apart from
        # being saved to the model_draft.
        # ONLY set to True if you know what you are doing.
        results_to_oedb(
            session,
            network,
            args,
            grid='hv',
            safe_results=safe_results)

    # write PyPSA results to csv to path
    if not args['results'] is False:
        results_to_csv(network, args['results'])

    # close session
    # session.close()

    return network


if __name__ == '__main__':
    # execute etrago function
    print(datetime.datetime.now())
    network = etrago(args)
    print(datetime.datetime.now())
    # plots
    # make a line loading plot
    # plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    # plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    # storage_distribution(network)
    # extension_overlay_network(network)
