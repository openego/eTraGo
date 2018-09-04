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


import datetime
import os
import os.path
import time
import numpy as np

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"


if 'READTHEDOCS' not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly
    from etrago.cluster.disaggregation import (
            MiniSolverDisaggregation,
            UniformDisaggregation)
    
    from etrago.cluster.networkclustering import (
        busmap_from_psql,
        cluster_on_extra_high_voltage,
        kmean_clustering)
    
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
        group_parallel_lines,
        add_missing_components,
        distribute_q,
        set_q_foreign_loads,
        clip_foreign,
        foreign_links,
        set_line_country_tags,
        crossborder_correction,
        ramp_limits)
    
    from etrago.tools.extendable import extendable, extension_preselection
    from etrago.cluster.snapshot import snapshot_clustering, daily_bounds
    from egoio.tools import db
    from sqlalchemy.orm import sessionmaker

args = {  # Setup and Configuration:
    'db': 'oedb',  # database session
    'gridversion': 'v0.4.4',  # None for model_draft or Version number
    'method': 'lopf',  # lopf or pf
    'pf_post_lopf': False,  # perform a pf after a lopf simulation
    'start_snapshot': 1,
    'end_snapshot': 2,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {'threads':4, 'method':2, 'BarHomogeneous':1,
         'NumericFocus': 3, 'BarConvTol':1.e-5,'FeasibilityTol':1.e-6, 'logFile':'gurobi_eTraGo.log'},  # {} for default or dict of solver options
    'scn_name': 'NEP 2035',  # a scenario: Status Quo, NEP 2035, eGo100
    # Scenario variations:
    'scn_extension': None,  # None or array of extension scenarios
    'scn_decommissioning':None, # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'results': ' ./results',  # save results as csv: False or /path/tofolder
    'export': False,  # export the results back to the oedb
    # Settings:
    'extendable': ['network', 'storages'],  # Array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'minimize_loading': False,
    'ramp_limits': True,
    'crossborder_correction': 'ntc', #state if you want to correct interconnector capacities. 'ntc' or 'thermal'
    # Clustering:
    'network_clustering_kmeans': 10,  # False or the value k for clustering
    'load_cluster': False,  # False or predefined busmap for k-means
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': 'uniform', # or None, 'mini' or 'uniform'
    'snapshot_clustering': False,  # False or the number of 'periods'
    # Simplifications:
    'parallelisation': False,  # run snapshots parallely.
    'skip_snapshots': False,
    'line_grouping': False,  # group lines parallel lines
    'branch_capacity_factor': 0.7,  # factor to change branch capacities
    'load_shedding': True, # meet the demand at very high cost
    'foreign_lines' : 'AC', # carrier of lines to/between foreign countries
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

   scn_extension : NoneType or list
       None,
       Choose extension-scenarios which will be added to the existing
       network container. Data of the extension scenarios are located in
       extension-tables (e.g. model_draft.ego_grid_pf_hv_extension_bus)
       with the prefix 'extension_'.
       Currently there are three overlay networks:
           'nep2035_confirmed' includes all planed new lines confirmed by the
           Bundesnetzagentur
           'nep2035_b2' includes all new lines planned by the
           Netzentwicklungsplan 2025 in scenario 2035 B2
           'BE_NO_NEP 2035' includes planned lines to Belgium and Norway and adds
           BE and NO as electrical neighbours

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

    extendable : list
        ['network', 'storages'],
        Choose components you want to optimize.
        Settings can be added in /tools/extendable.py.
        The most important possibilities:
            'network': set all lines, links and transformers extendable
            'transformers': set all transformers extendable
            'overlay_network': set all components of the 'scn_extension'
                               extendable
            'storages': allow to install extendable storages
                        (unlimited in size) at each grid node in order to meet
                        the flexibility demand.
            'network_preselection': set only preselected lines extendable,
                                    method is chosen in funcion call


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
    
    foreign_lines : str
        'AC'
        Choose transmission technology of foreign lines: 'AC' or 'DC'

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
    
    # Set q_sets of foreign loads
    network =  set_q_foreign_loads(network, cos_phi = 1)
    
    # Change transmission technology of foreign lines
    if args['foreign_lines'] == 'DC':
        foreign_links(network)

    # TEMPORARY vague adjustment due to transformer bug in data processing
    if args['gridversion'] == 'v0.2.11':
        network.transformers.x = network.transformers.x * 0.0001

    # set SOC at the beginning and end of the period to equal values
    network.storage_units.cyclic_state_of_charge = True

    # set extra_functionality to default
    extra_functionality = None
    
    # set disaggregated_network to default
    disaggregated_network = None

    # set clustering to default
    clustering = None
    
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

    # Branch loading minimization
    if args['minimize_loading']:
        extra_functionality = loading_minimization
    
    # scenario extensions 
    if args['scn_extension'] is not None:
        for i in range(len(args['scn_extension'])):
            network = extension(
                    network,
                    session,
                    version = args['gridversion'],
                    scn_extension=args['scn_extension'][i],
                    start_snapshot=args['start_snapshot'],
                    end_snapshot=args['end_snapshot'])
            
    # scenario decommissioning
    if args['scn_decommissioning'] is not None:
        network = decommissioning(
            network,
            session,
            version = args['gridversion'],
            scn_decommissioning=args['scn_decommissioning'])

    # Add missing lines in Munich and Stuttgart
    network =  add_missing_components(network)

    # investive optimization strategies 
    if args['extendable'] != []:
        network = extendable(
                    network,
                    args)
        network = convert_capital_costs(
            network, args['start_snapshot'], args['end_snapshot'])
    
    # skip snapshots
    if args['skip_snapshots']:
        network.snapshots = network.snapshots[::args['skip_snapshots']]
        network.snapshot_weightings = network.snapshot_weightings[
            ::args['skip_snapshots']] * args['skip_snapshots']
            
    # snapshot clustering
    if not args['snapshot_clustering'] is False:
        network = snapshot_clustering(
            network, how='daily', clusters=args['snapshot_clustering'])
        extra_functionality = daily_bounds  # daily_bounds or other constraint
        
    # set Branch capacity factor for lines and transformer
    if args['branch_capacity_factor']:
        network.lines.s_nom = network.lines.s_nom * \
            args['branch_capacity_factor']
        network.transformers.s_nom = network.transformers.s_nom * \
            args['branch_capacity_factor']

    # load shedding in order to hunt infeasibilities
    if args['load_shedding']:
        load_shedding(network)

    # ehv network clustering
    if args['network_clustering_ehv']:
        network.generators.control = "PV"
        busmap = busmap_from_psql(network, session, scn_name=args['scn_name'])
        network = cluster_on_extra_high_voltage(
            network, busmap, with_time=True)

    # k-mean clustering
    if not args['network_clustering_kmeans'] == False:
        clustering = kmean_clustering(network,
                n_clusters=args['network_clustering_kmeans'],
                load_cluster=args['load_cluster'],
                line_length_factor= 1,
                remove_stubs=False,
                use_reduced_coordinates=False,
                bus_weight_tocsv=None,
                bus_weight_fromcsv=None)
        disaggregated_network = (
                network.copy() if args.get('disaggregation') else None)
        network = clustering.network.copy()

    # preselection of extendable lines
    if 'network_preselection' in args['extendable']:
        extension_preselection(network, args, 'snapshot_clustering', 2)
        
    # skip snapshots
    if args['skip_snapshots']:
        network.snapshot_weightings=network.snapshot_weightings*args['skip_snapshots']

    if args['crossborder_correction']:
        set_line_country_tags(network)
        crossborder_correction(network, args['crossborder_correction'],
                               args['branch_capacity_factor'])
    
    if args['ramp_limits']:
        ramp_limits(network)                 

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
        x = time.time()
        pf_solution = pf_post_lopf(network, 
                                   args['foreign_lines'], 
                                   add_foreign_lopf=True)
        y = time.time()
        z = (y - x) / 60
        print("Time for PF [min]:", round(z, 2))
        calc_line_losses(network)
        network = distribute_q(network, allocation = 'p_nom')
        
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

    if clustering:
        disagg = args.get('disaggregation')
        skip = () if args['pf_post_lopf'] else ('q',)
        t = time.time()
        if disagg:
            if disagg == 'mini':
                disaggregation = MiniSolverDisaggregation(
                        disaggregated_network,
                        network,
                        clustering,
                        skip=skip)
            elif disagg == 'uniform':
                disaggregation = UniformDisaggregation(disaggregated_network,
                                                       network,
                                                       clustering,
                                                       skip=skip)

            else:
                raise Exception('Invalid disaggregation command: ' + disagg)

            disaggregation.execute(scenario, solver=args['solver'])
            # temporal bug fix for solar generator which ar during night time
            # nan instead of 0            
            disaggregated_network.generators_t.p.fillna(0, inplace=True)
            disaggregated_network.generators_t.q.fillna(0, inplace=True)
            
            disaggregated_network.results = network.results
            print("Time for overall desaggregation [min]: {:.2}"
                .format((time.time() - t) / 60))

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
        # If it is set to 'True' the result set will be saved
        # to the versioned grid schema eventually apart from
        # being saved to the model_draft.
        # ONLY set to True if you know what you are doing.
        results_to_oedb(
            session,
            network,
            dict([("disaggregated_results", False)] + list(args.items())),
            grid='hv',
            safe_results=safe_results)
        if disaggregated_network:
            results_to_oedb(
                session,
                disaggregated_network,
                dict([("disaggregated_results", True)] + list(args.items())),
                grid='hv',
                safe_results=safe_results)

    # write PyPSA results to csv to path
    if not args['results'] is False:
        if not args['pf_post_lopf']:
            results_to_csv(network, args)
        else:
            results_to_csv(network, args,pf_solution = pf_solution)

        if disaggregated_network:
            results_to_csv(
                    disaggregated_network,
                    {k: os.path.join(v, 'disaggregated')
                        if k == 'results' else v
                        for k, v in args.items()})

    # close session
    # session.close()

    return network, disaggregated_network


if __name__ == '__main__':
    # execute etrago function
    print(datetime.datetime.now())
    network, disaggregated_network = etrago(args)
    print(datetime.datetime.now())
    # plots
    # make a line loading plot
    # plot_line_loading(network)
    # plot stacked sum of nominal power for each generator type and timestep
    # plot_stacked_gen(network, resolution="MW")
    # plot to show extendable storages
    # storage_distribution(network)
    # extension_overlay_network(network)
