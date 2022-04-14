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

import sys
import datetime
import os
import os.path
import numpy as np
import pandas as pd

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, lukasol, wolfbunke, mariusves, s3pp"


if 'READTHEDOCS' not in os.environ:
    # Sphinx does not run this code.
    # Do not import internal packages directly

    from etrago import Etrago

args = {
    # Setup and Configuration:
    'db': 'egon-data-local',  # database session 
    'gridversion': None,  # None for model_draft or Version number
    'method': { # Choose method and settings for optimization
        'type': 'lopf', # type of optimization, currently only 'lopf'
        'n_iter': 5, # abort criterion of iterative optimization, 'n_iter' or 'threshold'
        'pyomo': True}, # set if pyomo is used for model building
    'pf_post_lopf': {
        'active': False, # choose if perform a pf after a lopf simulation
        'add_foreign_lopf': True, # keep results of lopf for foreign DC-links
        'q_allocation': 'p_nom'}, # allocate reactive power via 'p_nom' or 'p'
    'start_snapshot': 1,
    'end_snapshot': 8760,
    'solver': 'gurobi',  # glpk, cplex or gurobi
    'solver_options': {},
    'model_formulation': 'kirchhoff', # angles or kirchhoff
    'scn_name': 'eGon2035',  # a scenario: eGon2035 or eGon100RE
    # Scenario variations:
    'scn_extension': None,  # None or array of extension scenarios
    'scn_decommissioning': None,  # None or decommissioning scenario
    # Export options:
    'lpfile': False,  # save pyomo's lp file: False or /path/tofolder
    'csv_export': 'results',  # save results as csv: False or /path/tofolder
    # Settings:
    'extendable': ['as_in_db'],  # Array of components to optimize
    'generator_noise': 789456,  # apply generator noise, False or seed number
    'extra_functionality':{},  # Choose function name or {}
    # Clustering:
    'network_clustering_kmeans': {
        'active': True, # choose if clustering is activated
        'n_clusters': 30, # number of resulting nodes
        'n_clusters_gas': 30, # number of resulting nodes
        'kmeans_busmap': False, # False or path/to/busmap.csv
        'kmeans_gas_busmap': False, # False or path/to/ch4_busmap.csv
        'line_length_factor': 1, #
        'remove_stubs': False, # remove stubs before kmeans clustering
        'use_reduced_coordinates': False, #
        'bus_weight_tocsv': None, # None or path/to/bus_weight.csv
        'bus_weight_fromcsv': None, # None or path/to/bus_weight.csv
        'n_init': 10, # affects clustering algorithm, only change when neccesary
        'max_iter': 100, # affects clustering algorithm, only change when neccesary
        'tol': 1e-6,}, # affects clustering algorithm, only change when neccesary
    'sector_coupled_clustering': {
        'active': True, # choose if clustering is activated
        'carrier_data': { # select carriers affected by sector coupling
            'H2_ind_load': {
                'base': ['H2_grid'],
                'strategy': 'consecutive'
            },
            'central_heat': {
                'base': ['CH4', 'AC'],
                'strategy': 'consecutive'
            },
            'rural_heat': {
                'base': ['CH4', 'AC'],
                'strategy': 'consecutive'
            },
        },
    },
    'network_clustering_ehv': False,  # clustering of HV buses to EHV buses.
    'disaggregation': None,  # None, 'mini' or 'uniform'
    'snapshot_clustering': {
        'active': False, # choose if clustering is activated
        'method':'typical_periods', # 'typical_periods' or 'segmentation'
        'extreme_periods': 'None', # optional adding of extreme period
        'how': 'daily', # type of period - only relevant for 'typical_periods'
        'storage_constraints': '', # additional constraints for storages  - only relevant for 'typical_periods'
        'n_clusters': 5, #  number of periods - only relevant for 'typical_periods'
        'n_segments': 5}, # number of segments - only relevant for segmentation
    # Simplifications:
    'skip_snapshots': False, # False or number of snapshots to skip
    'branch_capacity_factor': {'HV': 0.5, 'eHV': 0.7},  # p.u. branch derating
    'load_shedding': False,  # meet the demand at value of loss load cost
    'foreign_lines': {'carrier': 'AC', # 'DC' for modeling foreign lines as links
                      'capacity': 'osmTGmod'}, # 'osmTGmod', 'ntc_acer' or 'thermal_acer'
    'comments': None}

def run_etrago(args, json_path, path, number):
    """The etrago function works with following arguments:


    Parameters
    ----------

    db : str
        ``'oedb'``,
        Name of Database session setting stored in *config.ini* of *.egoio*

    gridversion : NoneType or str
        ``'v0.4.6'``,
        Name of the data version number of oedb: state ``'None'`` for
        model_draft (sand-box) or an explicit version number
        (e.g. 'v0.4.6') for the grid schema.

    method : dict
        {'type': 'lopf', 'n_iter': 5, 'pyomo': True},
        Choose 'lopf' for 'type'. In case of extendable lines, several lopfs
        have to be performed. Choose either 'n_init' and a fixed number of
        iterations or 'thershold' and a threashold of the objective function as
        abort criteria.
        Set 'pyomo' to False for big optimization problems, currently only
        possible when solver is 'gurobi'.

    pf_post_lopf :dict
        {'active': True, 'add_foreign_lopf': True, 'q_allocation': 'p_nom'},
        Option to run a non-linear power flow (pf) directly after the
        linear optimal power flow (and thus the dispatch) has finished.
        If foreign lines are modeled as DC-links (see foreign_lines), results
        of the lopf can be added by setting 'add_foreign_lopf'.
        Reactive power can be distributed either by 'p_nom' or 'p'.

    start_snapshot : int
        1,
        Start hour of the scenario year to be calculated.

    end_snapshot : int
        2,
        End hour of the scenario year to be calculated.
        If temporal clustering is used, the selected snapshots should cover
        whole days.

    solver : str
        'glpk',
        Choose your preferred solver. Current options: 'glpk' (open-source),
        'cplex' or 'gurobi'.

    solver_options: dict
        Choose settings of solver to improve simulation time and result.
        Options are described in documentation of choosen solver.

    model_formulation: str
        'angles'
        Choose formulation of pyomo-model.
        Current options: angles, cycles, kirchhoff, ptdf

    scn_name : str
        'eGon2035',
        Choose your scenario. Currently, there are two different
        scenarios: 'eGon2035', 'eGon100RE'.

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
           'BE_NO_NEP 2035' includes planned lines to Belgium and Norway and
           adds BE and NO as electrical neighbours

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

    csv_export : obj
        False,
        State if and where you want to save results as csv files.Options:
        False or '/path/tofolder'.

    extendable : list
        ['network', 'storages'],
        Choose components you want to optimize.
        Settings can be added in /tools/extendable.py.
        The most important possibilities:
            'as_in_db': leaves everything as it is defined in the data coming
                        from the database
            'network': set all lines, links and transformers extendable
            'german_network': set lines and transformers in German grid
                            extendable
            'foreign_network': set foreign lines and transformers extendable
            'transformers': set all transformers extendable
            'overlay_network': set all components of the 'scn_extension'
                               extendable
            'storages': allow to install extendable storages
                        (unlimited in size) at each grid node in order to meet
                        the flexibility demand.
            'network_preselection': set only preselected lines extendable,
                                    method is chosen in function call

    generator_noise : bool or int
        State if you want to apply a small random noise to the marginal costs
        of each generator in order to prevent an optima plateau. To reproduce
        a noise, choose the same integer (seed number).

    extra_functionality : dict or None
        None,
        Choose extra functionalities and their parameters for PyPSA-model.
        Settings can be added in /tools/constraints.py.
        Current options are:
            'max_line_ext': float
                Maximal share of network extension in p.u.
            'min_renewable_share': float
                Minimal share of renewable generation in p.u.
            'cross_border_flow': array of two floats
                Limit cross-border-flows between Germany and its neigbouring
                countries, set values in p.u. of german loads in snapshots
                for all countries
                (positiv: export from Germany)
            'cross_border_flows_per_country': dict of cntr and array of floats
                Limit cross-border-flows between Germany and its neigbouring
                countries, set values in p.u. of german loads in snapshots
                for each country
                (positiv: export from Germany)
            'max_curtailment_per_gen': float
                Limit curtailment of all wind and solar generators in Germany,
                values set in p.u. of generation potential.
            'max_curtailment_per_gen': float
                Limit curtailment of each wind and solar generator in Germany,
                values set in p.u. of generation potential.
            'capacity_factor': dict of arrays
                Limit overall energy production for each carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_gen': dict of arrays
                Limit overall energy production for each generator by carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_cntr': dict of dict of arrays
                Limit overall energy production country-wise for each carrier,
                set upper/lower limit in p.u.
            'capacity_factor_per_gen_cntr': dict of dict of arrays
                Limit overall energy production country-wise for each generator
                by carrier, set upper/lower limit in p.u.

    network_clustering_kmeans : dict
         {'active': True, 'n_clusters': 30, 'n_clusters_gas': 30,
          'kmeans_busmap': False, 'line_length_factor': 1.25,
          'remove_stubs': False, 'use_reduced_coordinates': False,
          'bus_weight_tocsv': None, 'bus_weight_fromcsv': None, 'n_init': 10,
          'max_iter': 300, 'tol': 1e-4, 'n_jobs': 1},
        State if you want to apply a clustering of all network buses down to
        only ``'n_clusters'`` buses. The weighting takes place considering
        generation and load at each node. ``'n_clusters_gas'`` refers to the
        total amount of gas buses after clustering. Note, that the number of
        gas buses of Germanies neighboring countries is not modified. in this
        process.
        With ``'kmeans_busmap'`` you can choose if you want to load cluster
        coordinates from a previous run.
        Option ``'remove_stubs'`` reduces the overestimating of line meshes.
        The other options affect the kmeans algorithm and should only be
        changed carefully, documentation and possible settings are described
        in sklearn-package (sklearn/cluster/k_means_.py).
        This function doesn't work together with ``'line_grouping = True'``.

    sector_coupled_clustering : nested dict
        {'active': True, 'carrier_data': {
         'H2_ind_load': {'base': ['H2_grid']},
         'central_heat': {'base': ['CH4']},
         'rural_heat': {'base': ['CH4']}}
        }
        State if you want to apply clustering of sector coupled carriers, such
        as central_heat or rural_heat. The approach builds on already clustered
        buses (e.g. CH4 and AC) and builds clusters around the topology of the
        buses with carrier ``'base'`` for all buses of a specific carrier, e.g.
        ``'H2_ind_load'``.

    network_clustering_ehv : bool
        False,
        Choose if you want to cluster the full HV/EHV dataset down to only the
        EHV buses. In that case, all HV buses are assigned to their closest EHV
        sub-station, taking into account the shortest distance on power lines.

    snapshot_clustering : dict
        {'active': False, 'method':'typical_periods', 'how': 'daily',
         'storage_constraints': '', 'n_clusters': 5, 'n_segments': 5},
        State if you want to apply a temporal clustering and run the optimization
        only on a subset of snapshot periods.
        You can choose between a method clustering to typical periods, e.g. days
        or a method clustering to segments of adjacent hours.
        With ``'how'``, ``'storage_constraints'`` and ``'n_clusters'`` you choose
        the length of the periods, constraints considering the storages and the number
        of clusters for the usage of the method typical_periods.
        With ``'n_segments'`` you choose the number of segments for the usage of
        the method segmentation.

    branch_capacity_factor : dict
        {'HV': 0.5, 'eHV' : 0.7},
        Add a factor here if you want to globally change line capacities
        (e.g. to "consider" an (n-1) criterion or for debugging purposes).

    load_shedding : bool
        False,
        State here if you want to make use of the load shedding function which
        is helpful when debugging: a very expensive generator is set to each
        bus and meets the demand when regular
        generators cannot do so.

    foreign_lines : dict
        {'carrier':'AC', 'capacity': 'osmTGmod}'
        Choose transmission technology and capacity of foreign lines:
            'carrier': 'AC' or 'DC'
            'capacity': 'osmTGmod', 'ntc_acer' or 'thermal_acer'

    comments : str
        None

    Returns
    -------
    network : `pandas.DataFrame<dataframe>`
        eTraGo result network based on `PyPSA network
        <https://www.pypsa.org/doc/components.html#network>`_
    """
    
    # to procede in loop
    if path == 'skip_snapshots':
        args['skip_snapshots'] = number
    elif path == 'typical_days' or path == 'typical_hours' or path == 'typical_weeks' or path == 'segmentation':
        args['snapshot_clustering']['active'] = True
        if args['snapshot_clustering']['method'] == 'typical_periods':
            args['snapshot_clustering']['n_clusters'] = number
        elif args['snapshot_clustering']['method'] == 'segmentation':
            args['snapshot_clustering']['n_segments'] = number
        
    path = path +'/'+ str(number) +'/'
    
    etrago = Etrago(args, json_path)

    # import network from database
    etrago.build_network_from_db()
    
    # hot fixes
    
    etrago.network.lines.type = ''
    etrago.network.lines.carrier.fillna('AC', inplace=True)
    etrago.network.buses.v_mag_pu_set.fillna(1., inplace=True)
    etrago.network.loads.sign = -1
    etrago.network.links.capital_cost.fillna(0, inplace=True)
    etrago.network.links.p_nom_min.fillna(0, inplace=True)
    etrago.network.transformers.tap_ratio.fillna(1., inplace=True)
    etrago.network.stores.e_nom_max.fillna(np.inf, inplace=True)
    etrago.network.links.p_nom_max.fillna(np.inf, inplace=True)
    etrago.network.links.efficiency.fillna(1., inplace=True)
    etrago.network.links.marginal_cost.fillna(0., inplace=True)
    etrago.network.links.p_min_pu.fillna(0., inplace=True)
    etrago.network.links.p_max_pu.fillna(1., inplace=True)
    etrago.network.links.p_nom.fillna(0.1, inplace=True)
    etrago.network.storage_units.p_nom.fillna(0, inplace=True)
    etrago.network.stores.e_nom.fillna(0, inplace=True)
    etrago.network.stores.capital_cost.fillna(0, inplace=True)
    etrago.network.stores.e_nom_max.fillna(np.inf, inplace=True)
    etrago.network.storage_units.efficiency_dispatch.fillna(1., inplace=True)
    etrago.network.storage_units.efficiency_store.fillna(1., inplace=True)
    etrago.network.storage_units.capital_cost.fillna(0., inplace=True)
    etrago.network.storage_units.p_nom_max.fillna(np.inf, inplace=True)
    etrago.network.storage_units.standing_loss.fillna(0., inplace=True)
    etrago.network.storage_units.lifetime = np.inf
    etrago.network.lines.v_ang_min.fillna(0., inplace=True)
    etrago.network.links.terrain_factor.fillna(1., inplace=True)
    etrago.network.lines.v_ang_max.fillna(1., inplace=True)
    etrago.network.lines.lifetime = 40 # only temporal fix until either the 
                                       # PyPSA network clustering function 
                                       # is changed (taking the mean) or our 
                                       # data model is altered, which will 
                                       # happen in the next data creation run
    etrago.network.lines.x /= (etrago.network.lines.v_nom*1000)**2 / (100 * 10e6)
        # only temporal fix until the data model is altered, which will happen in the next data creation run
    
    etrago.network.generators_t['p_max_pu'].mask(etrago.network.generators_t['p_max_pu']<0.001, 0, inplace=True)                                       
    for t in etrago.network.iterate_components():
        if 'p_nom_max' in t.df:
            t.df['p_nom_max'].fillna(np.inf, inplace=True)
    
    for t in etrago.network.iterate_components():
        if 'p_nom_min' in t.df:
                        t.df['p_nom_min'].fillna(0., inplace=True)
                        
    # marginale Kosten von Generatoren
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier=='gas'] = 41
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier=='lignite'] = 17
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier=='oil'] = 69
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier=='nuclear'] = 6
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier.str.contains('biomass')] = 41
    etrago.network.generators.marginal_cost[etrago.network.generators.carrier=='other_non_renewable'] = 69

    etrago.adjust_network()

    # Set marginal costs for gas feed-in
    etrago.network.generators.marginal_cost[
        etrago.network.generators.carrier=='CH4']+= 25.6+0.201*76.5

    # ehv network clustering
    #etrago.ehv_clustering()

    # k-mean clustering
    etrago.kmean_clustering()
    
    etrago.kmean_clustering_gas()
    
    # more hot fixes
    
    for t in etrago.network.iterate_components():
        if 'p_nom_max' in t.df:
            t.df['p_nom_max'].fillna(np.inf, inplace=True)

    for t in etrago.network.iterate_components():
        if 'p_nom_min' in t.df:
                t.df['p_nom_min'].fillna(0., inplace=True)
    
    for t in etrago.network.iterate_components():
        if 'marginal_cost' in t.df:
            np.random.seed(174)
            t.df['marginal_cost'] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)
    
    etrago.args['load_shedding']=True
    etrago.load_shedding()

    ###########################################################################
    
    # preparations for 2-Level Approach
    
    # save original timeseries
    original_snapshots = etrago.network.snapshots
    original_weighting = etrago.network.snapshot_weightings
    
    # save format for dispatch using original timeseries
    
    loads_pset = etrago.network.loads_t.p_set.copy()
    loads_qset = etrago.network.loads_t.q_set.copy()
    loads_p = etrago.network.loads_t.p.copy()
    loads_q = etrago.network.loads_t.q.copy()
    
    gen_pmax = etrago.network.generators_t.p_max_pu.copy()
    gen_p = etrago.network.generators_t.p.copy()
    
    lines_lower = etrago.network.lines_t.mu_lower.copy()
    lines_upper = etrago.network.lines_t.mu_upper.copy()
    lines_p0 = etrago.network.lines_t.p0.copy()
    lines_p1 = etrago.network.lines_t.p1.copy()
    lines_q0 = etrago.network.lines_t.q0.copy()
    lines_q1 = etrago.network.lines_t.q1.copy()
    
    links_pmin = etrago.network.links_t.p_min_pu.copy()
    links_pmax = etrago.network.links_t.p_max_pu.copy()
    links_eff = etrago.network.links_t.efficiency.copy()
    links_lower = etrago.network.links_t.mu_lower.copy()
    links_upper = etrago.network.links_t.mu_upper.copy()
    links_p0 = etrago.network.links_t.p0.copy()
    links_p1 = etrago.network.links_t.p1.copy()
    
    store_emin = etrago.network.stores_t.e_min_pu.copy()
    store_emax = etrago.network.stores_t.e_max_pu.copy()
    store_e = etrago.network.stores_t.e.copy()
    store_p = etrago.network.stores_t.p.copy()
    store_q = etrago.network.stores_t.q.copy()
    
    stun_p = etrago.network.storage_units_t.p.copy()
    stun_p_dispatch = etrago.network.storage_units_t.p_dispatch.copy()
    stun_p_store = etrago.network.storage_units_t.p_store.copy()
    stun_spill = etrago.network.storage_units_t.spill.copy()
    stun_state_of_charge = etrago.network.storage_units_t.state_of_charge.copy()
    
    bus_p = etrago.network.buses_t.p.copy()
    bus_price = etrago.network.buses_t.marginal_price.copy()
    bus_vang = etrago.network.buses_t.v_ang.copy()
    bus_vmag = etrago.network.buses_t.v_mag_pu.copy()
    
    ###########################################################################
    
    print(' ')
    print('Start Time Series Aggregation')
    t1 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # skip snapshots    
    etrago.skip_snapshots()
    if args['skip_snapshots'] != False:
        print(' ')
        print(etrago.network.snapshots)
        print(' ')

    # snapshot clustering
    etrago.snapshot_clustering()
    
    print(' ')
    print('Stop Time Series Aggregation')
    t2 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    etrago.args['csv_export'] = path+'results/level1/'
    #etrago.args['lpfile'] = path+'lp-level1.lp' 

    print(' ')
    print('Start LOPF Level 1')
    t3 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')

    # start linear optimal powerflow calculations
    etrago.lopf()
    
    print(' ')
    print('Stop LOPF Level 1')
    t4 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # save results of this optimization
    etrago.export_to_csv(path+'Level-1')
    
    # calculate central etrago results
    etrago.calc_results()
    print(' ')
    print(etrago.results)
    etrago.results.to_csv(path+'results-1')
    print(' ') 
    
    ###########################################################################

    # LOPF Level 2
    
    etrago.args['snapshot_clustering']['active']=False
    etrago.args['skip_snapshots']=False
    
    # use original timeseries
    
    etrago.network.snapshots = original_snapshots
    etrago.network.snapshot_weightings = original_weighting
    
    # drop dispatch from LOPF1
    
    etrago.network.loads_t.p_set = loads_pset
    etrago.network.loads_t.q_set = loads_qset
    etrago.network.loads_t.p = loads_p
    etrago.network.loads_t.q = loads_q
    
    etrago.network.generators_t.p_max_pu = gen_pmax
    etrago.network.generators_t.p = gen_p
    
    etrago.network.lines_t.mu_lower = lines_lower
    etrago.network.lines_t.mu_upper = lines_upper
    etrago.network.lines_t.p0 = lines_p0
    etrago.network.lines_t.p1 = lines_p1
    etrago.network.lines_t.q0 = lines_q0
    etrago.network.lines_t.q1 = lines_q1
    
    etrago.network.links_t.p_min_pu = links_pmin
    etrago.network.links_t.p_max_pu = links_pmax
    etrago.network.links_t.efficiency = links_eff
    etrago.network.links_t.mu_lower = links_lower
    etrago.network.links_t.mu_upper = links_upper
    etrago.network.links_t.p0 = links_p0
    etrago.network.links_t.p1 = links_p1
    
    etrago.network.stores_t.e_min_pu = store_emin
    etrago.network.stores_t.e_max_pu = store_emax
    etrago.network.stores_t.e = store_e
    etrago.network.stores_t.p = store_p
    etrago.network.stores_t.q = store_q
    
    etrago.network.storage_units_t.p = stun_p
    etrago.network.storage_units_t.p_dispatch = stun_p_dispatch
    etrago.network.storage_units_t.p_store = stun_p_store
    etrago.network.storage_units_t.spill = stun_spill
    etrago.network.storage_units_t.state_of_charge = stun_state_of_charge
    
    etrago.network.buses_t.p = bus_p
    etrago.network.buses_t.marginal_price = bus_price
    etrago.network.buses_t.v_ang = bus_vang
    etrago.network.buses_t.v_mag_pu = bus_vmag
    
    # use network and storage expansion from LOPF 1
        
    etrago.args['extendable'] = []
    
    etrago.network.lines['s_nom'] = etrago.network.lines['s_nom_opt']
    etrago.network.lines['s_nom_extendable'] = False
    
    etrago.network.transformers['s_nom'] = etrago.network.transformers['s_nom_opt']
    etrago.network.transformers.s_nom_extendable = False
    
    etrago.network.storage_units['p_nom'] = etrago.network.storage_units['p_nom_opt']
    etrago.network.storage_units['p_nom_extendable'] = False
    
    etrago.network.stores['e_nom'] = etrago.network.stores['e_nom_opt']
    etrago.network.stores['e_nom_extendable'] = False
    
    etrago.network.links['p_nom'] = etrago.network.links['p_nom_opt']
    etrago.network.links['p_nom_extendable'] = False
    
    # use original timeseries
    
    etrago.network.snapshots = original_snapshots
    etrago.network.snapshot_weightings = original_weighting
    
    etrago.args['csv_export'] = path+'results/level2/'
    #etrago.args['lpfile'] = path+'lp-level2.lp'
    
    print(' ')
    print('Start LOPF Level 2')
    t5 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # optimization of dispatch with complex timeseries
    etrago.lopf()
    
    print(' ')
    print('Stop LOPF Level 2')
    t6 = datetime.datetime.now()
    print(datetime.datetime.now())
    print(' ')
    
    # save results of this optimization
    etrago.export_to_csv(path+'Level-2')
    
    # calculate central etrago results
    etrago.calc_results()
    print(' ')
    print(etrago.results)
    etrago.results.to_csv(path+'results-2')
    print(' ')
    
    ###########################################################################
    
    t = pd.Series([t1, t2, t3, t4, t5, t6]) 
    t.to_csv(path+'time')

    # check if should be combined with etrago.lopf()
    # needs to be adjusted for new sectors
    # etrago.pf_post_lopf()

    # spatial disaggregation
    # needs to be adjusted for new sectors
    # etrago.disaggregation()

    # calculate central etrago results
    #etrago.calc_results()

    return etrago

###############################################################################

# Zeithorizont
args['start_snapshot'] = 1
args['end_snapshot'] = 8760 ###

# räumliche Auflösung
args['network_clustering_kmeans']['active'] = True
args['network_clustering_kmeans']['n_clusters'] = 50
args['network_clustering_kmeans']['n_clusters_gas'] = 30
args['network_clustering_kmeans']['kmeans_busmap'] = 'kmeans_busmap_50_result.csv' # False
args['network_clustering_kmeans']['kmeans_gas_busmap'] = 'kmeans_ch4_busmap_30_result.csv' # False

# Lösbarkeit
args['load_shedding'] = False # wird später extra aufgerufen 
args['method']['pyomo'] = True
args['method']['n_iter'] = 4
args['solver_options'] =  {
        'BarConvTol': 1.e-5,
        'FeasibilityTol': 1.e-5,
        'method':2,
        'crossover':0,
        'logFile': 'solver_50ac_30gas.log'
        }
        
# zeitliche Auflösung
args['snapshot_clustering']['active'] = False
args['snapshot_clustering']['method'] = 'typical_periods' # 'typical_periods', 'segmentation'
args['snapshot_clustering']['extreme_periods'] = 'None' # 'None', 'append', 'replace_cluster_center'
args['snapshot_clustering']['how'] = 'daily' # 'daily', 'weekly', 'monthly'
args['snapshot_clustering']['storage_constraints'] = 'soc_constraints' # 'soc_constraints', 'soc_constraints_simplified', ''
args['skip_snapshots'] = True

###############################################################################

skip_snapshots = [10, 5, 3] 

segmentation = [880, 1700, 2900] 

if args['snapshot_clustering']['active'] == True and args['skip_snapshots'] == True:
    raise ValueError("Decide for temporal aggregation method!")
elif args['snapshot_clustering']['active'] == False and args['skip_snapshots'] == False:
    loop = [0]
    path = 'original'
if args['snapshot_clustering']['active'] == True:
    if args['snapshot_clustering']['method'] == 'typical_periods':
        loop = typical_days
        path = 'typical_days'
    elif args['snapshot_clustering']['method'] == 'segmentation':
        loop = segmentation
        path = 'segmentation'
elif args['skip_snapshots'] == True:
    loop = skip_snapshots
    path = 'skip_snapshots'
    
for no in loop:

    old_stdout = sys.stdout
    path_log = path +'/'+ str(no)
    os.makedirs(path_log, exist_ok=True)
    log_file = open(path_log+'/console.log',"w")
    sys.stdout = log_file
    
    print(' ')
    print('pyomo:')
    print(args['method'])
    print('snapshot_clustering:')
    print(args['snapshot_clustering'])
    print(args['snapshot_clustering']['method'])
    print(args['snapshot_clustering']['how'])
    print(args['snapshot_clustering']['extreme_periods'])
    print('skip_snapshots:')
    print(args['skip_snapshots'])
    print(' ') 
    
    print(' ')
    print('Calculation using: '+ path)
    print('with number = '+ str(no))
    print(' ')
    
    etrago = run_etrago(args=args, json_path=None, path = path, number=no)
    
    sys.stdout = old_stdout
    log_file.close()

###############################################################################
